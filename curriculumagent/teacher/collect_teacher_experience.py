"""In this file, we analyse the action library generated by Teacher1 and Teacher2.

Through

    1. filtering out actions that decrease rho less effectively
    2. filtering out actions that occur less frequently
    3. filtering out "do nothing" action
    4. add your filtering rules...,

We obtain an action space in the form of numpy array.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
import logging
from pathlib import Path
from typing import Union, Collection, List, Optional

import defopt
import grid2op
import numpy as np
import pandas as pd
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from matplotlib import pyplot as plt

from curriculumagent.teacher.submodule.common import affected_substations
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction


def read_experience(experience_files: Union[Path, Collection[Path]]) -> pd.DataFrame:
    """Read teacher_experience data from csv file and concat it if needed.

    Args:
        experience_files:Either one file or multiple files with teacher_experience data.

    Returns:
        Dataframe with teacher_experience.

    """
    data = None
    for exp_file in experience_files:
        if data is None:
            data = pd.read_csv(exp_file, on_bad_lines="skip")
        else:
            data = pd.concat((data, pd.read_csv(exp_file, on_bad_lines="skip")))
    return data


def filter_good_experience(data: pd.DataFrame) -> pd.DataFrame:
    """Filter actions based on criteria.

    The actions are either filtered if:
        1. The action decreases the rho by less than 2%
        2. The action does nothing

    Args:
        data: Input of teacher_experience.

    Returns:
        Reduced dataframe with filtered action/teacher_experience.

    """
    good_samples = data["rho_improvement"] > 0.02
    return data[good_samples]


def rank_actions_simple(
        data: pd.DataFrame, env: BaseEnv, best_n: int = 416, plot_choice: Optional[bool] = False
) -> List[BaseAction]:
    """Rank the given teacher_experience data by counting and sorting them and cutting of the best_n actions from the top.

    Args:
        data: Loaded teacher_experience data.
        env: The environment used to generate the teacher_experience data.
        best_n: The cutoff point determining the maximum number of actions returned.
        plot_choice: Whether the frequency should be plotted.

    Returns:
        The ranked actions as a list.
    """
    decode_action = np.vectorize(lambda act: EncodedTopologyAction.decode_action(act, env))

    # Resort actions
    best_action_freq = data["best_action"].value_counts()
    logging.info(f"There are {len(best_action_freq)} unique actions")

    if plot_choice:
        plt.plot(best_action_freq.values)
        plt.vlines(best_n, ymin=0, ymax=best_action_freq.values.max(), color="red")
        plt.title("Sorted frequency of the actions")
        plt.legend(["Sorted Frequency", "best_n"])

    best_actions = best_action_freq[:best_n]
    best_actions_array = decode_action(best_actions.index.to_numpy())

    return list(best_actions_array)


def rank_actions_tuples(
        data: pd.DataFrame, env: BaseEnv, min_freq_unitary: int = 400, min_freq_tuple: int = 6
) -> List[BaseAction]:
    """Rank the given teacher_experience by sorting them according to their ranking (in the top125 actions list),
    then splitting them into single and tuple actions and mixing both together.

    Args:
        data: Loaded teacher_experience data.
        env: The environment used to generate the teacher_experience of data.
        min_freq_unitary: The minimum frequency of unitary actions to be included in the actionspace.
        min_freq_tuple: The minimum frequency of tuple actions to be included in the actionspace.

    Returns:
        The generated actions as a list.
        The first actions are the tuple actions whereas the later ones are unitary actions.

    """
    decode_action = np.vectorize(lambda act: EncodedTopologyAction.decode_action(act, env))

    # Resort actions based on their ranking and frequency
    top_actions = data.iloc[:, 4:]
    actions = {}

    logging.info("Sorting actions...")
    for rank in range(125):
        top_i = top_actions.iloc[:, rank]
        actions[rank] = top_i

    all_actions = []
    for rank, actions in actions.items():
        for action in actions:
            all_actions.append((rank, action))
    all_actions_df = pd.DataFrame(all_actions, columns=["rank", "action"])

    logging.info("Remove duplicate actions...")
    best_action_freq = all_actions_df["action"].value_counts()

    best_actions = best_action_freq[best_action_freq >= 1]
    best_actions_array = decode_action(best_actions.index.to_numpy())

    best_actions_df = pd.DataFrame({"action": best_actions_array, "count": best_actions})

    logging.info("Counting substations...")
    np_count_actions = np.vectorize(lambda act: len(affected_substations(act)))
    best_actions_df["subs_affected"] = np_count_actions(best_actions_df["action"])

    unique, counts = np.unique(best_actions_df["subs_affected"], return_counts=True)
    logging.info(f"Number of actions: {unique}, counts: {counts}")

    all_unitary_actions = best_actions_df[best_actions_df["subs_affected"] == 1]
    all_tuple_actions = best_actions_df[best_actions_df["subs_affected"] == 2]

    logging.info(f"Filtering by frequency threshold {min_freq_unitary}(unitary) and {min_freq_tuple}(tuple)")
    # Mix and match
    best_unitary_actions = all_unitary_actions[all_unitary_actions["count"] >= min_freq_unitary]
    best_tuple_actions = all_tuple_actions[all_tuple_actions["count"] >= min_freq_tuple]
    mixed_actions = pd.concat([best_tuple_actions, best_unitary_actions])
    logging.info(
        f"actions: {len(mixed_actions)} = {len(best_tuple_actions)}(tuple) + {len(best_unitary_actions)}(unitary)"
    )

    return list(mixed_actions["action"])


def save_actionspace_binbinchen(save_path: Path, actions: List[BaseAction]):
    """Save the given actionspace as a numpy array file (which was used by the binbinchen tutor).

    Args:
        save_path: The path where to save the file to.
        actions: The actionspace to save.

    Returns:
        None.
    """
    sample = actions[0].to_vect()
    action_space = np.zeros((len(actions), sample.shape[0]))
    for i, action in enumerate(actions):
        action_space[i] = action.to_vect()

    np.save(str(save_path), action_space)
    logging.info(f"Save an action space with the size of {action_space.shape[0]:d}")


def make_unitary_actionspace(
        action_space_file_path: Path,
        experience_csv_files: List[Path],
        env_name_path: Union[Path, str],
        best_n: int,
        plot_choice: Optional[bool] = True,
):
    """Using the provided teacher_experience files to generate a unitary space file for the tutor.

    Args:
        action_space_file_path: Where to save the action space file containing the best actions.
        experience_csv_files: The csv files containing the teacher_experience from the teachers.
        env_name_path: The name or path to the environment that is used for teacher_experience generation.
        best_n: The best_n actions to keep.
        plot_choice: Whether the selection process should be plotted for the visualization.

    Returns:
        None, but data is saved under action_space_file_path.
    """

    data = read_experience(experience_csv_files)
    logging.info(f"Read {len(data)} teacher_experience samples from {experience_csv_files}")
    data = filter_good_experience(data)
    logging.info(f"Kept {len(data)} after filtering")

    env = grid2op.make(env_name_path)

    actions = rank_actions_simple(data, env, best_n=best_n, plot_choice=plot_choice)
    save_actionspace_binbinchen(action_space_file_path, actions)
    logging.info(f"Saved {len(actions)} actions after ranking with best_n={best_n} to {action_space_file_path}")


def make_tuple_actionspace(
        action_space_file_path: Path,
        experience_csv_files: List[Path],
        env_name_path: Union[Path, str],
        min_freq_unitary: int,
        min_freq_tuple: int,
):
    """Generate a tuple action space for the tutor using the provided teacher_experience files.

    Args:
        action_space_file_path: Where to save the action space file containing the best actions.
        experience_csv_files: The csv files containing the teacher_experience from the teachers.
        env_name_path: The name or path to the environment that is used for teacher_experience generation.
        min_freq_unitary: The minimum frequency of unitary actions to be included in the actionspace.
        min_freq_tuple: The minimum frequency of tuple actions to be included in the actionspace.

    Returns:
        None, but data is saved under action_space_file_path.
    """

    data = read_experience(experience_csv_files)
    data = filter_good_experience(data)

    env = grid2op.make(env_name_path)

    actions = rank_actions_tuples(data, env, min_freq_unitary=min_freq_unitary, min_freq_tuple=min_freq_tuple)
    save_actionspace_binbinchen(action_space_file_path, actions)
    logging.info(f"Saved {len(actions)} actions to {action_space_file_path}")


def make_seq_actionspace(
        action_space_file_path: Path,
        experience_csv_files: List[Path],
        env_name_path: Union[Path, str],
        seq_length: int,
        best_n: int,
        plot_choice: Optional[bool] = True,
):
    """Generate a tuple action space for the tutor using the provided teacher_experience files.

    Args:
        action_space_file_path: Where to save the action space file containing the best actions.
        experience_csv_files: The csv files containing the teacher_experience from the teachers.
        env_name_path: The name or path to the environment that is used for teacher_experience generation.
        seq_length: Length of the sequential actions that should be kept. E.g. if seq_length=3 than only triple actions.
        are selected.
        best_n: The best_n actions to keep.
        plot_choice: Whether the selection process should be plotted for the visualization.


    Returns:
        None, but data is saved under action_space_file_path.
    """

    data = read_experience(experience_csv_files)
    data = filter_good_experience(data)

    env = grid2op.make(env_name_path)

    actions = rank_actions_seq(data, env, seq=seq_length, best_n=best_n, plot_choice=plot_choice)
    save_actionspace_binbinchen(action_space_file_path, actions)
    logging.info(f"Saved {len(actions)} actions to {action_space_file_path}")


def rank_actions_seq(data: pd.DataFrame, env: BaseEnv, seq: int = 3, best_n: int = 200, plot_choice=False):
    """Rank the given teacher_experience data by counting and sorting them and cutting of the best_n actions from the top.
    This method is used for sequential actions and only selects a specific length of actions.

    Args:
        data: Loaded teacher_experience data.
        env: The environment used to generate the teacher_experience of data.
        seq: Number of actions that has to be within the set.
        best_n: The cutoff point determining the maximum number of actions returned.
        plot_choice: Whether the frequency should be plotted.

    Returns:
        The ranked actions as a list.
    """
    decode_action = np.vectorize(lambda act: EncodedTopologyAction.decode_action(act, env))

    # Resort actions
    best_action_freq = data["best_action"].value_counts()
    logging.info(f"There are {len(best_action_freq)} unique actions")

    if plot_choice:
        plt.plot(best_action_freq.values)
        plt.vlines(best_n, ymin=0, ymax=best_action_freq.values.max(), color="red")
        plt.title("Sorted frequency of the actions")
        plt.legend(["Sorted Frequency", "best_n"])

    best_actions_array = list(decode_action(best_action_freq.index.to_numpy()))
    best_actions_array = [a for a in best_actions_array if len(a.as_dict()["set_bus_vect"]["modif_subs_id"]) == seq]
    best_actions_array = best_actions_array[:best_n]

    return best_actions_array

