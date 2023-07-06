"""This module/file contains various implementations of greedy searches of topology actions.

"""


import logging
from random import randint
from typing import List, Tuple

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from tqdm import tqdm

from curriculumagent.common.utilities import find_best_line_to_reconnect, is_valid
from curriculumagent.teacher.submodule.common import affected_substations
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction


def topology_search_topk(
        env: BaseEnv, obs: BaseObservation, all_actions: List[BaseAction], top_k=1000
) -> List[Tuple[float, BaseAction]]:
    """Perform a search over all topology actions of the given environment and return the top_k best ones, sorted by
    their rho improvement.

    Args:
        env: The environment to search in.
        obs: The current observation of the given environment.
        all_actions: All actions to simulate and rank.
        top_k: How many final actions should be in the returned list.

    Returns:
        A list containing tuples of the rho_improvement and action.
    """
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    old_max_rho = min_rho  # save old max_rho for comparison

    performed_actions = []

    for action in all_actions:
        valid, reason = env._game_rules(action, env)  # noqa
        if not valid:
            logging.debug(f"Found action is not valid: {reason}")
            continue

        obs_, _, done, _ = obs.simulate(action)
        if not done:
            new_max_rho = obs_.rho.max()
            rho_improvement = old_max_rho - new_max_rho
            performed_actions.append((rho_improvement, action))

    best_actions = sorted(performed_actions, key=lambda tup: tup[0], reverse=True)[:top_k]

    return best_actions


def topology_search_tuples(
        env: BaseEnv,
        best_unitary_actions: List[BaseAction],
        all_unitary_actions: List[BaseAction],
        all_unitary_actions_uncommon: List[BaseAction],
        sample_size: int = 1000,
        most_common_station: int = 16,
        show_progress: bool = False,
) -> List[Tuple[float, EncodedTopologyAction]]:
    """Search for good actions that occur on one or two substations at the same time.

    Note: In the l2rpn_neurips_2020_track1_small station 16 is heavily dominating as a station that is affected
    by unitary actions, hence we made a special case for single stations here.

    Args:
        env: The environment to use for the search.
        best_unitary_actions: List of the best unitary actions, used to find tuple actions.
        all_unitary_actions: List of all unitary actions.
        all_unitary_actions_uncommon: The list of all actions not affecting the most
                                      common station with id of most_common_station.
        sample_size: How many other actions should be sampled from all actions to combine with best actions.
        most_common_station: The id of the most common station in the network/environment.
        show_progress: Set to true to show a progress bar.

    Returns:
        A list of tuple actions ranked by their rho_improvement.
    """
    obs: BaseObservation = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    old_max_rho = min_rho  # save old max_rho for comparison
    actions_results: List[Tuple[float, EncodedTopologyAction]] = []

    # Unitary search
    # For each of these actions simulate the unitary effect and write out rho_max.
    for action in tqdm(best_unitary_actions, disable=not show_progress):
        valid, reason = env._game_rules(action, env)
        if not valid:
            logging.debug(f"Found action is not valid: {reason}")
            continue

        obs_, _, done, _ = obs.simulate(action)
        if not done:
            new_max_rho = obs_.rho.max()
            rho_improvement: float = old_max_rho - new_max_rho
            actions_results.append((rho_improvement, EncodedTopologyAction(action)))

    # Tuple search
    for first_action in tqdm(best_unitary_actions, disable=not show_progress):
        first_affected_substations = affected_substations(first_action)
        assert len(first_affected_substations) == 1, "first action is unitary"
        first_affected_substation = first_affected_substations[0]

        affected_substation_id = int(first_action.as_dict()["set_bus_vect"]["modif_subs_id"][0])
        if affected_substation_id == most_common_station:
            # (2) if the first action is from substation 16 then loop through all other actions.
            for second_action in all_unitary_actions_uncommon:
                tuple_action = first_action + second_action
                valid, reason = env._game_rules(action, env)
                if not valid:
                    logging.debug(f"Found action is not valid: {reason}")
                    continue
                affected_substation = affected_substations(second_action)[0]
                if first_affected_substation == affected_substation:
                    continue
                assert len(affected_substations(tuple_action)) == 2
                obs_, _, done, info = obs.simulate(tuple_action)
                if not done:
                    new_max_rho = obs_.rho.max()
                    rho_improvement: float = old_max_rho - new_max_rho
                    if rho_improvement > 0:
                        logging.debug(f"[1] Found good tuple action {action} with {rho_improvement}")
                        actions_results.append((rho_improvement, EncodedTopologyAction(tuple_action)))
                    else:
                        logging.debug(f"Tuple action has bad rho_improvement: {rho_improvement}")
        else:
            # (1) if the first action is not from substation 16 then sample other actions with sample size e.g. 1000;
            # without replacement and check that actions are legal;
            random_indices = np.random.choice(len(all_unitary_actions), sample_size, replace=False)
            logging.debug(f"randomly sample other actions for {affected_substation_id}")
            for ind in random_indices:
                second_action = all_unitary_actions[ind]
                tuple_action = first_action + second_action
                valid, reason = env._game_rules(tuple_action, env)
                if not valid:
                    logging.debug(f"Action is invalid: reason: {reason}")
                    continue
                affected_substation = affected_substations(second_action)[0]
                if first_affected_substation == affected_substation:
                    continue
                assert len(affected_substations(tuple_action)) == 2
                obs_, _, done, info = obs.simulate(tuple_action)
                if not done:
                    new_max_rho = obs_.rho.max()
                    rho_improvement: float = old_max_rho - new_max_rho
                    if rho_improvement > 0:
                        logging.debug(f"[2] Found good tuple action {action} with {rho_improvement}")
                        actions_results.append((rho_improvement, EncodedTopologyAction(tuple_action)))

    best_actions = sorted(actions_results, key=lambda tup: tup[0], reverse=True)
    return best_actions


def topology_search_triples(
        env: BaseEnv,
        best_tuple_actions: List[BaseAction],
        all_unitary_actions: List[BaseAction],
        sample_size: int = 1000,
        show_progress: bool = False,
) -> List[Tuple[float, EncodedTopologyAction]]:
    """Search for good actions that occur on three substations at the same time, by combining unitary and tuple actions.

    Args:
        env: The environment used for the search.
        best_tuple_actions: The best tuple actions, found by topology_search_tuples.
        all_unitary_actions: All unitary actions of the environment.
        sample_size: How many unitary actions should be tried to be combined with tuple actions.
        show_progress: Set to true to show a progress bar.

    Returns:
        A list of triple actions ranked by their rho_improvement.
    """
    obs: BaseObservation = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    old_max_rho = min_rho  # save old max_rho for comparison
    n_unitary_actions = len(all_unitary_actions)
    actions_results: List[Tuple[float, EncodedTopologyAction]] = []

    # Triple search
    for first_action in tqdm(best_tuple_actions, disable=not show_progress):
        first_affected_substations = affected_substations(first_action)
        assert len(first_affected_substations) == 2, "first action is tuple action"

        # Sample actions
        actions_to_sample = sample_size
        while actions_to_sample > 0:
            second_action = all_unitary_actions[randint(0, n_unitary_actions - 1)]
            if affected_substations(second_action)[0] in first_affected_substations:
                # Resample if substation already affected in first_action
                continue
            triple_action = first_action + second_action
            assert len(affected_substations(triple_action)) == 3
            obs_, _, done, info = obs.simulate(triple_action)
            if not done:
                new_max_rho = obs_.rho.max()
                rho_improvement: float = old_max_rho - new_max_rho
                if rho_improvement > 0:
                    logging.debug(f"[2] Found good triple action {triple_action} with {rho_improvement}")
                    actions_results.append((rho_improvement, EncodedTopologyAction(triple_action)))

            actions_to_sample -= 1

    best_actions = sorted(actions_results, key=lambda tup: tup[0], reverse=True)
    return best_actions


def topology_search_sequential_x_steps(
        env: BaseEnv, sub_action_set: dict, steps: int = 1, show_progress: bool = False
) -> List[Tuple[float, BaseAction]]:
    """Specific topology search that creates tuple, triple or more sequential action sets based on
    a provided set of sub-actions. This method does not require to overwrite the number of possible
    actions, because it ensures that for each step the action is executed.

    Note that this Agent takes the number of steps in the environment as provided.

    Args:
        env: Grid2Op Environment,
        sub_action_set: Dictionary containing all actions of the action set, seperated by substation,
        steps: Number of steps,
        show_progress: Set to true to show a progress bar,

    Returns:
        Action set.

    """

    obs: BaseObservation = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()

    actions_results: List[Tuple[float, EncodedTopologyAction]] = []

    full_action = env.action_space({})
    performed_actions = []
    effected_substations = []
    rho_improvement = [min_rho]
    done = False
    # a[0].as_dict()['set_bus_vect']['modif_subs_id'][0]==
    for step in range(steps):
        best_id = None
        best_action = env.action_space({})
        if done:
            break

        new_max_rho = min_rho

        # Select all actions that were not effected:
        available_actions = []
        for k, v in sub_action_set.items():
            if k not in effected_substations:
                available_actions += v

        # For each step we run through the whole set of sub_actions and only select the best action
        for idx, action in tqdm(enumerate(available_actions), disable=not show_progress):
            valid, reason = env._game_rules(action, env)  # noqa
            if not valid:
                logging.debug(f"Found action is not valid: {reason}")
                continue
            if action.as_dict()["set_bus_vect"]["modif_subs_id"][0] in effected_substations:
                logging.debug("Substation of action already in action set")
                continue

            obs_, _, ddone, info = obs.simulate(action)

            valid = is_valid(observation=obs, act=action, done_sim=ddone, info_sim=info)

            if valid and (obs_.rho.max() < new_max_rho):
                new_max_rho = obs_.rho.max()
                best_action = action
                best_id = idx

        # Save best action:
        if best_id:
            performed_actions.append(best_id)
            effected_substations.append(best_action.as_dict()["set_bus_vect"]["modif_subs_id"][0])

        full_action = full_action + best_action

        # Based on the best action, we now run one step in the environment:
        # Note that we reconnect if possible. This might dilute the overall performacne.
        best_action = find_best_line_to_reconnect(obs, best_action)
        obs, _, done, _ = env.step(best_action)
        min_rho = obs.rho.max()
        rho_improvement.append(min_rho)

    # print results:
    logging.info(
        f"The following action ids where selected {performed_actions} which affected the substations"
        f"{effected_substations} and led to a reduction of {rho_improvement}."
    )

    return full_action
