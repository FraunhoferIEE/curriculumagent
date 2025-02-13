import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from random import random
from typing import List, Union, Optional, Tuple

import grid2op
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from matplotlib import pyplot as plt

from curriculumagent.common.utilities import split_and_execute_action, find_best_line_to_reconnect
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor


def collect_topology_tutor_one_chronic(
        action_paths: Union[Path, List[Path]],
        chronics_id: int,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        seed: Optional[int] = None,
        enable_logging: bool = True,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {}
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect tutor experience of one chronic.

    Args:
        action_paths: List of Paths for the tutor. This can be set to None if you want to use the
        DoNothingAgent for testing.
        chronics_id: Number of chronic to run.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        seed: Whether to init the Grid2Op env with a seed
        enable_logging: Whether to log the Tutor experience search.
        subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected. Further note, that it is possible to say "graph" in order to get the connectivity_matrix as well.
        TutorAgent: Tutor Agent which should be used for the search.
        tutor_kwargs: Additional arguments for the tutor, e.g. the max rho or the topology argument.
        env_kwargs: Optional arguments that should be used when initializing the environment.

    Returns:
        None.
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        env = grid2op.make(dataset=env_name_path, backend=backend, **env_kwargs)
    except ImportError:  # noqa
        env = grid2op.make(dataset=env_name_path, **env_kwargs)
        logging.warning("Not using lightsim2grid! Operation will be slow!")

    if seed:
        env.seed(seed)

    env.set_id(chronics_id)
    env.reset()
    logging.info(f"current chronic:{env.chronics_handler.get_name()}")

    # After initializing the environment, let's init the tutor
    if tutor_kwargs is None:
        tutor_kwargs = {}
    else:
        logging.info(f"Run Tutor with these additional kwargs {tutor_kwargs}")

    if action_paths is not None:
        tutor_kwargs["action_space_file"] = action_paths

    tutor = TutorAgent(action_space=env.action_space, **tutor_kwargs)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    done, step, obs = False, 0, env.get_obs()

    vect_obs = obs.to_vect()
    # action ID + vector of observation
    obs_records = np.empty((0, 1 + len(vect_obs) + len(env.action_space().to_vect())), dtype=np.float32)

    # Count, topo_vect
    # Start with the default topology
    topo_records = np.hstack((0, obs.topo_vect)).reshape(1,-1)

    vect_obs = None
    a_comb = env.action_space({})
    while not done:
        # Get Tutor action
        action = tutor.act(observation=obs, reward=0, done=done)

        if isinstance(action, np.ndarray):
            action: grid2op.Action.BaseAction = tutor.action_space.from_vect(action)

        # Option1: Tutor is active, let's save the observation and execute the action
        if action.as_dict() != {}:

            # 3. Save vect_obs and add actions. Save only if first action of tutor in this run.
            if a_comb.as_dict() == {}:
                vect_obs = obs.to_vect().copy()
            a_comb += action

            # Execute Action:
            # This method does up to three steps and returns the output
            obs, _, done, _ = split_and_execute_action(env=env, action_vect=action.to_vect())

        # Option2: The Tutor produces a do nothing action. This happends, because the grid is stable
        # or it doesn't find an adequate method. One way or the other, we save the topology vector and
        # the observation!
        else:
            # First, we have to save the topo records and count the number of "do-nothing" actions of the tutor:
            topo_id = np.where((obs.topo_vect == topo_records[:, 1:]).all(axis=1))[0]

            if len(topo_id) > 0:
                topo_records[topo_id, 0] = topo_records[topo_id, 0] + 1
            else:
                topo_records = np.concatenate(
                    (topo_records, np.hstack([1, obs.topo_vect]).reshape(1, -1)), axis=0)
                # Set Topo_id:
                topo_id = len(topo_records)

            # Second, if this do nothing happened due to a tutor action, we expect it to save the
            # topology action of the topo_records with the observation and action !
            if a_comb.as_dict() != {}:
                # Save obs_record & set a_comb back to 0
                obs_records = np.concatenate(
                    (obs_records, np.hstack([topo_id, vect_obs, a_comb.to_vect()]).astype(np.float32).reshape(1, -1)),
                    axis=0)

                a_comb = env.action_space({})

            # Use Do-Nothing Action
            act_with_line = find_best_line_to_reconnect(obs, env.action_space({}))
            obs, _, done, _ = env.step(act_with_line)

        step = env.nb_time_step

    logging.info(f"game over at step-{step} with a total of {len(topo_records)} different topologies.")

    return topo_records, obs_records


def generate_topo_tutor_experience(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        seed: Optional[int] = None,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {},
):
    """Method to run the Tutor in parallel.

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. If num_sample is smaller than num chronics,
            a subset is taken. If it is larger, the chronics are sampled with replacement.
        subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected. Further, note that it is possible to say graph in order to get the connectivity_matrix as well.
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments
        TutorAgent: Tutor Agent which should be used for the search, default is the GeneralTutor.
        tutor_kwargs: Optional arguments that should be passed to the tutor agent.
        env_kwargs: Optional arguments that should be used when initializing the environment.

    Returns:
        None, saves results as numpy file.

    """
    log_format = "(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]"
    logging.basicConfig(level=logging.INFO, format=log_format)

    if jobs == -1:
        jobs = os.cpu_count()

    tasks = []

    # Make sure we can initialize the environment
    # This also makes sure that the environment actually exits or gets downloaded
    env: BaseEnv = grid2op.make(env_name_path)
    chronics_path = env.chronics_handler.path
    if chronics_path is None:
        raise ValueError(f"Can't determine chronics path of given environment {env_name_path}")

    if num_chronics is None:
        num_chronics = len(os.listdir(chronics_path))

    if num_sample:
        if num_sample <= num_chronics:
            sampled_chronics = random.sample(range(num_chronics), num_sample)
        else:
            sampled_chronics = random.choices(np.arange(num_chronics), k=num_sample)
    else:
        sampled_chronics = np.arange(num_chronics)

    for chronic_id in sampled_chronics:
        tasks.append((action_paths, chronic_id, env_name_path, seed, True, TutorAgent, tutor_kwargs,env_kwargs))
    if jobs == 1:
        # This makes debugging easier since we don't fork into multiple processes
        logging.info(f"The following {len(tasks)} tasks will executed sequentially: {tasks}")
        out_result = []
        for task in tasks:
            out_result.append(collect_topology_tutor_one_chronic(*task))
    else:
        logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers:")
        start = time.time()
        with Pool(jobs) as p:
            # This now countains touples:
            out_result = p.starmap(collect_topology_tutor_one_chronic, tasks)
        end = time.time()
        elapsed = end - start
        logging.info(f"Time: {elapsed}s")

    ############# Combine the results BUT also adjust the respective Ids!
    # Now concatenate the result:

    logging.info("Running through the individual runs per scenario and combining them to one"
                 "file\n")
    # Save beginning files
    global_topo_id, global_obs_records = combine_topo_datasets(out_result)

    logging.info(f"Lenght of topologies is {len(global_topo_id)}")
    logging.info(f"Lenght of obsact is {len(global_obs_records)}")

    if save_path.is_dir():
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        save_path = save_path / f"tutor_experience_{now}.npz"

    np.savez_compressed(save_path, topo_id=global_topo_id, obs_records=global_obs_records)
    logging.info(f"Tutor experience has been saved to {save_path}")


def combine_topo_datasets(comb_res:list)->Tuple[np.ndarray,np.ndarray]:
    """ Runs through all data of the provided list and then combines both the topology action as well
    as the global_obs_records

    Args:
        comb_res: Output of previously runs in a list. Within the list, each run should hav the order (topo_id,
        obs_records).

    Returns:

    """
    global_topo_id = comb_res[0][0].copy()
    global_obs_records = comb_res[0][1].copy()

    for dat in comb_res[1:]:
        topo_records, obs_records = dat

        # First check, whether the topologies are already in the global_topo_id
        key_tuples = []
        for i, topo in enumerate(topo_records):
            topo_id = np.where((topo[1:] == global_topo_id[:, 1:]).all(axis=1))[0]
            # The topology already exists:
            if len(topo_id) > 0:
                key_tuples.append((int(i), int(topo_id)))
                global_topo_id[topo_id, 0] += topo[0]
            else:
                global_topo_id = np.vstack((global_topo_id, topo))

        # Now we overwrite the obsact record an save it:
        for i, j in key_tuples:
            obs_records[obs_records[:, 0] == i, 0] = j

        global_obs_records = np.vstack((global_obs_records, obs_records))
    return global_topo_id,global_obs_records

def prepare_topo_dataset(
        traindata_path: Path,
        target_path: Path,
        dataset_name: str,
        seed: Optional[int] = 42,
        top_k_topologies:int = None,
        plot_choice:bool=False,
        env:grid2op.Environment =None
):
    """ A specific dataset to load all required data / action types and save them again into training/val/testing

    Args:
        traindata_path: The path to the generated records_* files by the tutor to use.
        target_path: The path to save the dataset files to.
        dataset_name: Name of the dataset which gets used to create the files.
        seed: The random seed for shuffling the samples.
        top_k_topologies: whether to only return a given set of topologies,i.e., only look at the top k of topologies.
        If set to none, we look at all topologies.
        plot_choice: Whether to plot the number of topolgies
        env: provide a grid2op Env to only extract the observation (and not the action as well)

    Returns:
        None.

    """
    np.random.seed(seed)

    assert traindata_path.exists() and traindata_path.is_dir(), f"{traindata_path} is not a valid directory"

    # Change this more globaly
    npy_files = list(traindata_path.glob("*.npz"))
    assert len(npy_files), f"There are no files at {traindata_path}"

    logging.info(f"Creating final dataset at directory {target_path}")

    npy_files = sorted(npy_files)  # This will hopefully make this reproducible

    loaded_actions = [np.load(str(f)) for f in npy_files]
    extracted_dat = [[f["topo_id"],f["obs_records"]] for f in loaded_actions]
    global_topo_id, global_obs_records = combine_topo_datasets(extracted_dat)

    logging.info(f"Lenght of topologies is {len(global_topo_id)}")
    logging.info(f"Lenght of obsact is {len(global_obs_records)}")

    # Topologies:
    top_freq = global_topo_id[global_topo_id[:, 0].argsort()[::-1],0]

    if plot_choice:
        plt.plot(top_freq[:1000])
        if top_k_topologies:
            plt.vlines(top_k_topologies, ymin=0, ymax=top_freq.max(), color="red")
        plt.title("Sorted topology based on of the top 1000 topologies")
        plt.legend(["Sorted frequency of topology", "best_n"])
        plt.show()

    if top_k_topologies:
        # Now extract the subset:
        logging.info(f"Only considering the top k topologies with {top_k_topologies}")
        top_k = global_topo_id[:, 0].argsort()[-top_k_topologies:]
        mask = [i in top_k for i in global_obs_records[:, 0]]
        sub_topo_ids = global_topo_id[top_k[::-1]]
        sub_obs_records = global_obs_records[mask]
        for i, j in enumerate(top_k[::-1]):
            mask = sub_obs_records[:, 0] == j
            sub_obs_records[mask, 0] = i
        unique_obs = np.unique(sub_obs_records,axis=0)

        logging.info(f"We have a total of {sub_obs_records.shape[0]} observation, where {unique_obs.shape[0]} "
                     f"are unique. This is a total of {unique_obs.shape[0]/sub_obs_records.shape[0]}%")

    else:
        unique_obs = np.unique(global_obs_records, axis=0)

        logging.info(f"We have a total of {global_obs_records.shape[0]} observation, where {unique_obs.shape[0]} "
                     f"are unique. This is a total of {unique_obs.shape[0] / global_obs_records.shape[0]}%")
        sub_topo_ids = global_topo_id


    # Now let's save the observation and action sets:
    np.random.shuffle(unique_obs)

    assert target_path.exists() and target_path.is_dir(), "Given target_path is not a directory that exists"

    # Dataset partition
    num_sampling = unique_obs.shape[0]
    train_size = num_sampling * 8 // 10
    # validate_size = num_sampling // 10
    test_size = num_sampling // 10
    # s for state, feature; a for action, label.

    if env:
        max_steps = env.get_obs().to_vect().shape[0] +1
    else:
        max_steps = None

    # Check whether the input is a Tuple ID or not. If yes
    s_train, a_train = unique_obs[:train_size, 1:max_steps], unique_obs[:train_size, :1].astype(int)
    s_validate, a_validate = unique_obs[train_size:-test_size, 1:max_steps], unique_obs[train_size:-test_size, :1].astype(int)
    s_test, a_test = unique_obs[-test_size:, 1:max_steps], unique_obs[-test_size:, :1].astype(int)

    logging.info(f"TrainSet Size: {len(s_train)}")
    logging.info(f"ValidationSet Size: {len(s_validate)}")
    logging.info(f"TestSet Size: {len(s_test)}")

    train_path = target_path / f"{dataset_name}_train.npz"
    np.savez_compressed(train_path, s_train=s_train, a_train=a_train)

    validation_path = target_path / f"{dataset_name}_val.npz"
    np.savez_compressed(validation_path, s_validate=s_validate, a_validate=a_validate)

    test_path = target_path / f"{dataset_name}_test.npz"
    np.savez_compressed(test_path, s_test=s_test, a_test=a_test)

    np.save(target_path / "topologies.npy",sub_topo_ids)