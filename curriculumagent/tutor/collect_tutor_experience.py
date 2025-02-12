"""This file consist the CreateJuniorData class which converts the tutor data into readable data for the
junior agent.

"""
import logging
import os
import random
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union, Tuple, Optional

import defopt
import grid2op
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv

from curriculumagent.common.obs_converter import obs_to_vect
from curriculumagent.common.utilities import split_and_execute_action, find_best_line_to_reconnect
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor


def collect_tutor_experience_one_chronic(
        action_paths: Union[Path, List[Path]],
        chronics_id: int,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        seed: Optional[int] = None,
        enable_logging: bool = True,
        subset: Optional[Union[bool, str]] = False,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {}
):
    """Collect tutor experience of one chronic.

    Args:
        action_paths: List of Paths for the tutor.
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
        env = grid2op.make(dataset=env_name_path, backend=backend,**env_kwargs)
    except ImportError:  # noqa
        env = grid2op.make(dataset=env_name_path,**env_kwargs)
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

    tutor = TutorAgent(action_space=env.action_space, action_space_file=action_paths, **tutor_kwargs)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    done, step, obs = False, 0, env.get_obs()
    if subset:
        vect_obs = obs_to_vect(obs)
    elif subset == "graph":
        vect_obs = obs_to_vect(obs, True)
    else:
        vect_obs = obs.to_vect()

    records = np.zeros((1, 1 + len(vect_obs)), dtype=np.float32)

    while not done:
        action, idx = tutor.act_with_id(obs)

        if isinstance(action,np.ndarray):
            action:grid2op.Action.BaseAction = tutor.action_space.from_vect(action)

        if action.as_dict()!={} and (idx != -1):

            # Note that we exclude the TOPO Actions of the Tutor!!!
            if subset:
                vect_obs = obs_to_vect(obs)
            elif subset == "graph":
                vect_obs = obs_to_vect(obs, True)
            else:
                vect_obs = obs.to_vect()

            records = np.concatenate(
                (records, np.hstack([idx, vect_obs]).astype(np.float32).reshape(1, -1)), axis=0
            )

            # Execute Action:
            # This method does up to three steps and returns the output
            obs, _, done, _ = split_and_execute_action(env=env, action_vect=action.to_vect())



        else:
            # Use Do-Nothing Action
            act_with_line = find_best_line_to_reconnect(obs, env.action_space({}))
            obs, _, done, _ = env.step(act_with_line)

        step = env.nb_time_step

    logging.info(f"game over at step-{step}")

    return records


def generate_tutor_experience(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        subset: Optional[Union[bool, str]] = False,
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
        tasks.append((action_paths, chronic_id, env_name_path, seed, True, subset, TutorAgent, tutor_kwargs,env_kwargs))
    if jobs == 1:
        # This makes debugging easier since we don't fork into multiple processes
        logging.info(f"The following {len(tasks)} tasks will executed sequentially: {tasks}")
        out_result = []
        for task in tasks:
            out_result.append(collect_tutor_experience_one_chronic(*task))
    else:
        logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers:")
        start = time.time()
        with Pool(jobs) as p:
            out_result = p.starmap(collect_tutor_experience_one_chronic, tasks)
        end = time.time()
        elapsed = end - start
        logging.info(f"Time: {elapsed}s")

    # Now concatenate the result:
    all_experience = np.concatenate(out_result, axis=0)
    if save_path.is_dir():
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        save_path = save_path / f"tutor_experience_{now}.npy"

    np.save(save_path, all_experience)
    logging.info(f"Tutor experience has been saved to {save_path}")


def prepare_dataset(
        traindata_path: Path,
        target_path: Path,
        dataset_name: str,
        extend: bool = False,
        seed: Optional[int] = 42,
):
    """Prepare/process the training data given by the tutor. The data is seperated into
    training, validation and test dataset, saved at the target_path.

    Args:
        traindata_path: The path to the generated records_* files by the tutor to use.
        target_path: The path to save the dataset files to.
        dataset_name: Name of the dataset which gets used to create the files.
        extend: Whether to extend the existing training data instead of creating a new dataset.
        seed: The random seed for shuffling the samples.

    Returns:
        None.

    """
    np.random.seed(seed)

    assert traindata_path.exists() and traindata_path.is_dir(), f"{traindata_path} is not a valid directory"

    # Change this more globaly
    npy_files = list(traindata_path.glob("*.npy"))
    assert len(npy_files), f"There are no files at {traindata_path}"

    if not extend:
        logging.info(f"Creating final dataset at directory {target_path}")
        make_dataset(target_path=target_path, dataset_name=dataset_name, npy_files=npy_files)
        logging.info("Done")
    else:
        train_file_path = target_path / f"{dataset_name}_train.npz"
        extend_dataset(npy_files=npy_files, train_file_path=train_file_path)


def merge_dataset(npy_file_paths: List[Path]) -> np.ndarray:
    """Merge the given numpy files at npy_file_paths and return merged dataset.

    Note:
         If the input consists of tuple action IDs, then the IDs are merged to one id!

    Args:
        npy_file_paths:  Input numpy files, created by the Tutor.

    Returns:
        The unique actions in a merged array.

    """
    assert len(npy_file_paths) > 0, "No files given for merging"

    npy_file_paths = sorted(npy_file_paths)  # This will hopefully make this reproducible

    loaded_actions = [np.load(str(f)) for f in npy_file_paths]
    concated_actions = np.concatenate(loaded_actions, axis=0)
    unique_actions = np.unique(concated_actions, axis=0)

    logging.info(f"All Experience: {concated_actions.shape}")
    logging.info(f"Unique Experience: {unique_actions.shape}")
    logging.info(f"Unique Actions: {len(np.unique(concated_actions[:, 0], axis=0))}")
    return unique_actions


def create_dataset(
        target_path: Union[Path, str],
        data: np.ndarray,
        dataset_name: Optional[str] = "Junior",
        chosen: Optional[Union[bool, list]] = False,
):
    """Given all samples data, select the right features and labels and split the dataset up.
    Save the dataset in three different files at target_path.

    Args:
        target_path: Path, where to save the training, validation and testing data for the junior.
        data: Dataset for the junior model.
        dataset_name: Optional name for the data set.
        chosen: Boolean or List for filtering the observations of the original observations.

    Returns:
        None. Saves  '{dataset_name}_train.npz','{dataset_name}_val.npz','{dataset_name}_test.npz' in the
    target path.

    """
    # Shuffle actions
    np.random.shuffle(data)

    assert target_path.exists() and target_path.is_dir(), "Given target_path is not a directory that exists"

    # Subset
    if chosen:
        data = data[:, chosen]

    # Dataset partition
    num_sampling = data.shape[0]
    train_size = num_sampling * 8 // 10
    # validate_size = num_sampling // 10
    test_size = num_sampling // 10
    # s for state, feature; a for action, label.

    # Check whether the input is a Tuple ID or not. If yes
    s_train, a_train = data[:train_size, 1:], data[:train_size, :1].astype(int)
    s_validate, a_validate = data[train_size:-test_size, 1:], data[train_size:-test_size, :1].astype(int)
    s_test, a_test = data[-test_size:, 1:], data[-test_size:, :1].astype(int)

    logging.info(f"TrainSet Size: {len(s_train)}")
    logging.info(f"ValidationSet Size: {len(s_validate)}")
    logging.info(f"TestSet Size: {len(s_test)}")

    train_path = target_path / f"{dataset_name}_train.npz"
    np.savez_compressed(train_path, s_train=s_train, a_train=a_train)

    validation_path = target_path / f"{dataset_name}_val.npz"
    np.savez_compressed(validation_path, s_validate=s_validate, a_validate=a_validate)

    test_path = target_path / f"{dataset_name}_test.npz"
    np.savez_compressed(test_path, s_test=s_test, a_test=a_test)


def load_dataset(
        dataset_path: Union[str, Path], dataset_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset for the training of the Junior model.

    Args:
        dataset_path: Path, where to find the dataset.
        dataset_name: Name of the dataset. There should be '{dataset_name}_train.npz',
        '{dataset_name}_val.npz' and a '{dataset_name}_test.npz' in the target path.


    Returns:
        Returns x,y values for train, val and test data.

    """
    train_path = dataset_path / f"{dataset_name}_train.npz"
    train_data = np.load(train_path)

    validation_path = dataset_path / f"{dataset_name}_val.npz"
    validation_data = np.load(validation_path)

    test_path = dataset_path / f"{dataset_name}_test.npz"
    test_data = np.load(test_path)

    return (
        train_data["s_train"],
        train_data["a_train"],
        validation_data["s_validate"],
        validation_data["a_validate"],
        test_data["s_test"],
        test_data["a_test"],
    )


def make_dataset(
        target_path: Path, dataset_name: str, npy_files: List[Path]):
    """Join all experience into one file and then execute the create_dataset.

    Args:
        target_path: Path, where to write the new experience.
        dataset_name: Name of the files.
        npy_files: list of np.ndarrays.

    Returns:
        None.

    """
    logging.info("Merging data...")
    experience = merge_dataset(npy_files)

    logging.info("Splitting data...")
    create_dataset(target_path=target_path, dataset_name=dataset_name, data=experience)


def extend_dataset(npy_files: List[Path], train_file_path: Path) -> object:
    """Extend existing Training Dataset with new files.

    Args:
        npy_files: List of np.ndarrays
        train_file_path: Path of the training data.

    Returns:
        An extended dataset.

    """
    logging.info(f"Trying to add data to existing dataset {train_file_path}")
    # 2. Add new data to train set

    cache_file_path = Path("merged_new_data.npz")
    if cache_file_path.exists():
        new_experience = np.load(str(cache_file_path))["experience"]
    else:
        new_experience = merge_dataset(npy_files)

    final_dataset = np.load(str(train_file_path))
    s_train, a_train = final_dataset["s_train"], final_dataset["a_train"]

    np.random.shuffle(new_experience)

    s_train = np.concatenate([s_train, new_experience[:, 1:]])
    a_train = np.concatenate([a_train, new_experience[:, :1].astype(int)])

    new_filename = train_file_path.name.split(".")[0] + "_v2.npz"
    logging.info(f"Saving new dataset to {new_filename}")
    np.savez_compressed(new_filename, s_train=s_train, a_train=a_train)

