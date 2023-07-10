"""This file consist of the N-1 Tutor, that executes the actions similar to the N-1 Teacher.

"""
import logging
import os
import random
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union, Tuple, List

import grid2op
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Exceptions import DivergingPowerFlow
from grid2op.dtypes import dt_int

from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor


def collect_divergent_powerflow_experience(
        action_paths: Union[Path, List[Path]],
        chronics_id: int,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        seed: Optional[int] = None,
        enable_logging: bool = True,
        runs_per_chronic: int = 100,
        TutorAgent: BaseAgent = GeneralTutor,
) -> List[np.array]:
    """Collect teacher_experience of the tutor right before the DivergingPowerFlow error. For that we run
    through multiple iterations of the environment, in order to collect as much information
    within previously observation to detect the DivergingPowerFlow in advance.

    In this run, we save both the T-1 and T-3 observation, with T being the moment when the agent
    hits a diverging powerflow.

    Args:
        action_paths: List of Paths for the tutor.
        chronics_id: Number of chronic to run.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        seed: Whether to init the numpy seed which is used for the Grid2Op seeds.
        runs_per_chronic: How many times the chronic should be iterated with different seeds.
        enable_logging: Whether to log the Tutor teacher_experience search.
        TutorAgent: Tutor Agent which should be used for the search.

    Returns:
        Returns two array containing the T-1 and T-3 observations of all cases where a Diverging Powerflow
        occurred.
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        env = grid2op.make(dataset=env_name_path, backend=backend)
    except ImportError:  # noqa
        env = grid2op.make(dataset=env_name_path)
        logging.warning("Not using lightsim2grid! Operation will be slow!")

    if seed:
        np.random.seed(seed)

    # After initializing the environment, let's init the tutor
    tutor = TutorAgent(action_space=env.action_space, action_space_file=action_paths)

    max_int = np.iinfo(dt_int).max
    env_seeds = list(np.random.randint(max_int, size=runs_per_chronic))
    print(env_seeds)

    logging.info(f"Run current chronic:{env.chronics_handler.get_name()} " f"with a total of {runs_per_chronic} seeds.")

    # We run through multiple itartions of the chronic:
    records_t_minus_1 = []
    records_t_minus_3 = []
    for env_seed in env_seeds:
        env.set_id(chronics_id)
        env.seed(env_seed)
        env.reset()

        done, obs, info = False, env.get_obs(), []
        obs_lists = []
        act_list = []
        while not done:
            action, idx = tutor.act_with_id(obs)

            act = find_best_line_to_reconnect(obs=obs, original_action=env.action_space.from_vect(action))
            obs, _, done, info = env.step(act)

            if not done:
                # Save the last three steps:
                obs_lists.append(obs.copy())
                del obs_lists[:-3]
                act_list.append(idx)
                del act_list[:-3]

        if isinstance(info["exception"], list) and len(info["exception"]) > 0:
            if isinstance(info["exception"][0], DivergingPowerFlow):
                logging.info(f"Divergin Powerflow detected at step {obs.current_step}")
                records_t_minus_1.append(
                    np.hstack([act_list[-1], obs_lists[-1].to_vect()]).astype(np.float32).reshape(1, -1)
                )
                records_t_minus_3.append(
                    np.hstack([act_list[-3], obs_lists[-3].to_vect()]).astype(np.float32).reshape(1, -1)
                )

    return [np.array(records_t_minus_1), np.array(records_t_minus_1)]


def generate_divergent_exp(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        seed: Optional[int] = None,
        TutorAgent: BaseAgent = GeneralTutor,
):
    """Method to run the Divergent Powerflow Search in parallel. Quite similar to the tutor.

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the teacher_experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. If num_sample is smaller than num chronics,
        a subset is taken. If it is larger, the chronics are sampled with replacement.
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments.
        TutorAgent: Tutor Agent which should be used for the search, default is the GeneralTutor.

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
        tasks.append((action_paths, chronic_id, env_name_path, seed, TutorAgent))
    if jobs == 1:
        # This makes debugging easier since we don't fork into multiple processes
        logging.info(f"The following {len(tasks)} tasks will executed sequentially: {tasks}")
        out_result = []
        for task in tasks:
            out_result.append(collect_divergent_powerflow_experience(*task))
    else:
        logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers:")
        start = time.time()
        with Pool(jobs) as p:
            out_result = p.starmap(collect_divergent_powerflow_experience, tasks)
        end = time.time()
        elapsed = end - start
        logging.info(f"Time: {elapsed}s")

    # Now concatenate the result:
    all_experience = np.concatenate(out_result, axis=0)
    if save_path.is_dir():
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        save_path = save_path / f"divergent_powerflow_{now}.npy"

    np.save(save_path, all_experience)
    logging.info(f"Divergent PF teacher_experience has been saved to {save_path}")
