"""In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. while not game over:
    3.     if not overflow:
    4.         step a "reconnect disconnected line" or "do nothing" action
    5.     else:
    6.         search a greedy action to minimize the max rho (~60k possible actions)
    7.         save the tuple of (None, observation, action) to a csv file.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import defopt
import grid2op
import numpy as np
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import BaseEnv

from curriculumagent.teacher.submodule.common import save_sample_new
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from curriculumagent.teacher.submodule.topology_action_search import topology_search_topk


def collect_general_experience(
        save_path: Path,
        env_name_path: Union[str, Path],
        chronics_path: str,
        seed: int,
        rho_acting_threshold: float = 0.925,
        top_k: int = 125,
        chronic_limit: int = None,
):
    """Collect general teacher_experience by running a greedy search whenever the agent encounters a high power load.

    Args:
        save_path: The path to save the teacher_experience to.
        env_name_path: The name or path of the environment used.
        chronics_path: The path to the chronics of the environment.
        seed: The seed of the environment.
        rho_acting_threshold: When the environments max_rho goes above this threshold the agent begins to act.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        chronic_limit: How many chronics to run. Runs endless if set to None.

    Returns:
        None.
    """
    if isinstance(env_name_path, Path):
        env_name_path = str(env_name_path)

    # Setup environment with best possible backend
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
    except ImportError:  # noqa
        backend = PandaPowerBackend()
        logging.warning("Not using lightsim2grid! Operation will be slow!")
    if chronics_path:
        # Use env_name_path as path together with chronics_path
        env: BaseEnv = grid2op.make(dataset=env_name_path, chronics_path=chronics_path, backend=backend)
    else:
        # Use env_name_path as name
        env: BaseEnv = grid2op.make(dataset=env_name_path, backend=backend)

    all_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)
    np.random.seed(seed)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    num_chronics = len(os.listdir(chronics_path))
    if chronic_limit:
        num_chronics = min(num_chronics, chronic_limit)
    for _ in range(num_chronics):
        env.reset()  # env.reset() loads the next chronic
        dst_step = 0
        logging.info(f"Scenario to test is [{env.chronics_handler.get_name()}]，start from step-{dst_step:d}... ...")
        env.fast_forward_chronics(dst_step)
        obs, done = env.get_obs(), False
        while not done:
            if obs.rho.max() >= rho_acting_threshold:
                best_actions = topology_search_topk(env, obs, all_actions, top_k=125)
                if len(best_actions) > 0:
                    action = best_actions[0][1]
                    best_actions = [(ri, EncodedTopologyAction(act)) for ri, act in best_actions]
                    obs_, _, done, info = env.step(action)
                    save_sample_new(Path(save_path), best_actions, obs, obs_, top_k=top_k)
                    obs = obs_
                else:
                    action = env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)
                    obs, _, done, info = env.step(action)
            else:
                action = env.action_space({})
                action = find_best_line_to_reconnect(obs, action)
                obs, _, done, info = env.step(action)
            if done and obs.current_step < obs.max_step:
                logging.info(
                    f"Game over at t={obs.current_step}/{obs.max_step} at chronic: {env.chronics_handler.get_name()}"
                )


def run_general_teacher(
        save_path: Path,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        rho_acting_threshold: float = 0.925,
        *,
        n_episodes: int = 100,
        limit_chronics: int = None,
        top_k: int = 125,
        jobs: int = os.cpu_count(),
        seed: int = 42,
):
    """This teacher is trying to collect useful actions by doing topology search while running an
    environment when an overflow occurs.

    Args:
        save_path: Path to save the teacher_experience file to.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        rho_acting_threshold: When the environments max_rho goes above this threshold the agent begins to act.
        n_episodes: Number of episodes (run through all scenarios) to run.
        limit_chronics: Limit the number of chronics to run through.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        jobs: Number of jobs to use for parallel teacher_experience collection.
        seed: The random seed to use.

    Returns:
        None.
    """

    # Make sure we can initialize the environment
    # This also makes sure that the environment actually exits or gets downloaded
    env: BaseEnv = grid2op.make(env_name_path)
    chronics_path = env.chronics_handler.path
    if chronics_path is None:
        raise ValueError(f"Can't determine chronics path of given environment {env_name_path}")

    # Parallel implementation
    start = time.time()
    tasks = []
    for episode in range(n_episodes):
        tasks.append(
            (save_path, env_name_path, chronics_path, seed + episode, rho_acting_threshold, top_k, limit_chronics)
        )
    if jobs == 1:
        # This makes debugging easier since we don't fork into multiple processes
        logging.info(f"The following {len(tasks)} tasks will executed sequentially: {tasks}")
        for task in tasks:
            collect_general_experience(*task)
    else:
        logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers: {tasks}")
        with Pool(jobs) as p:
            p.starmap(collect_general_experience, tasks)
    end = time.time()
    elapsed = end - start
    logging.info(f"Parallel time: {elapsed}s")


