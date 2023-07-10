"""In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. sample a time-step every 6 hours
    3. disconnect a line which is under possible attack
    4. search a greedy action to minimize the max rho (~60k possible actions)
    5. save the tuple of (attacked line, observation, action) to a csv file.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
import logging
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional

import defopt
import grid2op
import numpy as np
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import BaseEnv

from curriculumagent.teacher.submodule.common import save_sample_new
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from curriculumagent.teacher.submodule.topology_action_search import topology_search_topk


def collect_attacker_experience(
        save_path: Path,
        env_name_path: Union[str, Path],
        chronics_path: str,
        line_to_disconnect: int,
        n_episodes: int,
        top_k: int,
        seed: Optional[int] = None,
):
    """Collect useful actions by attacking/disconnecting specific lines at random timesteps
    and trying to find the best action to resolve that problem.

    Args:
        save_path: Path to save the teacher_experience file to.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        line_to_disconnect: The line to disconnect in this trial/scenario.
        chronics_path: The path to the chronics of the environment.
        n_episodes: Number of episodes(run through all scenarios) to run.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        seed: Optional seed for Grid2Op Env and sampling.

    Returns:
        None.
    """
    # Setup environment with the best possible backend
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

    if seed:
        np.random.seed(seed)
        env.seed(seed)

    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    # traverse all scenarios
    all_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)
    for _ in range(len(os.listdir(chronics_path))):
        env.reset()  # env.reset() loads the next chronic
        timestep = n_episodes * 72 + random.randint(0, 72)  # a random sampling every 6 hours
        logging.info(
            f"Scenario[{env.chronics_handler.get_name()}]: "
            f"at step[{timestep:d}], disconnect line-{line_to_disconnect:d}"
            f"(from bus-{env.line_or_to_subid[line_to_disconnect]:d} "
            f"to bus-{env.line_ex_to_subid[line_to_disconnect]:d}]"
        )
        # to the destination time-step
        env.fast_forward_chronics(timestep - 1)
        obs, _, done, _ = env.step(env.action_space({}))
        if done:
            break
        # disconnect the targeted line
        new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
        new_line_status_array[line_to_disconnect] = -1
        action = env.action_space({"set_line_status": new_line_status_array})
        obs, _, done, _ = env.step(action)
        if obs.rho.max() < 1:
            # not necessary to do a dispatch
            continue

        # search a greedy action
        best_actions = topology_search_topk(env, obs, all_actions, top_k=125)
        action = best_actions[0][1]
        best_actions = [(ri, EncodedTopologyAction(act)) for ri, act in best_actions]
        obs_, _, done, _ = env.step(action)
        save_sample_new(Path(save_path), best_actions, obs, obs_, top_k=top_k)


def run_attacking_teacher(
        save_path: Path,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        *,
        n_episodes: int = 100,
        top_k: int = 125,
        jobs: int = os.cpu_count(),
        seed: Optional[int] = None,
):
    """This teacher is trying to collect useful actions by attacking/disconnecting specific lines at random timesteps
    and trying to find the best action to resolve that problem.

    Args:
        save_path: Path to save the teacher_experience file to.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        n_episodes: Number of episodes(run through all scenarios) to run.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        jobs: Number of jobs to use for parallel teacher_experience collection.
        seed: Optional seed for the Grid2Op Environment and random sampling.

    Returns:
        None.
    """
    # hyper-parameters
    env_lines2attack = {
        "l2rpn_case14_sandbox": [1, 10, 15],
        "l2rpn_neurips_2020_track1_small": [45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
    }

    env_name = env_name_path
    # Try to resolve env_name using path, e.g. ~/data_grid2op/env_name -> env_name
    for key in env_lines2attack:
        if key in str(env_name_path):
            env_name = key

    try:
        lines2attack = env_lines2attack[env_name]
    except KeyError as error:
        raise ValueError(f"No lines to attack available for {env_name}") from error

    env_name_path = Path(env_name_path).expanduser()
    chronics_path = None
    if env_name_path.is_dir():
        chronics_path = str(env_name_path / "chronics")
    env_name_path = str(env_name_path)

    start = time.time()
    # Parallel implementation
    tasks = []
    for episode in range(n_episodes):
        for line_to_disconnect in lines2attack:
            tasks.append((save_path, env_name_path, chronics_path, line_to_disconnect, episode, top_k, seed))
    logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers:")
    logging.info(tasks)
    with Pool(jobs) as p:
        p.starmap(collect_attacker_experience, tasks)

    end = time.time()
    elapsed = end - start
    logging.info(f"Time: {elapsed}s")

