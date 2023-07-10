"""
This file runs the tuple and triple search defined in :mod:`topology_action_search` and saves the collected
teacher_experience in a csv file for further processing using :mod:`generate_action_space`.
"""
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import defopt
import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from tqdm import tqdm

from curriculumagent.teacher.submodule.common import save_sample_new
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from curriculumagent.teacher.submodule.topology_action_search import topology_search_sequential_x_steps


def collect_sequential_experience(
        save_path: Path,
        env_name_path: Union[str, Path],
        chronics_path: Optional[str],
        steps: int,
        seed: int,
        rho_acting_threshold: float = 0.9,
        actions_dict: Optional[dict] = None,
        chronic_limit: Optional[int] = None,
) -> None:
    """Teacher class which collects the sequential teacher_experience in order to have the best values.

    Args:
        save_path: The path to save the teacher_experience to.
        env_name_path: The name or path of the environment used.
        chronics_path: The path to the chronics of the environment.
        steps: Length of combined actions, e.g. if steps is 3, then sequentially 3 substations are effected.
        seed: The seed of the environment.
        actions_dict: Dictionary containing all available actions with the substations as keys.
        rho_acting_threshold: When the environments max_rho goes above this threshold the agent begins to act.
        chronic_limit: How many chronics to run. Runs endless if set to None.

    Returns:
        None.
    """
    if isinstance(env_name_path, Path):
        env_name_path = str(env_name_path)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        if chronics_path:
            # Use env_name_path as path together with chronics_path
            env: BaseEnv = grid2op.make(dataset=env_name_path, chronics_path=chronics_path, backend=backend)
        else:
            # Use env_name_path as name
            env: BaseEnv = grid2op.make(dataset=env_name_path, backend=backend)
    except Exception as e:  # noqa
        logging.warning("Not using lightsim2grid! Operation will be slow!")
        raise e

    # Loading the different values
    logging.info("Separate all available action on the respective substations for faster execution.")
    all_available_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)

    if not isinstance(actions_dict, dict):
        actions_dict = {}
        for act in tqdm(all_available_actions):
            sub_effected = act.as_dict()["set_bus_vect"]["modif_subs_id"][0]

            if sub_effected in actions_dict:
                actions_dict[sub_effected].append(act)
            else:
                actions_dict[sub_effected] = [act]
        logging.info(
            f"Done with separating the actions. There is a total of {len(actions_dict.keys())} substations with "
            f"overall {len(all_available_actions)} number of actions."
        )

    # Setup chronics with deterministic shuffling
    np.random.seed(seed)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    num_chronics = len(os.listdir(chronics_path))
    if chronic_limit:
        num_chronics = min(num_chronics, chronic_limit)

    # Create jobs for multiprocessing:
    for _ in range(num_chronics):
        env.reset()  # env.reset() loads the next chronic
        dst_step = 0
        logging.info(f"Scenario to test is [{env.chronics_handler.get_name()}], start from step-{dst_step:d}... ...")
        env.fast_forward_chronics(dst_step)
        obs, done = env.get_obs(), False
        while not done:
            if obs.rho.max() >= rho_acting_threshold:

                # Now run through the sequential topological search. Note that this will continue the
                # environment by x-steps.
                # Note that we do not reconnect the lines to not divert the
                rho_old = obs.rho.max()
                best_actions = topology_search_sequential_x_steps(
                    env=env, sub_action_set=actions_dict, steps=steps, show_progress=False
                )

                done = env.done

                if not done or (done and obs.current_step == obs.max_step):
                    rho_new = env.get_obs().rho.max()
                    rho_improvement: float = rho_old - rho_new
                    if "set_bus_vect" in best_actions.as_dict().keys():
                        save_sample_new(
                            Path(save_path),
                            [(rho_improvement, EncodedTopologyAction(best_actions))],
                            obs,
                            env.get_obs(),
                            top_k=1,
                        )

            else:
                action = env.action_space({})
                action = find_best_line_to_reconnect(obs, action)
                obs, _, done, _ = env.step(action)
            dst_step += 1

        if done and obs.current_step < obs.max_step:
            logging.info(
                f"Game over at t={obs.current_step}/{obs.max_step} at chronic: {env.chronics_handler.get_name()}"
            )


def run_sequential_teacher(
        save_path: Path,
        env_name_path: str = "l2rpn_neurips_2020_track1_small",
        steps: int = 3,
        rho_acting_threshold: float = 0.925,
        *,
        n_episodes: int = 100,
        limit_chronics: int = None,
        jobs: int = os.cpu_count(),
        seed: int = 42,
) -> Path:
    """This teacher is trying to collect useful actions by doing topology search while running an
    environment when an overflow occurs. It also tries to combine actions to form tuple or triple actions.

    Args:
        save_path: Path to save the teacher_experience file to.
        steps: Number of combined sequential steps for the teacher to search for. Default is three sequential steps.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        rho_acting_threshold: When the environments max_rho gues above this threshold the agent begins to act.
        n_episodes: Number of episodes(run through all scenarios) to run.
        limit_chronics: Limit the number of chronics to run through.
        jobs: Number of jobs to use for parallel teacher_experience collection.
        seed: The random seed to use.

    Returns: Saving path.
    """

    env_path = Path(env_name_path)
    chronics_path = None
    if env_path.is_dir():
        env_name_path = env_path
        chronics_path = str(env_name_path / "chronics")

    start = time.time()
    tasks = []
    for episode in range(n_episodes):
        tasks.append(
            (save_path, env_name_path, chronics_path, steps, seed + episode, rho_acting_threshold, None, limit_chronics)
        )

    logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers: {tasks}")
    # Sequential implementation, useful for debugging more easily
    if jobs == 1:
        for task in tasks:
            collect_sequential_experience(*task)
    else:
        # Parallel implementation
        with Pool(jobs, maxtasksperchild=1) as p:
            p.starmap(collect_sequential_experience, tasks)
    end = time.time()
    elapsed = end - start
    logging.info(f"Needed time: {elapsed}s")

    return save_path

