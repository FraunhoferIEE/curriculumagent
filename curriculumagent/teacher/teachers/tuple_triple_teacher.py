"""This file runs the tuple and triple search defined in :mod:`topology_action_search` and saves the collected
teacher_experience in a csv file for further processing using :mod:`generate_action_space`.

"""
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, List, Tuple, Union

import defopt
import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Parameters import Parameters

from curriculumagent.teacher.submodule.common import save_sample_new
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from curriculumagent.teacher.submodule.topology_action_search import topology_search_tuples, topology_search_triples


def collect_multi_experience(
        save_path: Path,
        env_name_path: Union[str, Path],
        chronics_path: Optional[str],
        use_triple_search: bool,
        seed: int,
        rho_acting_threshold: float = 0.925,
        top_k: int = 125,
        chronic_limit: int = None,
):
    """Collect general teacher_experience by running a greedy search whenever the agent encounters a high power load.
    It also tries to combine actions to tuple actions.

    Args:
        save_path: The path to save the teacher_experience to.
        env_name_path: The name or path of the environment used.
        chronics_path: The path to the chronics of the environment.
        use_triple_search: Whether to use triple search instead of tuple search.
        seed: The seed of the environment.
        rho_acting_threshold: When the environments max_rho gues above this threshold the agent begins to act.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        chronic_limit: How many chronics to run. Runs forever if set to None.

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

    action_space = env.action_space

    params: Parameters = env.parameters
    params.MAX_SUB_CHANGED = 99
    params.MAX_LINE_STATUS_CHANGED = 99
    env.change_parameters(params)

    # Load action sets from files used by search algorithm
    def load_compressed_actionspace(path: Path) -> List[BaseAction]:
        return [action_space.from_vect(row) for row in np.load(str(path))["actions"]]

    data_path: Path = Path(__file__).parent.parent.parent / "data"

    all_unitary_actions = load_compressed_actionspace(data_path / "all_unitary_actions.npz")
    if use_triple_search:
        best_tuple_actions = load_compressed_actionspace(data_path / "best_tuple_actions.npz")

        def topo_search(enviro: BaseEnv) -> List[Tuple[float, EncodedTopologyAction]]:
            return topology_search_triples(enviro, best_tuple_actions, all_unitary_actions, show_progress=True)

    else:
        all_unitary_actions_not16 = load_compressed_actionspace(data_path / "all_unitary_actions_not16.npz")
        best_unitary_actions = load_compressed_actionspace(data_path / "best_unitary_actions300.npz")

        def topo_search(enviro: BaseEnv) -> List[Tuple[float, EncodedTopologyAction]]:
            return topology_search_tuples(
                enviro, best_unitary_actions, all_unitary_actions, all_unitary_actions_not16, show_progress=True
            )

    # Setup chronics with deterministic shuffling
    np.random.seed(seed)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    num_chronics = len(os.listdir(chronics_path))
    if chronic_limit:
        num_chronics = min(num_chronics, chronic_limit)
    for _ in range(num_chronics):
        env.reset()  # env.reset() loads the next chronic
        dst_step = 0
        logging.info(f"Scenario to test is [{env.chronics_handler.get_name()}], start from step-{dst_step:d}... ...")
        env.fast_forward_chronics(dst_step)
        obs, done = env.get_obs(), False
        while not done:
            if obs.rho.max() >= rho_acting_threshold:
                best_actions = topo_search(env)
                if len(best_actions) > 0:
                    action = best_actions[0][1].to_action(env)
                    obs_, _, done, _ = env.step(action)
                    save_sample_new(save_path, best_actions, obs, obs_, top_k=top_k)
                    obs = obs_
                else:
                    action = env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)
                    obs, _, done, _ = env.step(action)
            else:
                action = env.action_space({})
                action = find_best_line_to_reconnect(obs, action)
                obs, _, done, _ = env.step(action)
            dst_step += 1


def run_multi_teacher(
        save_path: Path,
        use_triple_search: bool,
        env_name_path: str = "l2rpn_neurips_2020_track1_small",
        rho_acting_threshold: float = 0.925,
        *,
        n_episodes: int = 100,
        top_k: int = 125,
        limit_chronics: int = None,
        jobs: int = os.cpu_count(),
        seed: int = 42,
) -> Path:
    """This teacher is trying to collect useful actions by doing topology search while running an
    environment when an overflow occurs. It also tries to combine actions to form tuple or triple actions.

    Args:
        save_path: Path to save the teacher_experience file to.
        use_triple_search: Execute the triple teacher instead of the tuple teacher.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        rho_acting_threshold: When the environments max_rho goes above this threshold the agent begins to act.
        n_episodes: Number of episodes (run through all scenarios) to run.
        limit_chronics: Limit the number of chronics to run through.
        top_k: How many of the top best actions should be saved in the teacher_experience file at save_path.
        jobs: Number of jobs to use for parallel teacher_experience collection.
        seed: The random seed to use.

    Returns:
        Saving path.
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
            (
                save_path,
                env_name_path,
                chronics_path,
                use_triple_search,
                seed + episode,
                rho_acting_threshold,
                top_k,
                limit_chronics,
            )
        )

    logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers: {tasks}")
    # Sequential implementation, useful for debugging more easily
    if jobs == 1:
        for task in tasks:
            collect_multi_experience(*task)
    else:
        # Parallel implementation
        with Pool(jobs, maxtasksperchild=1) as p:
            p.starmap(collect_multi_experience, tasks)
    end = time.time()
    elapsed = end - start
    logging.info(f"Needed time: {elapsed}s")

    return save_path



