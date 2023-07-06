"""This file contains the N-1 Teacher which tries to find actions that make the power network more robust to attacks.

For this teacher it is possible to enforce some line disconnection.
"""
import logging
import multiprocessing
import os
import time
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from typing import Union, Optional, List, Tuple

import defopt
import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from curriculumagent.teacher.submodule.common import save_sample_new
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from curriculumagent.teacher.submodule.topology_action_search import topology_search_topk


class NMinusOneTeacher:
    """Class to collect the N-1 data.

    """

    def __init__(
            self,
            lines_to_attack: Optional[List[float]] = None,
            rho_n0_threshold: Optional[float] = 0.9,
            rho_max_threshold: Optional[float] = 1.0,
            seed: Optional[int] = None,
    ):
        """Constructor method of the N-1 Teacher.

        Difference between rho_n0_threshold and rho_max_threshold is that for the first,
        the N-1 Search starts, while the latter is necessary to just use the best greedy action.

        Args:
            lines_to_attack: Lines to select for the N-1 check.
            rho_n0_threshold: Threshold, when the do-nothing agent should stop and start searching for N-1.
            rho_max_threshold: Threshold, when the N-1 search should stop and only the best greedy.
            action should be selected.
            seed: Optional seed for the Grid2Op Environment and random sampling.

        Returns:
            None.
        """
        if lines_to_attack is None:
            lines_to_attack = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]
        self.lines_to_attack = lines_to_attack
        self.rho_n0 = rho_n0_threshold
        self.rho_threshold = rho_max_threshold

        # Set internally:
        self.seed = seed
        self.top_k: int = 100
        self.all_actions: Optional[List[BaseAction]] = None  # Used to cache all_actions, see method below

    def get_all_actions(self, env: BaseEnv) -> List[BaseAction]:
        """Set and get all actions for the action space of the provided environment.

        Args:
            env: Grid2Op Environment

        Returns:
            All possible actions.

        """
        if self.all_actions is None:
            self.all_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)

        return self.all_actions

    def do_nothing_and_run_through_lines_action(self, env: BaseEnv, obs: BaseObservation) -> BaseAction:
        """Method to check, whether a disconnection of the lines would result in to a breach of the
        rho_threshold.

        For this step, run through all lines and check which disconnection would result the most
        problematic rho value. If found, return that action.

        Note:
            Only values between the rho_n0 and the rho_max value are accepted.

        Args:
            env: Grid2Op environment.
            obs: Observation of the Grid2Op Environment.

        Returns:
            Action with problematic rho value.
        """
        cur_rho = self.rho_n0
        chosen_action = env.action_space({})
        # What does it do exactly?
        for line in self.lines_to_attack:
            new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
            new_line_status_array[line] = -1
            action = env.action_space({"set_line_status": new_line_status_array})
            obs_sim, _, done, _ = obs.simulate(action)
            if (obs_sim.rho.max() > cur_rho) and obs_sim.rho.max() < self.rho_threshold and done is False:
                cur_rho = obs_sim.rho.max()
                chosen_action = action
        return chosen_action

    def calculate_attacked_max_rho(self, obs: BaseObservation, action: BaseAction) -> float:
        """Method to run through all lines and check the maximum rho  if the lines is disconnected.

        This is to measure how robust the supplied action is against the attacks, the lower, the better.

        Note:
            If the game is done, the rho will be set to np.inf.

        Args:
            obs: Observation of the Grid2Op Environment.
            action: Current selected action.

        Returns:
            The maximum rho of the action.

        """
        p_max_max = 0
        for line in self.lines_to_attack:
            new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
            new_line_status_array[line] = -1
            disconnect_action = deepcopy(action)
            disconnect_action.update({"set_line_status": new_line_status_array})
            obs_new, _, done, _ = obs.simulate(disconnect_action)
            rho_max = obs_new.rho.max()
            if rho_max > p_max_max and done is False:
                p_max_max = rho_max
            if done:
                p_max_max = np.inf

        return p_max_max

    def search_best_n_minus_one_action(self, env: BaseEnv, obs: BaseObservation) -> Tuple[BaseAction, bool]:
        """Method to collect the best actions that might be N-1.

        Args:
            env: The environment to collect the actions in.
            obs: Observation of the Grid2Op Env.

        Returns:
            The selected action as well as boolean which indicates whether the action was N-1.
            The selected action can either be the N-1 action or the best greedy action.
        """
        # First select the best greedy actions:
        all_actions = self.get_all_actions(env)
        greedy_action_set = topology_search_topk(env, obs, all_actions, top_k=self.top_k)
        # No good action found -> abort
        if len(greedy_action_set) == 0:
            return env.action_space({}), False

        # Now run simulate then N-1 actions

        # Only consider actions that decrease the max_rho below rho_n0
        action_set = [new_act for rho_delta, new_act in greedy_action_set if (obs.rho.max() - rho_delta) < self.rho_n0]

        n_1_action_found = False
        if action_set:
            chosen_action = action_set[0]
            rho_max_max = np.inf  # Note: This can be problematic?
            for act in action_set:
                current_rho_max_max = self.calculate_attacked_max_rho(obs=obs, action=act)
                if current_rho_max_max < rho_max_max:
                    rho_max_max = current_rho_max_max
                    chosen_action = act
                    n_1_action_found = True

        if not n_1_action_found:
            chosen_action = greedy_action_set[0][1]

        return chosen_action, n_1_action_found

    def n_minus_one_agent(
            self,
            env_path: str,
            chronics_path: str,
            chronics_id: Union[str, int],
            save_path: Path,
            save_greedy: bool = False,
            active_search: bool = False,
            disable_opponent: bool = False,
    ):

        """Running the N-1 Agent Search for specific chronic.

        For this purpose, we first reinitialize the environment, set a special chronics_id and
        then run the N-1 agent.

        Args:
            env_path: Path of the Grid2Op Environment as a str.
            chronics_path: Path of the chronics.
            chronics_id: ID of the chronic, which the env should be run on.
            save_path: Where to save the selected actions.
            save_greedy: Whether the greedy action should be saved, if False - only the N-1 actions are saved.
            active_search: Variable, whether the do-nothing agent should disconnect the lines.
            disable_opponent: Whether to disable the opponent of the environment or not. This is principally a
            good idea when you activate active search. In that case, only the N-1 agent should deactivate the
            lines.

        Returns:
            Action set that are N-1.
        """
        try:
            # if lightsim2grid is available, use it.
            from lightsim2grid import LightSimBackend

            backend = LightSimBackend()
        except ImportError:  # noqa
            backend = PandaPowerBackend()
            logging.warning("Not using lightsim2grid! Operation will be slow!")

        kwargs = dict()
        if disable_opponent:
            kwargs["opponent_budget_per_ts"] = 0
            kwargs["opponent_init_budget"] = 0

        if chronics_path:
            env = grid2op.make(dataset=env_path, chronics_path=chronics_path, backend=backend, **kwargs)
        else:
            env = grid2op.make(dataset=env_path, backend=backend, **kwargs)

        if self.seed:
            env.seed(self.seed)

        # Set chronic:
        env.set_id(chronics_id)
        env.reset()
        logging.info(f"[{env.chronics_handler.get_name()}] N-1 Search started")

        done = False
        obs = env.get_obs()
        # Run through episode:

        import warnings

        warnings.filterwarnings("ignore")

        while not done:
            save_action = False
            rho_max = obs.rho.max()
            if rho_max < self.rho_n0:  # < 0.9
                # Do Nothing and disconnect lines if necessary
                # Added only one line disconnect
                if active_search and all(obs.line_status):
                    # Why is it helping considering the most problematic action?
                    action = self.do_nothing_and_run_through_lines_action(env=env, obs=obs)
                else:
                    action = env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)

            # If no line is disconnected and the rho lies below the threshold, we continue with the n-1 search.
            elif (rho_max < self.rho_threshold) and all(obs.line_status):  # < 1.0
                # Sort through N-1 actions becuse rho_n0 was breached:
                logging.info(
                    f"[{env.chronics_handler.get_name()}] ({obs.current_step}/{obs.max_step}) (rho={rho_max}): Looking for N-1 action"
                )
                action, n_1_action_found = self.search_best_n_minus_one_action(env=env, obs=obs)
                action = find_best_line_to_reconnect(obs, action)
                if n_1_action_found or (n_1_action_found is False and save_greedy):
                    save_action = True

            else:
                logging.info(
                    f"[{env.chronics_handler.get_name()}] ({obs.current_step}/{obs.max_step}) (rho={rho_max}): Looking for "
                    f"greedy action"
                )
                all_actions = self.get_all_actions(env)
                greedy_action_set = topology_search_topk(env, obs, all_actions, top_k=self.top_k)
                if len(greedy_action_set) == 0:
                    action = env.action_space({})
                    save_action = False
                else:
                    action = greedy_action_set[0][1]
                    save_action = save_greedy
                action = find_best_line_to_reconnect(obs, action)

            obs_, _, done, _ = env.step(action)
            # logging.info(f'[{env.chronics_handler.get_name()}] {obs.current_step}/{obs.max_step}: {done}')

            # Calculate rho_improvement for saving
            rho_max_old = obs.rho.max()
            rho_max_new = obs_.rho.max()
            rho_improvement: float = rho_max_old - rho_max_new

            if save_action:
                save_sample_new(Path(save_path), [(rho_improvement, EncodedTopologyAction(action))], obs, obs_, top_k=1)

            obs = obs_
        logging.info(f"[{env.chronics_handler.get_name()}] Search stopped at {obs.current_step}/{obs.max_step}")

    def collect_n_minus_1_experience(
            self,
            save_path: Path,
            env_name_path: Path,
            number_of_years: Optional[int] = None,
            jobs: int = -1,
            save_greedy: bool = False,
            active_search: bool = True,
            disable_opponent: bool = True,
    ):

        """Collect useful actions by attacking/disconnecting specific lines at random timesteps
        and trying to find the best action to resolve that problem.

        Args:
            save_path: Path to save the teacher_experience file to.
            env_name_path: Path to Grid2Op dataset or the standard name of it.
            number_of_years: Optional number of years to restrict the cronics.
            jobs: Number of jobs to use for parallel teacher_experience collection. If set to -1, jobs is set
                to the number of CPUs
            save_greedy: Whether to save the greedy actions as well
            active_search: Should the N-1 agent be forced with line outages or not
            disable_opponent: Should the opponent be disabled or not

        Returns:
            None.
        """
        if jobs == -1:
            jobs = os.cpu_count()

        env_name_path = Path(env_name_path).expanduser()
        chronics_path = None
        if env_name_path.is_dir():
            chronics_path = str(env_name_path / "chronics")
        env_name_path = str(env_name_path)

        try:
            # if lightsim2grid is available, use it.
            from lightsim2grid import LightSimBackend

            backend = LightSimBackend()
        except ImportError:  # noqa
            backend = PandaPowerBackend()
            logging.warning("Not using lightsim2grid! Operation will be slow!")

        if chronics_path:
            # Use env_name_path as path together with chronics_path
            env = grid2op.make(dataset=env_name_path, chronics_path=chronics_path, backend=backend)
        else:
            # Use env_name_path as name
            env = grid2op.make(dataset=env_name_path, backend=backend)

        tasks = []
        if number_of_years:
            train_chronics = [
                chronic for chronic in env.chronics_handler.subpaths if int(chronic[-3:]) < number_of_years
            ]
        else:
            train_chronics = env.chronics_handler.subpaths

        for chronic_name in train_chronics:
            tasks.append(
                (env_name_path, chronics_path, chronic_name, save_path, save_greedy, active_search, disable_opponent)
            )
        logging.info(
            f"These {len(tasks)} scenarios will be trained using a pool of {jobs} workers:\n {pformat(train_chronics)}"
        )
        start = time.time()
        if jobs == 1:
            for task in tasks:
                self.n_minus_one_agent(*task)

        with Pool(jobs) as p:
            p.starmap(self.n_minus_one_agent, tasks)

        end = time.time()
        elapsed = end - start
        logging.info(f"Time: {elapsed}s")



