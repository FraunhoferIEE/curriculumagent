"""This file consist of the N-1 Tutor, that executes the actions similar to the N-1 Teacher.

"""
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import grid2op
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation

from curriculumagent.common.utilities import simulate_action, find_best_line_to_reconnect, revert_topo
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor, print_status


class NminusOneTutor(GeneralTutor):
    """The N-1 Tutor that searches through the actions, while checking whether the action is N-1 compatible.

    """

    def __init__(
            self,
            action_space: grid2op.Action.ActionSpace,
            action_space_file: Union[Path, List[Path]],
            do_nothing_threshold: Optional[float] = 0.9,
            best_action_threshold: Optional[float] = 0.99,
            rho_greedy_threshold: Optional[float] = 0.99,
            lines_to_check: Optional[Union[List[int], bool]] = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
            return_status: Optional[bool] = True,
            revert_to_original_topo: Optional[bool] = False,
    ):
        """Simplified constructor method of the Tutor class

        The required actions are either a Path variable leading to a numpy array or a list with
        multiple entries (paths). If multiple entries are supplied, the tutor concatenates the actions to one
        combined list.

        Args:
            action_space: Action space object from Gird2Op environment.
            action_space_file: Either Numpy file with actions or List with multiple actions.
                If multiple action sets are supplied, the first is used for the N-1 search.
            do_nothing_threshold: Threshold, when the do nothing action is not sufficient.
            rho_greedy_threshold: Threshold, between do_nothing_threshold and best_action_threshold.
                If the max_rho is below this threshold, we check for N-1 search possibility.
            best_action_threshold: Threshold, when the collected action is sufficient and the search can
            be stopped. This is used for the greedy actions.
            lines_to_check: Indicator, whether the lines should be checked for N-1 actions.
            Can either be a boolean (to check all or none) or a specific list containing the
            different lines.
            return_status: Whether each step should be logged
            revert_to_original_topo: Boolean, whether to revert to original topology.

        Returns:
            None.

        """
        super().__init__(
            action_space=action_space,
            action_space_file=action_space_file,
            do_nothing_threshold=do_nothing_threshold,
            best_action_threshold=best_action_threshold,
            return_status=return_status,
        )

        if isinstance(lines_to_check, list):
            # List
            self.lines_to_attack = lines_to_check
            # Bool:
        elif lines_to_check:
            self.lines_to_attack = list(np.arange(action_space.n_line))
        else:
            self.lines_to_attack = []

        if (rho_greedy_threshold <= best_action_threshold) and (do_nothing_threshold <= rho_greedy_threshold):
            self.rho_greedy_threshold = rho_greedy_threshold
        else:
            logging.warning(
                "The provided rho_greedy_threshold is not inbetween the do_nothing_threshold"
                "and best_action_threshold. Set variable to default"
            )
            self.rho_greedy_threshold = 0.99

        self.revert_to_original_topo = revert_to_original_topo

    def act_with_id(self, observation: BaseObservation) -> Tuple[np.ndarray, int]:
        """In this methode, we execute the N-1 Action search.

        If no line is disconnected , we check for each action which would be the most stable N-1 result
        (i.e. which returns the min rho the most fatal line gets disconnected).
        Then this action is selected.
        If no action is found OR one line is already disconnected, we run the greedy action similar to
        the general_tutor.

        Note: If multiple action sets are provided, the first is used for the N-1 search.

        Args:
            observation: Grid2Op observation.

        Returns:
            Best action as numpy array as well as a tuple index for the action.

        """
        start_time = time.time()

        # Check for do nothing
        if observation.rho.max() < self.do_nothing_threshold:
            # secure, return "do nothing" in bus switches.

            if self.revert_to_original_topo:
                act = revert_topo(self.action_space, observation)
            else:
                act = self.action_space({}).to_vect()

            return act, -1

        # Take lower values of either rho max or do nothing rho max
        obs_dn, _, _, _ = observation.simulate(self.action_space({}))

        min_rho = observation.rho.max()
        old_rho_max = min_rho.copy()
        action_chosen = None
        best_action_index = -1
        idx = None

        # N-1 Search IF no disconnected line ! or/and limit breached.
        if self.__n_minus_one_search_possible(obs=observation):
            rho_max_max = np.inf
            for idx, action_array in self.actions[0].items():
                act = self.action_space.from_vect(action_array)
                current_rho_max_max = self.calculate_attacked_max_rho(obs=observation, action=act)
                if current_rho_max_max < rho_max_max:
                    rho_max_max = current_rho_max_max
                    action_chosen = action_array
                    best_action_index = idx

            if action_chosen is not None:
                # Now look at the best option and again simulate the effect of the action prior to returning it.
                rho_max, valid_action = simulate_action(
                    action_vect=action_chosen, action_space=self.action_space, obs=observation
                )
                if rho_max < min_rho and valid_action:
                    if self.return_status:
                        print_status(observation, best_action_index, observation.rho.max(), rho_max, start_time)
                    logging.info(f"Gathered a N-1 Action with the id{idx}")
                    return action_chosen, best_action_index

        action_chosen = None

        for actions in self.actions:
            # Run through action set
            for idx, action_array in actions.items():
                obs_sim, valid_action = simulate_action(
                    action_vect=action_array, action_space=self.action_space, obs=observation
                )
                if not valid_action:
                    continue

                # Simulate action (even though it might be illegal for tuple or triple action)
                if obs_sim < min_rho:
                    min_rho = obs_sim
                    action_chosen = action_array
                    best_action_index = idx

            if min_rho <= self.action_threshold:
                break

        if self.return_status:
            print_status(observation, best_action_index, old_rho_max, min_rho, start_time)

        if action_chosen is not None:
            out = action_chosen, best_action_index

        else:
            out = self.action_space({}).to_vect(), -1
        return out

    def calculate_attacked_max_rho(self, obs: BaseObservation, action: BaseAction) -> float:
        """Method to run through all lines and check the maximum rho if the lines is disconnected.
        Note that if the game is done, the rho will be set to np.inf
        This is to measure how robust the supplied action is against the attacks, the lower, the better.

        Args:
            obs: Observation of the Grid2Op Env.
            action: Current selected action.

        Returns:
            The maximum rho of the action.

        """
        p_max_max = 0.0
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

    def __n_minus_one_search_possible(self, obs: grid2op.Observation.BaseObservation) -> bool:
        """Small private method, that checks whether the N-1 Search is actually possible.

        Args:
            obs: Current observation.

        Returns:
             Boolean, whether the N-1 Search can be computed.

        """
        search_possible = True

        #  Check if below threshold:
        if obs.rho.max() > self.rho_greedy_threshold:
            search_possible = False
        # Check if any lines are disconnected:
        if not all(obs.line_status):
            search_possible = False
        # Check if actually any lines can be checked:
        if len(self.lines_to_attack) == 0:
            search_possible = False
        return search_possible
