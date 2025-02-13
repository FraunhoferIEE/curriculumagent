"""This file consist of the GeneralTutor class. In comparison to the original approach, this class offers
more variables to tweak the performance (as well as including multiple action spaces).

"""
import logging
import time
from pathlib import Path
from typing import Optional, Union, Tuple, List

import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

from curriculumagent.common.utilities import simulate_action, find_best_line_to_reconnect, split_action_and_return, \
    map_actions, revert_topo


class GeneralTutor(BaseAgent):
    """The class of the tutor agent which takes a reduced action space acts greedily using it.
    """

    def __init__(
            self,
            action_space: grid2op.Action.ActionSpace,
            action_space_file: Union[Path, List[Path]],
            do_nothing_threshold: Optional[float] = 0.925,
            best_action_threshold: Optional[float] = 0.999,
            return_status: Optional[bool] = True,
            revert_to_original_topo: Optional[bool] = False
    ):
        """Simplified __init__ method of the Tutor class.

        The required actions are either a Path variable leading to a numpy array or a list with
        multiple entries (paths). If multiple entries are supplied, the tutor executes the list sequentially.
        That means, if the threshold of the best action is met in the first set of actions from the
        list, the greedy search stops.

        Args:
            action_space: action space object from Gird2Op environment
            action_space_file: Either Numpy file with actions or List with multiple actions.
            do_nothing_threshold: Threshold, when the do nothing action is not sufficient.
            best_action_threshold: Threshold, when the collected action is sufficient and the search can
            be stopped.
            return_status: Whether each step should be logged.
            revert_to_original_topo: Should the agent revert the grid to the orignal state
            if it is stable ?

        Returns:
            None.

        """
        super().__init__(action_space=action_space)

        # Set self actions to a list to iterate for later
        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            list_of_actions = [np.load(str(Path(action_space_file)))]

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            list_of_actions = [np.load(str(act_path)) for act_path in action_space_file]

        # Now map action ids:
        self.actions = map_actions(list_of_actions)

        self.do_nothing_threshold = do_nothing_threshold
        self.action_threshold = best_action_threshold
        self.return_status = return_status
        self.next_actions = None
        self.revert_to_original_topo = revert_to_original_topo

    def act_with_id(self, observation: BaseObservation) -> Tuple[np.ndarray, int]:
        """Compute greedy search of Tutor.

        In this greedy search, we iterate over the keys of the dictionary.
        If the threshold is met, the iteration stops and the best action is returned.

        Args:
            observation: Grid2Op observation.

        Returns:
            Best action as numpy array as well as a tuple index for the action.

        """
        start_time = time.time()
        out = self.action_space({}).to_vect(),-1
        # Check for do nothing
        if observation.rho.max() < self.do_nothing_threshold:
            # Early return if below threshold return "do nothing" in bus switches.
            if self.revert_to_original_topo:
                out =  revert_topo(self.action_space, observation),-1
            return out

        # Take lower values of either rho max or do nothing rho max
        obs_dn, _, _, _ = observation.simulate(self.action_space({}))

        min_rho = observation.rho.max()
        old_rho_max = min_rho
        action_chosen = None
        best_action_index = -1

        # Run through the list of actions
        # Set len act to 0 in order to assert the correct idx:

        for actions in self.actions:
            # actions is a dictionary with the actions
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

        return out

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """Compute greedy search of Tutor.

        This is wrapper for the act_with_id method.

        Args:
            observation: Grid2Op observation.

        Returns:
            Returns a base action.

        """
        next_action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
            except StopIteration:
                self.next_actions = None

        if self.next_actions is None:
            act_array, _ = self.act_with_id(observation=observation)

            # Create generator
            self.next_actions = split_action_and_return(observation, self.action_space, act_array)
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)

        return next_action


def print_status(observation, best_action_index: int, old_rho_max, min_rho, start_time) -> None:
    """Logging values for evaluation.

    Args:
        observation: Observation of Grid2Op.
        best_action_index: Combined index of the action.
        old_rho_max: Previous rho max that triggered the tutor.
        min_rho: The best rho value from the actions.
        start_time: Time variable for printing.

    Returns:
        None.

    """
    rho_improvement = old_rho_max - min_rho
    if best_action_index != -1:
        logging.info(
            f"t={observation.get_time_stamp()}, line={observation.rho.argmax():03d} overflowed"
            f"=> Use action {best_action_index} => rho_delta: {rho_improvement:.3f}"
            f"({old_rho_max:.2f} -> {min_rho:.2f}) [time: {time.time() - start_time:.02f}s]"
        )
    else:
        logging.info(
            f"t={observation.get_time_stamp()}, line={observation.rho.argmax():03d} overflowed" f"=> No action found!"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
