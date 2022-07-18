"""
In this file, an expert agent (named Tutor), which does a greedy search
in the reduced action space  is built (default of original code is 208 actions).
It receives an observation, and returns the action that decreases the rho
most, as well as its index [api: Tutor.act(obs)].

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""

import os
import logging
import time
from pathlib import Path
from typing import Union, Optional

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

from curriculumagent.common.utilities import array2action, is_legal


class Tutor(BaseAgent):
    """
    The class of the tutor agent which takes a reduced action space acts greedily using it.
    """

    def __init__(self, action_space: BaseAction, action_space_file: Optional[Path] = None,
                 old_actionspace_path: Optional[Path] = None, tuple_actions_count: Optional[int] = None):
        """ __init__ method of the Tutor class

        Here we specifically set the paths for all the actions of the tutor.

        Args:
            action_space: action space object from Gird2Op environment
            action_space_file: Optional file, where to find that selected action spaces from the teacher
            old_actionspace_path: Old action space path, consisting of action from Binbinchen
            tuple_actions_count: If this is set we handle action_space_file as if it contains tuple actions.
                                 The agent will try to apply the tuple actions first.
        """
        BaseAgent.__init__(self, action_space=action_space)
        self.use_new_actions = action_space_file is not None
        if action_space_file:
            assert action_space_file.is_file(), f"Actionspace file {action_space_file} exists"
            self.actions = np.load(str(Path(action_space_file)))
            self.unitary_actions = []
        elif old_actionspace_path:
            assert old_actionspace_path.is_dir(), f"Path at {old_actionspace_path} is a directory"
            assert not tuple_actions_count, "Can't use tuple actions with old actionspace"
            self.actions62 = np.load(str(old_actionspace_path / 'actions62.npy'))
            self.actions146 = np.load(str(old_actionspace_path / 'actions146.npy'))

        self.tuple_actions_count = tuple_actions_count
        self.action_space_size = self.action_space.size()

    @staticmethod
    def reconnect_array(obs) -> np.ndarray:
        """ Checking, whether lines can be reconnected.

        If lines are found, they are reconnected automatically.

        Args:
            obs: observation of Grid2Op

        Returns: Array containing possible reconnection.

        """
        new_line_status_array = np.zeros_like(obs.rho, dtype=np.int)
        disconnected_lines = np.where(obs.line_status == False)[0]  # pylint: disable=singleton-comparison
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                # this line is disconnected, and, it is not cooling down.
                line_to_reconnect = line
                new_line_status_array[line_to_reconnect] = 1
                break  # reconnect the first one
        return new_line_status_array

    @staticmethod
    def print_status(observation, best_action_index, old_rho_max, min_rho, start_time) -> None:
        """ Logging values for evaluation

        Args:
            observation: observation of Grid2Op
            best_action_index: index of the best action for the Grid2Op file
            old_rho_max: previous rho max that triggered the tutor
            min_rho: best rho value from the actions
            start_time: time variable for printing

        Returns: None,

        """
        rho_improvement = old_rho_max - min_rho
        if best_action_index != -1:
            logging.info(
                f"t={observation.get_time_stamp()}, line={observation.rho.argmax():03d} overflowed"
                f"=> Use action {best_action_index} => rho_delta: {rho_improvement:.3f}"
                f"({old_rho_max:.2f} -> {min_rho:.2f}) [time: {time.time() - start_time:.02f}s]")
        else:
            logging.info(f"t={observation.get_time_stamp()}, line={observation.rho.argmax():03d} overflowed"
                         f"=> No action found!")

    def _act_tuple(self, observation: BaseObservation) -> BaseAction:
        """
        Perform a tuple search on the environment. It prioritizes tuple actions first.

        Args:
            observation: The current observation fo the environment.

        Returns: The tuple or unitary action that improves the performance.

        """
        start_time = time.time()
        reconnect_array = self.reconnect_array(observation)

        if observation.rho.max() < 0.925:
            # secure, return "do nothing" in bus switches.
            return self.action_space({'set_line_status': reconnect_array})

        # not secure, do a greedy search
        old_rho_max = observation.rho.max()
        min_rho = old_rho_max
        action_chosen = None
        best_action_index = -1

        # Search through all tuple actions first
        for idx, action_array in enumerate(self.actions[:self.tuple_actions_count]):
            a = array2action(self.action_space, action_array, reconnect_array)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                best_action_index = idx
        if min_rho <= 0.999:
            self.print_status(observation, best_action_index, old_rho_max, min_rho, start_time)
            return action_chosen if action_chosen else \
                array2action(self.action_space, np.zeros(self.action_space_size),
                             reconnect_array)

        # If they don't improve the situation, try a unitary action after that
        for idx, action_array in enumerate(self.actions[self.tuple_actions_count:]):
            a = array2action(self.action_space, action_array, reconnect_array)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                best_action_index = idx

        self.print_status(observation, best_action_index, old_rho_max, min_rho, start_time)
        return action_chosen if action_chosen else array2action(self.action_space, np.zeros(self.action_space_size),
                                                                reconnect_array)

    def _act_old(self, observation: BaseObservation) -> BaseAction:
        """
        Perform the unitary action search through the two hierarchies of the original 208 actions(62 + 146).

        Args:
            observation: The current observation fo the environment.

        Returns: The unitary action that improves the performance.

        """
        start_time = time.time()
        reconnect_array = self.reconnect_array(observation)

        if observation.rho.max() < 0.925:
            # secure, return "do nothing" in bus switches.
            return self.action_space({'set_line_status': reconnect_array})

        # not secure, do a greedy search
        old_rho_max = observation.rho.max()
        min_rho = old_rho_max
        action_chosen = None
        best_action_index = -1

        # hierarchy-1: 62 actions.
        for idx, action_array in enumerate(self.actions62):
            a = array2action(self.action_space, action_array, reconnect_array)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                best_action_index = idx
        if min_rho <= 0.999:
            self.print_status(observation, best_action_index, old_rho_max, min_rho, start_time)
            return action_chosen if action_chosen else \
                array2action(self.action_space, np.zeros(self.action_space_size),
                             reconnect_array)
        # hierarchy-2: 146 actions.
        for idx, action_array in enumerate(self.actions146):
            a = array2action(self.action_space, action_array, reconnect_array)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                best_action_index = idx + 62

        self.print_status(observation, best_action_index, old_rho_max, min_rho, start_time)
        return action_chosen if action_chosen else array2action(self.action_space, np.zeros(self.action_space_size),
                                                                reconnect_array)

    def _act_simple(self, observation: BaseObservation) -> BaseAction:
        """
        Perform a simple search through the reduced action space returning the unitary action with the best rho.

        Args:
            observation: The current observation fo the environment.

        Returns: The unitary action that improves the performance.

        """
        start_time = time.time()
        reconnect_array = self.reconnect_array(observation)

        if observation.rho.max() < 0.925:
            # secure, return "do nothing" in bus switches.
            return self.action_space({'set_line_status': reconnect_array})

        # not secure, do a greedy search
        old_rho_max = observation.rho.max()
        min_rho = old_rho_max
        action_chosen = None
        best_action_index = -1

        # Simply search through all actions
        for idx, action_array in enumerate(self.actions):
            a = array2action(self.action_space, action_array, reconnect_array)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                best_action_index = idx

        self.print_status(observation, best_action_index, old_rho_max, min_rho, start_time)
        return action_chosen if action_chosen else array2action(self.action_space, np.zeros(self.action_space_size),
                                                                reconnect_array)

    def act(self, observation: BaseObservation, reward, done=False) -> BaseAction:
        """ Compute greedy search of Tutor

        Args:
            observation: Grid2Op observation

        Returns: best action

        """
        if self.use_new_actions:
            if self.tuple_actions_count:
                return self._act_tuple(observation)
            else:
                return self._act_simple(observation)
        else:
            return self._act_old(observation)


def generate_original_tutor_data(env_name_path: Union[Path, str],
                                 save_path: Union[Path, str],
                                 action_paths: Path,
                                 num_chronics: Optional[int] = None,
                                 seed: Optional[int] = None):
    """ Method to run the orignal tutor, similar to the general tutor. Note: The orignal tutor
    is depriciated and will be removed in later versions.

    Args:
        env_name_path: Path to grid2op dataset or the standard name of it.
        save_path: Where to save the experience
        action_paths: List of action sets (in .npy format)
        num_chronics: Total numer of chronics
        seed: Whether to set a seed to the sampling of environments

    Returns: Experience of the tutor

    """

    scenarios_path = Path(env_name_path) / 'chronics'

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=str(env_name_path), chronics_path=str(scenarios_path), backend=backend)
    except Exception:
        env = grid2op.make(dataset=str(env_name_path), chronics_path=str(scenarios_path))
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])

    if seed:
        env.seed(seed)
        np.random.seed(seed)

    tutor = Tutor(env.action_space, action_paths, seed=seed)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    obs_size = env.observation_space.size()
    records = np.zeros((1, 1 + obs_size), dtype=np.float32)
    for num in range(num_chronics):
        env.reset()
        logging.info(f'current chronic:{env.chronics_handler.get_name()}')
        done, step, obs = False, 0, env.get_obs()
        while not done:
            action, idx = tutor.act(obs)
            if idx != -1:
                # TODO: This is probably slow because it reallocates memory,
                #       should be using a list of ndarrays and concat them at the end or when saving
                # save a record
                records = np.concatenate((records, np.concatenate(([idx], obs.to_vect())).astype(np.float32)[None, :]),
                                         axis=0)
            obs, _, done, _ = env.step(action)
            step += 1
        logging.info(f'game over at step-{step}\n\n\n')

        save_internal = 10
        # save current records
        if (num + 1) % save_internal == 0:
            filepath = os.path.join(save_path, 'records_%s.npy' % (time.strftime("%m-%d-%H-%M", time.localtime())))
            np.save(filepath, records)
            logging.info('# records are saved! #')


def main():
    """ Main Method to run the tutor.
    """
    # hyper-parameters
    # DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    # SCENARIO_PATH = '../training_data_track1/chronics'

    DATA_PATH = Path('~/data_grid2op/l2rpn_neurips_2020_track1_small').expanduser()

    SAVE_PATH = '../../junior/training_data'
    ACTION_SPACE_DIRECTORY = '../action_space'
    NUM_CHRONICS = 100
    generate_original_tutor_data(env_name_path=DATA_PATH,
                                 save_path=SAVE_PATH,
                                 action_paths=ACTION_SPACE_DIRECTORY,
                                 num_chronics=NUM_CHRONICS,
                                 seed=42)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
