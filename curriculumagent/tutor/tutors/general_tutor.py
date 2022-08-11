"""
This file consist of the GeneralTutor class. In comparison to the original approach, this class offers
more variables to tweak the performance (as well as including multiple action spaces).
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
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from curriculumagent.common.utilities import array2action, is_legal, find_best_line_to_reconnect, \
    split_and_execute_action, split_action_and_return


class GeneralTutor(BaseAgent):
    """
    The class of the tutor agent which takes a reduced action space acts greedily using it.
    """

    def __init__(self, action_space: BaseAction,
                 action_space_file: Union[Path, List[Path]],
                 do_nothing_threshold: Optional[float] = 0.9,
                 best_action_threshold: Optional[float] = 0.95,
                 return_status: Optional[bool] = True):
        """ Simplified __init__ method of the Tutor class

        The required actions are either a Path variable leading to a numpy array or a list with
        multiple entries. If multiple entries are supplied, the tutor concatenates the actions to one
        combined list.

        Args:
            action_space: action space object from Gird2Op environment
            action_space_file: Either Numpy file with actions or List with multiple actions.
            do_nothing_threshold: Threshold, when the do nothing action is not sufficient.
            best_action_threshold: Threshold, when the collected action is sufficient and the search can
            be stopped.
            return_status: Whether or not each step should be logged
        """
        BaseAgent.__init__(self, action_space=action_space)

        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            self.actions = np.load(str(Path(action_space_file)))

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            list_of_actions = [np.load(str(act_path)) for act_path in action_space_file]
            self.actions = np.concatenate(list_of_actions,axis=0)

        self.action_space_size = self.action_space.size()
        self.do_nothing_threshold = do_nothing_threshold
        self.action_threshold = best_action_threshold
        self.return_status = return_status
        self.next_actions = None

    def act_with_id(self, observation: BaseObservation) -> Tuple[np.ndarray, int, int]:
        """ Compute greedy search of Tutor

        In this greedy search, we iterate over the keys of the dictionary .
        If the threshhold is met, the iteration stops and the best action is returned.

        Args:
            observation: Grid2Op observation

        Returns: best action as numpy array as well as a tuple index for the action

        """
        start_time = time.time()

        # Check for do nothing
        if observation.rho.max() < self.do_nothing_threshold:
            # secure, return "do nothing" in bus switches.
            return np.zeros(self.action_space_size), -1

        # Not secure, do a greedy search
        old_rho_max = observation.rho.max()
        min_rho = old_rho_max
        action_chosen = None
        best_action_index = -1


        # Run through action set
        for idx, action_array in enumerate(self.actions):
            a = array2action(self.action_space, action_array)
            a = find_best_line_to_reconnect(obs=observation,
                                            original_action=a)

            if not is_legal(a, observation):
                continue

            # Simulate action (even though it might be illegal for tuple or triple action)
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = action_array
                best_action_index = idx

        if self.return_status:
            print_status(observation, best_action_index, old_rho_max, min_rho, start_time)

        if action_chosen is not None:
            out = action_chosen, best_action_index

        else:
            out = np.zeros(self.action_space_size), -1

        return out

    def act(self, observation: BaseObservation, reward, done=False) -> BaseAction:
        """ Compute greedy search of Tutor

        This is wrapper for the act_with_id method.

        Args:
            observation: Grid2Op observation

        Returns: Returns a base action

        """
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

        return next_action


def collect_tutor_experience_one_chronic(action_paths: Union[Path, List[Path]],
                                         chronics_id: int,
                                         env_name_path: Union[Path, str] = 'l2rpn_neurips_2020_track1_small',
                                         seed: Optional[int] = None,
                                         enable_logging: bool = True,
                                         TutorAgent: BaseAgent = GeneralTutor):
    """Collect tutor experience of one chronic

    Args:
        action_paths: List of Paths for the tutor.
        chronics_id: Number of chronic to run.
        env_name_path: Path to grid2op dataset or the standard name of it.
        seed: Whether to init the grid2op env with a seed
        enable_logging: Whether or not to log the Tutor experience search
        TutorAgent: Tutor Agent which should be used for the search
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
        env.seed(seed)

    env.set_id(chronics_id)
    env.reset()
    logging.info(f'current chronic:{env.chronics_handler.get_name()}')

    # After initializing the environment, let's init the tutor
    tutor = GeneralTutor(action_space=env.action_space,
                         action_space_file=action_paths)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    obs_size = env.observation_space.size()
    records = np.zeros((1, 1 + obs_size), dtype=np.float32)
    done, step, obs = False, 0, env.get_obs()
    while not done:
        action, idx = tutor.act_with_id(obs)
        if action.any():  # any element is nonzero => action is doing something
            # Record choice of tutor:
            # Note: Depending on the number of list is the first column either only zeros or the specific ID
            records = np.concatenate((records,
                                      np.hstack([idx, obs.to_vect()]).astype(np.float32).reshape(1, -1)),
                                     axis=0)

            # Execute Action:
            # This method does up to three steps and returns the output
            obs, _, done, _ = split_and_execute_action(env=env, action=action)
            step = env.nb_time_step
            logging.info(f'game over at step-{step}')

        else:
            # Use Do-Nothing Action
            act_with_line = find_best_line_to_reconnect(obs, env.action_space({}))
            obs, _, done, _ = env.step(act_with_line)

    return records

    # save current records


def generate_tutor_experience(env_name_path: Union[Path, str],
                              save_path: Union[Path, str],
                              action_paths: Union[Path, List[Path]],
                              num_chronics: Optional[int] = None,
                              num_sample: Optional[int] = None,
                              jobs: int = -1,
                              seed: Optional[int] = None,
                              TutorAgent: BaseAgent = GeneralTutor):
    """ Method to run the Tutor in parallel

    Args:
        env_name_path: Path to grid2op dataset or the standard name of it.
        save_path: Where to save the experience
        action_paths: List of action sets (in .npy format)
        num_chronics: Total numer of chronics
        num_sample: length of sample from the num_chronics. If num_sample is smaller than num chronics,
        a subset is taken. If it is larger, the chronics are sampled with replacement
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments
        TutorAgent: Tutor Agent which should be used for the search, default is the GeneralTutor

    Returns: None, Saves results as numpy file.

    """
    log_format = '(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]'
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
        save_path = save_path / f'tutor_experience_{now}.npy'

    # TODO: Should use savez_compressed since data has low entropy, have to adapt junior as well
    np.save(save_path, all_experience)
    logging.info(f'Tutor experience has been saved to {save_path}')


def main():
    DATA_PATH = Path('~/data_grid2op/l2rpn_neurips_2020_track1_small').expanduser()

    SAVE_PATH = '../../junior/training_data'
    NUM_CHRONICS = 100
    TOTAL_CHRONICS = len(os.listdir(DATA_PATH / 'chronics'))

    test_env_path = Path(__file__).parent.parent.parent / "tests" / "data" / "action_spaces"
    action_list = [test_env_path / "test_tuple.npy",
                   test_env_path / "test_tripple.npy",
                   test_env_path / "test_single.npy"]
    print(action_list)
    generate_tutor_experience(env_name_path=DATA_PATH,
                              action_paths=action_list,
                              save_path=SAVE_PATH,
                              num_chronics=TOTAL_CHRONICS,
                              num_sample=NUM_CHRONICS)


def print_status(observation, best_action_index: Tuple[int, int], old_rho_max, min_rho, start_time) -> None:
    """ Logging values for evaluation

    Args:
        observation: observation of Grid2Op
        best_action_index: Combined index of the action
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
