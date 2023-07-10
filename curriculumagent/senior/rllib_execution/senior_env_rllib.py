"""This file is build to construct the environment of the Senior Agent to run the PPO with ray and rllib.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Union, TypedDict, Tuple, List, Optional, Dict

import grid2op
import gymnasium as gym
import numpy as np
import ray
from grid2op.Exceptions import UnknownEnv
from gymnasium.spaces import Discrete, Box
from lightsim2grid import LightSimBackend
from ray._raylet import ObjectRef
from sklearn.base import BaseEstimator

from curriculumagent.common.obs_converter import obs_to_vect
from curriculumagent.common.utilities import find_best_line_to_reconnect, split_and_execute_action, revert_topo
from curriculumagent.senior.rllib_execution.alternative_rewards import PPO_Reward


class RLlibEnvConfig(TypedDict):
    """TypeDict Class.

     Attributes:
        action_space_path: Either path to the actions or a list containing mutliple actions.
        env_path: Path of the Grid2Op environment
        action_threshold: Between 0 and 1.
        subset: Should the obs.to_vect be filtered similar to the original Agent. If True,
                the obs.to_vect is filtered based on predefined values. If False, all values are considered.
                Alternatively, one can submit the once own values.
                testing: Indicator, whether the underlying Grid2op Env should be started in testing mode or not
        scaler: Optional Scaler of Sklearn Model. Either a Sklearn Scaler or its ray ID, if the scaler is saved via
                ray.put(). If scaler is provided, the environment will scale the observations based on
                scaler.transform().
        env_kwargs: Optional additional arguments for the Grid2Op environment that should be used when making the
        environment.

    """
    action_space_path: Union[Path, List[Path]]
    env_path: Union[str, Path]
    action_threshold: float
    filtered_obs: Union[bool, List]
    scaler: Optional[Union[ObjectRef, BaseEstimator, str]]
    topo: Optional[bool]
    alternative_rew: Optional[grid2op.Reward.BaseReward]
    env_kwargs: Optional[dict] = {}


class SeniorEnvRllib(gym.Env):
    """Environment class, in which the Grid2Op Backend is combined with the actions of the tutor agent.
    The environment can then be used in the RLlib framework for training.

    There is a difference between this environment and the GymEnv of grid2op.gym_compat, because we
    still access the Grid2Op Environment.

    Note: With the Update to Ray>=2.4 the environment is now based on the gymnasium package and
        not gym anymore. Thus, small changes are necessary.


    """

    def __init__(self, config: RLlibEnvConfig,
                 testing: bool = False):
        """Initialization of Environment with Grid2Op.

        If possible, the lightsim2grid backend is used. Further, we define based on the Gym format
        the environment.

        Args:
            config: RllibEnvConfig File, containing the following keys,value pairs:
                action_space_path: Either path to the actions or a list containing mutliple actions.
                env_path: Path of the Grid2Op environment
                action_threshold: Threshold between 0 and 1.
                subset: Should the obs.to_vect be filtered similar to the original Agent. If True,
                        the obs.to_vect is filtered based on predefined values by the obs_to_vect method.
                        If False, all values are considered. Alternatively, one can submit the own values.
                        testing: Indicator, whether the underlying Grid2op Env should be started in testing mode or not
                scaler: Optional Scaler of Sklearn Model. Either a Sklearn Scaler or its ray ID, if the scaler is saved
                        ray.put(). If scaler is provided, the environment will scale the observations based on
                        scaler.transform().
                alternative_rew: If wanted, you can supply an alternative reward for the grid2op Environment.

        Returns:
            None.

        """
        # Check if subsample of Obs:
        action_space_path = config["action_space_path"]
        action_threshold = config["action_threshold"]

        self.subset = config["subset"]

        data_path = str(config["env_path"])
        env_kwargs = config.get("env_kwargs", {})

        # Init environments: If testing is true, we test the test flag
        backend = LightSimBackend()
        try:
            if testing:
                env = grid2op.make(dataset=data_path, backend=backend,
                                   reward_class=PPO_Reward, test=True, **env_kwargs)
                logging.info("Test flag was set to true. Init Env in testing mode.")
            else:
                if "alternative_rew" in config.keys():
                    env = grid2op.make(dataset=data_path, backend=backend,
                                       reward_class=config["alternative_rew"], test=False, **env_kwargs)
                else:
                    env = grid2op.make(dataset=data_path, backend=backend,
                                       reward_class=PPO_Reward, test=False, **env_kwargs)
                logging.info("Init of Grid2Op Env works.")
            self.single_env = env.copy()
        except UnknownEnv as test_env:
            logging.info(f"Error testing was raised {test_env}. Assume testing, thus set test flag.")
            self.single_env = grid2op.make(dataset=data_path, test=True,
                                           backend=LightSimBackend(), reward_class=PPO_Reward)

        self.single_env.chronics_handler.shuffle(
            shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
        # Create multiple environments:
        self.obs_grid2op = self.single_env.reset()

        # Initialize the actions:
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logging.info(f"Action_space_path {action_space_path} is a file and will be loaded.")
                self.actions = np.load(str(action_space_path))
            elif action_space_path.is_dir():
                logging.info(f"Action_space_path {action_space_path} is a path. All available action files "
                             f" will be loaded.")
                all_action_files = [act for act in os.listdir(action_space_path) if "actions" in act and ".npy" in act]

                if not all_action_files:
                    raise FileNotFoundError("No actions files were found!")

                loaded_files = [np.load(str(action_space_path / act)) for act in all_action_files]
                self.actions = np.concatenate(loaded_files, axis=0)

        elif isinstance(action_space_path, list):
            logging.info(f"Action_space_path {action_space_path} is a list containing multiple actions.")
            loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
            self.actions = np.concatenate(loaded_files, axis=0)
        else:
            raise ValueError(f"The action_space_path variable {action_space_path} does neither consist of a single "
                             f"action nor of a path where actions can be found.")

        # Define Action and Observation Space in Gym Format:
        self.action_space = Discrete(len(self.actions))

        if isinstance(self.subset, list):
            vect_shape = len(self.subset)
        elif self.subset:
            vect_shape = obs_to_vect(self.single_env.get_obs(), False).shape[0]
        else:
            vect_shape = self.single_env.observation_space.size()
        self.observation_space = Box(shape=(vect_shape,), high=np.inf, low=-np.inf)

        logging.info(f"Initialize Environment with the following action space{self.action_space}  and the following "
                     f"observations space{self.observation_space}")

        # check scaler:
        self.scaler = config.get("scaler", None)
        if self.scaler is not None:
            if isinstance(self.scaler, (str, Path)):
                with open(self.scaler, "rb") as fp:
                    self.scaler = pickle.load(fp)

        self.topo = config.get("topo", False)

        # Initialize further arguments:
        assert 0.0 < action_threshold <= 1.0
        self.action_threshold = action_threshold

        self.step_in_env = 0
        self.passiv_reward = 0
        self.obs_rl = self.run_next_action()
        self.reward = None

        self.info: Dict = {}
        self.done = self.single_env.done
        # Added for gymnasium
        self.truncated, self.terminated = self.__check_terminated_truncated()

    def run_next_action(self) -> np.ndarray:
        """Run the environment until an action by the agent is required.

        In this function, the Grid2Op Environment will run until it requires help
        from the agent. Then it will return the observation and wait for action.

        Returns:
            Observation vector.

        """
        # init values
        continue_running = True
        cur_obs = self.single_env.get_obs()
        cumulated_rew = 0
        # Run through the env and do nothing.
        # The reward is collected until the agent is done.
        # If done, there is an extra reward.
        while continue_running:
            if cur_obs.rho.max() < self.action_threshold:
                if self.topo:
                    action_array = revert_topo(self.single_env.action_space, cur_obs)
                    default_action = self.single_env.action_space.from_vect(action_array)
                else:
                    default_action = self.single_env.action_space({})

                action = find_best_line_to_reconnect(obs=cur_obs,
                                                     original_action=default_action)
                # Execute in env:
                cur_obs, reward, _, self.info = self.single_env.step(action)
                cumulated_rew += reward
                self.step_in_env += 1

                if self.single_env.done:
                    if 'GAME OVER' in str(self.info['exception']):
                        cumulated_rew += -300
                    else:
                        cumulated_rew += 500
                    continue_running = False

            else:
                continue_running = False
        self.obs_grid2op = cur_obs
        self.passiv_reward += cumulated_rew

        # Select subset if wanted
        if isinstance(self.subset, list):
            obs_rl = self.obs_grid2op.to_vect()[self.subset]
        elif self.subset:
            obs_rl = obs_to_vect(self.obs_grid2op, False)
        else:
            obs_rl = self.obs_grid2op.to_vect()

        # Select scaler if wanted
        if self.scaler:
            if isinstance(self.scaler, BaseEstimator):
                obs_rl = self.scaler.transform(obs_rl.reshape(1, -1)).reshape(-1, )
            elif isinstance(self.scaler, ObjectRef):
                obs_rl = ray.get(self.scaler).transform(obs_rl.reshape(1, -1)).reshape(-1, )

        return obs_rl

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Union[float, int], bool, bool, dict]:
        """Conduct the action of the agent.

        This method does not only run the action of the agent but additionally accumulates the
        reward for the do-nothing agent AFTER the action of the agent. Thus, if the action "saves" a
        lot of steps, the agent receives a handsome reward.

        Note: This method checks, whether the action is a tuple or a triple action. If the action
        requires multiple steps indeed, then the action will be executed sequentially, i.e., 2 or 3 steps
        in the Grid2Op Environment.

        With the update to gymnasium the output is a little bit different

        Args:
            action: ID of the action within the action set.

        Returns:
            The observation as np.ndarray, the reward, the info if it is terminated, the boolean if truncated
            and the info, which is default.

        """
        # For some reason sometime the environment is done but ray does not reinit it
        if self.single_env.done:
            logging.info("The Grid2Op Environment seems to be not initialized. We just return the last"
                         "observation and do not run anything")
            self.done = True
            self.truncated, self.terminated = self.__check_terminated_truncated()

            return self.obs_rl, 0.0, self.terminated, self.truncated, self.info

        cur_obs, reward, self.done, self.info = split_and_execute_action(env=self.single_env,
                                                                         action_vect=self.actions[action])

        # get action:
        self.step_in_env += 1
        self.obs_grid2op = cur_obs

        # Select subset if wanted
        if isinstance(self.subset, list):
            obs_rl = self.obs_grid2op.to_vect()[self.subset]
        elif self.subset:
            obs_rl = obs_to_vect(self.obs_grid2op, False)
        else:
            obs_rl = self.obs_grid2op.to_vect()

        if self.scaler:
            if isinstance(self.scaler, BaseEstimator):
                obs_rl = self.scaler.transform(obs_rl.reshape(1, -1)).reshape(-1, )
            elif isinstance(self.scaler, ObjectRef):
                obs_rl = ray.get(self.scaler).transform(obs_rl.reshape(1, -1)).reshape(-1, )

        self.obs_rl = obs_rl

        if (cur_obs.rho.max() < self.action_threshold) and self.done is False:
            self.obs_rl = self.run_next_action()
            reward += self.passiv_reward

        self.done = self.single_env.done

        # We add the passive reward to the overall reward
        self.reward = reward

        # Now check for Truncation
        self.truncated, self.terminated = self.__check_terminated_truncated()

        return self.obs_rl, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, dict]:
        """ Resetting the environment.

        Args:
            seed: Optional seed that gets passed to the environment.
            options: Optional id for the Grid2Op Environment. When you want to reset to a specific chronic
            reset the env with a dictionary and the key "id". As example {"id": 42}.

        Returns:
            Tuple with the observation and the info statement

        """
        try:
            if seed:
                self.single_env.seed(seed)
            if isinstance(options, dict):
                if "id" in options.keys():
                    self.single_env.set_id(options)
        except Exception as e:
            logging.warning(f"The ID did not work with the environment and raised {e}. Reset without setting ID")

        # Reset environment and run the Grid2Op Env
        self.done = True
        while self.done:
            # We have to iterate over the environment, because it can happen that the environment
            # can be finished without the intervention of the agent. By this loop we ensure that the
            # Senior is needed.
            self.obs_grid2op = self.single_env.reset()
            self.step_in_env = 0
            self.passiv_reward = 0
            self.obs_rl = self.run_next_action()
            if not self.single_env.done:
                self.done = False

        self.truncated, self.terminated = self.__check_terminated_truncated()

        return self.obs_rl, self.info

    def render(self, mode='human') -> None:
        """Required for GYM Environment. However, we do not change anything.

        Args:
            mode: String.

        Returns:
             None.

        """

    def __check_terminated_truncated(self) -> Tuple[bool, bool]:
        """ Check the env for terminated and truncated

        Returns:

        """

        if self.done:
            trunc, term = True, False
            # Check for termination (and revert truncation)
            if self.single_env.nb_time_step == self.single_env.chronics_handler.max_episode_duration():
                trunc, term = False, True
        else:
            trunc, term = False, False

        return trunc, term
