"""
This file is build to construct the environment of the Senior Agent to run the PPO with ray and rllib.
"""
import os
from pathlib import Path
import logging
from typing import Union, TypedDict, Tuple, List, Optional
import gym
import ray
from grid2op.Exceptions import UnknownEnv
from gym.spaces import Discrete, Box
from lightsim2grid import LightSimBackend
import numpy as np
import grid2op
from ray._raylet import ObjectRef
from sklearn.base import BaseEstimator

from curriculumagent.senior.openai_execution.ppo_reward import PPO_Reward
from curriculumagent.common.utilities import find_best_line_to_reconnect, split_and_execute_action


class RLlibEnvConfig(TypedDict):
    """
    TypeDict Class with following inputs:
    action_space_path: Either path to the actions or a list containing mutliple actions.
    env_path: path of the Grid2Op environment
    action_threshold: between 0 and 1
    filtered_obs: Should the obs.to_vect be filtered similar to the original Agent. If True,
            the obs.to_vect is filted based on predefined values. If False, all values are considered.
            Alternatively, one can submit the once own values.
            testing: Indicator, whether the underlying Grid2op Env should be started in testing mode or not
    scaler: Optional Scaler of Sklearn Model. Either a Sklearn Scaler or its ray ID, if the scaler is saved via
            ray.put(). If sclaer is provided, the environment will scale the observations based on scaler.transform()
    """
    action_space_path: Union[Path, List[Path]]
    env_path: Union[str, Path]
    action_threshold: float
    filtered_obs: Union[bool, List]
    scaler: Optional[Union[ObjectRef,  BaseEstimator]]


class SeniorEnvRllib(gym.Env):
    """
    Environment class, in which the Grid2op Backend is combined with the actions of the tutor agent.
    The environment can then be used in the RLlib framework for training.

    There is a difference between this environment and the GymEnv of grid2op.gym_compat, because we
    still access the Grid2Op Environment.

    """

    def __init__(self, config: RLlibEnvConfig,
                 testing: bool = False):
        """ Initialization of Environment with Grid2Op

        If possible, the lightsim2grid backend is used. Further, we define based on the Gym format
        the environment



        Args:
            config: RllibEnvConfig File, containing the following keys,value pairs:
                action_space_path: Either path to the actions or a list containing mutliple actions.
                env_path: path of the Grid2Op environment
                action_threshold: between 0 and 1
                filtered_obs: Should the obs.to_vect be filtered similar to the original Agent. If True,
                        the obs.to_vect is filtered based on predefined values. If False, all values are considered.
                        Alternatively, one can submit the once own values.
                        testing: Indicator, whether the underlying Grid2op Env should be started in testing mode or not
                scaler: Optional Scaler of Sklearn Model. Either a Sklearn Scaler or its ray ID, if the scaler is saved via
                        ray.put(). If scaler is provided, the environment will scale the observations based on
                        scaler.transform()


        """
        # Check if subsample of Obs:
        filtered_obs = config["filtered_obs"]
        action_space_path = config["action_space_path"]
        action_threshold = config["action_threshold"]

        if isinstance(filtered_obs, list):
            self.chosen = filtered_obs
        else:
            if filtered_obs:
                self.chosen = self.__get_default_chosen()
            else:
                self.chosen = []

        data_path = str(config["env_path"])

        # Init environments: If testing is true, we test the test flag
        backend = LightSimBackend()
        try:
            if testing:
                env = grid2op.make(dataset=data_path, backend=backend,
                                   reward_class=PPO_Reward, test=True)
                logging.info("Test flag was set to true. Init Env in testing mode.")
            else:
                env = grid2op.make(dataset=data_path, backend=backend,
                                   reward_class=PPO_Reward, test=False)
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

        if self.chosen:
            self.observation_space = Box(shape=(len(self.chosen),), high=np.inf, low=-np.inf)
        else:
            self.observation_space = Box(shape=(self.single_env.observation_space.size(),), high=np.inf, low=-np.inf)

        logging.info(f"Initialize Environment with the following action space{self.action_space}  and the following "
                     f"observations space{self.observation_space}")

        # check scaler:
        if "scaler" in config.keys():
            self.scaler = config["scaler"]
        else:
            self.scaler = None

        # Initialize further arguments:
        assert 0.0 < action_threshold <= 1.0
        self.action_threshold = action_threshold

        self.step_in_env = 0
        self.passiv_reward = 0
        self.obs = self.run_next_action()
        self.reward = None

        self.info = {}
        self.done = None

    def run_next_action(self) -> np.ndarray:
        """ Run the environment until an action by the agent is required

        In this function, the grid2op env will run until it requires help
        from the agent. Then it will return the observation and wait for action.

        Returns: observation

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
                do_nothing = self.single_env.action_space({})
                action = find_best_line_to_reconnect(obs=cur_obs,
                                                     original_action=do_nothing)
                # Execute in env:
                cur_obs, reward, _, info = self.single_env.step(action)
                cumulated_rew += reward
                self.step_in_env += 1

                if self.single_env.done:
                    if 'GAME OVER' in str(info['exception']):
                        cumulated_rew += -300
                    else:
                        cumulated_rew += 500
                    continue_running = False

            else:
                continue_running = False
        self.obs_grid2op = cur_obs
        self.passiv_reward += cumulated_rew

        # Select subset if wanted
        if self.chosen:
            obs = self.obs_grid2op.to_vect()[self.chosen]
        else:
            obs = self.obs_grid2op.to_vect()

        # Select scaler if wanted
        if self.scaler:
            if isinstance(self.scaler,BaseEstimator):
                obs = self.scaler.transform(obs.reshape(1, -1)).reshape(-1, )
            elif isinstance(self.scaler,ObjectRef):
                obs = ray.get(self.scaler).transform(obs.reshape(1,-1)).reshape(-1,)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Union[float, int], bool, dict]:
        """ Conduct the action of the agent

        This method does not only run the action of the agent but additionally also accumulates the
        reward for the do nothing agent AFTER the action of the agent. Thus, if the action "saves" a
        lot of steps, the agent receives an handsome reward.

        Note: This method checks, whether the action is a tuple or tripple action. If the action
        requires indeed multiple steps, then the action will be executed sequentially, i.e., 2 or 3 steps
        in the grid2op Environment

        Args:
            action: id of the action within the action set

        Returns:

        """
        # check tuple/tripple action:
        cur_obs, reward, self.done, self.info = split_and_execute_action(env=self.single_env,
                                                                         action=self.actions[action])

        # get action:
        self.step_in_env += 1
        self.obs_grid2op = cur_obs

        if self.chosen:
            obs = cur_obs.to_vect()[self.chosen]
        else:
            obs = cur_obs.to_vect()

        if self.scaler:
            if isinstance(self.scaler,BaseEstimator):
                obs = self.scaler.transform(obs.reshape(1, -1)).reshape(-1, )
            elif isinstance(self.scaler,ObjectRef):
                obs = ray.get(self.scaler).transform(obs.reshape(1,-1)).reshape(-1,)

        self.obs = obs

        if (cur_obs.rho.max() < self.action_threshold) and self.done is False:
            self.obs = self.run_next_action()
            reward += self.passiv_reward
            self.done = self.single_env.done

        # We add the passive reward to the overall reward
        self.reward = reward

        return self.obs, self.reward, self.done, self.info

    def reset(self, id: Optional[int] = None) -> np.ndarray:
        """ Resetting the environment

        Args:
            id: Optional id for the Grid2Op Enviornment. When you want to reset to a specific chronic

        Returns:

        """
        if id:
            try:
                self.single_env.set_id(id)
            except:
                logging.warning("The ID did not work with the environment. Reset without setting ID")
        self.obs_grid2op = self.single_env.reset()
        self.step_in_env = 0
        self.passiv_reward = 0
        self.obs = self.run_next_action()
        return self.obs

    def render(self, mode='human') -> None:
        """ Required for gym environment. However, we do not change anything

        Args:
            mode: str

        Returns: None

        """

    def __get_default_chosen(self) -> List:
        """ Method to receive the default chosen values to filter the obs.to_vet

        Note: Contrary to the Tutor, we now have one value less. Thus the list starts at 1

        Returns: list of chosen values

        """
        # Assign label & features

        chosen = list(range(1, 6)) + list(range(6, 72)) + list(range(72, 183)) + list(range(183, 655))
        #       label      timestamp         generator-PQV            load-PQV                      line-PQUI
        chosen += list(range(655, 714)) + list(range(714, 773)) + list(range(773, 832)) + list(range(832, 1009))
        #               line-rho               line switch         line-overload steps          bus switch
        chosen += list(range(1009, 1068)) + list(range(1068, 1104)) + list(range(1104, 1163)) + list(range(1163, 1222))
        #          line-cool down steps   substation-cool down steps     next maintenance         maintenance duration
        return chosen


