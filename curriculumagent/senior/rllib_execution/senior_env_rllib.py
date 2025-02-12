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
from curriculumagent.common.utilities import find_best_line_to_reconnect, split_and_execute_action, revert_topo, \
    set_bus_from_topo_vect
from curriculumagent.senior.rllib_execution.alternative_rewards import PPO_Reward


class RLlibEnvConfig(TypedDict):
    """
    A TypedDict class representing the configuration parameters for an RLlib environment in the Grid2Op framework.

    This configuration is used to initialize and configure the environment, including options for action spaces,
    scaling, and rewards.

    Attributes:
        action_space_path (Union[Path, List[Path]]): Either a path to the action space file or a list containing
            multiple action space files.
        env_path (Union[str, Path]): Path to the Grid2Op environment.
        action_threshold (float): A threshold value between 0 and 1 that defines the cutoff for selecting actions.
        filtered_obs (Union[bool, List]): If True, the `obs.to_vect` is filtered based on predefined values.
            If False, all values are considered. Alternatively, a custom subset of values can be provided.
        scaler (Optional[Union[ObjectRef, BaseEstimator, str]]): An optional scaler for normalizing observations.
            This can be a Sklearn scaler, a ray object reference if saved using `ray.put()`, or a string ID.
        topo (Optional[bool]): Whether to revert the topology in the environment or not.
        alternative_rew (Optional[grid2op.Reward.BaseReward]): An optional alternative reward function for the environment.
        env_kwargs (Optional[dict]): Additional arguments to be passed to the Grid2Op environment.
        seed (Optional[int]): Seed value for reproducibility. If provided, ensures the same environment conditions
            are reproducible.
    """
    action_space_path: Union[Path, List[Path]]
    env_path: Union[str, Path]
    action_threshold: float
    filtered_obs: Union[bool, List]
    scaler: Optional[Union[ObjectRef, BaseEstimator, str]]
    topo: Optional[bool]
    alternative_rew: Optional[grid2op.Reward.BaseReward]
    env_kwargs: Optional[dict] = {}
    seed: Optional[int]


class SeniorEnvRllib(gym.Env):
    """
    A custom environment class that combines the Grid2Op backend with the actions of the tutor agent for use
    in the RLlib framework.

    This environment interacts with the Grid2Op environment, making it suitable for RLlib-based training.
    Unlike the `GymEnv` provided by `grid2op.gym_compat`, this environment maintains direct access to the Grid2Op
    environment, offering more flexibility in interaction.

    Note:
        With the update to Ray>=2.4, the environment now uses the `gymnasium` package instead of `gym`. This change
        requires small modifications to the environment's structure and behavior.

    """

    def __init__(self, config: RLlibEnvConfig,
                 testing: bool = False):
        """
        Initializes the Senior environment with the Grid2Op backend and RLlib-compatible settings.

        This method sets up the environment using a configuration dictionary and, if possible, uses the
        `lightsim2grid` backend. The environment is structured based on the Gym format.

        Args:
            config (RLlibEnvConfig): The configuration dictionary containing the following keys and values:
                action_space_path (Union[Path, List[Path]]): Path to the action space file or a list of action files.
                env_path (Union[str, Path]): Path to the Grid2Op environment.
                action_threshold (float): A threshold value between 0 and 1 for selecting actions.
                subset (Union[bool, List]): If True, filters the `obs.to_vect` based on predefined values. If False,
                    all values are considered. Alternatively, custom values can be provided.
                testing (bool): Indicates whether the Grid2Op environment should be started in testing mode.
                scaler (Optional[Union[ObjectRef, BaseEstimator, str]]): Optional scaler for normalizing observations.
                alternative_rew (Optional[grid2op.Reward.BaseReward]): An alternative reward function for the environment.
                env_kwargs (Optional[dict]): Additional keyword arguments for configuring the Grid2Op environment.
            testing (bool, optional): If True, initializes the environment in testing mode. Defaults to False.

        Returns:
            None.
        """
        seed = config.get("seed",np.random.choice(10000))
        logging.info(f"Setting seed {seed}")
        self.seed = seed

        np.random.seed(seed)


        # Check if subsample of Obs:
        action_space_path = config["action_space_path"]
        action_threshold = config.get("action_threshold",1.0)

        self.subset = config.get("subset",False)

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
                reward = config.get("alternative_rew",PPO_Reward)
                env = grid2op.make(dataset=data_path, backend=backend,
                                       reward_class=reward, test=False, **env_kwargs)
                logging.info("Init of Grid2Op Env works.")
            self.single_env = env.copy()
        except UnknownEnv as test_env:
            logging.info(f"Error testing was raised {test_env}. Assume testing, thus set test flag.")
            self.single_env = grid2op.make(dataset=data_path, test=True,
                                           backend=LightSimBackend(), reward_class=PPO_Reward)

        self.single_env.seed(self.seed)
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

        # Should we revert the topology ?
        self.topo = config.get("topo", False)

        # Check whether the actions are grid2op.Action.Baseactions in vector format or whether they are
        # actually topologies:
        topology_actions = not len(self.single_env.action_space().to_vect()) == self.actions.shape[1]
        self.topology_actions = config.get("topology_actions", topology_actions)

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
        self.truncated, self.terminated = self.check_terminated_truncated()

    def run_next_action(self) -> np.ndarray:
        """
        Runs the environment until an action by the agent is required.

        In this function, the Grid2Op environment runs without agent intervention until the environment requires
        an action. The cumulative reward is calculated during this process, and the observation is returned when
        the agent is needed to take action.

        Returns:
            np.ndarray: The observation vector after the environment requires an action from the agent.
        """
        # init values
        self.passiv_reward = 0
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
                    continue_running = False

            else:
                continue_running = False
        self.obs_grid2op = cur_obs
        self.passiv_reward = cumulated_rew

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
        """
        Executes the agent's action in the environment and computes the reward.

        This method performs the action specified by the agent and accumulates rewards for passive actions taken
        by the environment. It checks if multiple steps are required and handles the transition based on the
        Grid2Op environment's conditions. The action can be executed sequentially if necessary.

        Args:
            action (np.ndarray): The action to execute, provided by the agent.

        Returns:
            Tuple[np.ndarray, Union[float, int], bool, bool, dict]: A tuple containing the following:
                - Observation vector as np.ndarray.
                - The accumulated reward as either float or int.
                - A boolean indicating if the episode is terminated.
                - A boolean indicating if the episode is truncated.
                - A dictionary containing additional environment info.
        """
        # For some reason sometime the environment is done but ray does not reinit it
        if self.single_env.done:
            logging.info("The Grid2Op Environment seems to be not initialized. We just return the last"
                         "observation and do not run anything")
            self.done = True
            self.truncated, self.terminated = self.check_terminated_truncated()

            return self.obs_rl, 0.0, self.terminated, self.truncated, self.info

        # Here we have the topology grid:
        if self.topology_actions:
            topo_vect = self.single_env.get_obs().topo_vect
            new_topology = self.actions[action]
            transf_act = set_bus_from_topo_vect(topo_vect, new_topology, self.single_env.action_space).to_vect()

        else:
            transf_act = self.actions[action]

        cur_obs, reward, self.done, self.info = split_and_execute_action(env=self.single_env,
                                                                         action_vect=transf_act)

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

        # Now check for Truncation
        self.truncated, self.terminated = self.check_terminated_truncated()

        # We add the passive reward to the overall reward and scale it !
        game_over = 0
        if self.truncated:
            game_over = -300
        if self.terminated:
            game_over = 500

        self.reward = reward

        return self.obs_rl, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        This method resets the environment, optionally using a specific seed or resetting to a specific chronic
        ID provided in the options. It ensures the environment is ready for the agent's intervention.

        Args:
            seed (Optional[int]): An optional seed value for resetting the environment.
            options (Optional[Dict[str, int]]): Optional dictionary to reset to a specific chronic ID, e.g.,
                {"id": 42}.

        Returns:
            Tuple[np.ndarray, dict]: A tuple containing the following:
                - The observation vector after resetting the environment.
                - A dictionary with additional information about the environment.
        """
        if seed:
            self.single_env.seed(seed)
        if isinstance(options, dict):
            id = options.get("id", None)
            if id:
                self.single_env.set_id(id)

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

        self.truncated, self.terminated = self.check_terminated_truncated()

        return self.obs_rl, self.info

    def render(self, mode='human') -> None:
        """
        Renders the environment (required by Gym).

        This method is a placeholder for rendering the environment, which is required by the Gym interface.
        No changes are made by this method.

        Args:
            mode (str): A string representing the rendering mode. Defaults to 'human'.

        Returns:
            None.
        """

    def check_terminated_truncated(self) -> Tuple[bool, bool]:
        """
        Checks whether the environment is terminated or truncated.

        This method checks if the current episode is terminated or truncated based on the environment's status.
        Termination happens when the maximum episode duration is reached, while truncation occurs when the environment
        is finished early.

        Returns:
            Tuple[bool, bool]: A tuple containing two booleans:
                - The first boolean indicates if the episode is truncated.
                - The second boolean indicates if the episode is terminated.
        """

        if self.done:
            trunc, term = True, False
            # Check for termination (and revert truncation)
            if self.single_env.nb_time_step == self.single_env.chronics_handler.max_episode_duration():
                trunc, term = False, True
        else:
            trunc, term = False, False

        return trunc, term
