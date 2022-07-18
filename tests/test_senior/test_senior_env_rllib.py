import os.path
import pickle
from pathlib import Path

import tensorflow as tf
import numpy as np
import pytest
from grid2op.Environment import Environment
from grid2op.Exceptions import Grid2OpException
from gym.spaces import Discrete, Box
from sklearn.preprocessing import MinMaxScaler

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib


class TestEnvRllib():
    """
    Test Suite for the Rllib Env
    """

    def test_init_errors(self, test_paths_env, test_temp_save):
        """
        Testing whether the Errors are corretly raised.
        """
        test_env_path, _ = test_paths_env
        # Input completely wrong input!
        config = {"action_space_path": "Wrong Input",
                  "env_path": "Wrong data path",
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        with pytest.raises(Grid2OpException):
            SeniorEnvRllib(config)

        config = {"action_space_path": "Wrong Input",
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        with pytest.raises(ValueError):
            SeniorEnvRllib(config)

        # Input file with no actions.npy in it
        config = {"action_space_path": test_env_path,
                  "env_path": test_env_path,
                  "filtered_obs": False}
        with pytest.raises(KeyError):
            SeniorEnvRllib(config)

        # Input file with no actions.npy in it
        config = {"action_space_path": test_temp_save,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        with pytest.raises(FileNotFoundError):
            SeniorEnvRllib(config)

    def test_init_check_import(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """
        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.8,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config,
                             testing=True)

        assert isinstance(env, SeniorEnvRllib)
        assert isinstance(env.single_env, Environment)
        assert env.action_threshold == 0.8

    def test_init_check_action_space(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)

        assert isinstance(env.action_space, Discrete)
        assert env.action_space == Discrete(208)

    def test_init_check_action_tuple_tripple(self, test_paths_env, test_action_paths):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """
        single_path, tuple_path, tripple_path = test_action_paths
        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": [single_path, tuple_path, tripple_path],
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)

        assert env.action_space == Discrete(30)

    def test_init_check_actions_import(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)
        assert env.actions.shape == (208, 494)

        test_action_file = Path(test_action_path) / "actions62.npy"
        config = {"action_space_path": test_action_file,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)
        assert env.actions.shape == (62, 494)

    def test_init_check_observation_space(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)

        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (1429,)

    def test_scaler(self, test_paths_env):
        """
        Testing whether the standard scaler works. For this, we have to use the filtered observations !
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, test_action_path = test_paths_env

        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": True}
        env = SeniorEnvRllib(config, testing=True)
        env.reset(id=1)
        max_obs = max(env.obs)

        scaler_path = Path(__file__).parent.parent / "data" / "scaler_junior.pkl"
        with open(scaler_path, "rb") as fp:  # Pickling
            scaler = pickle.load(fp)

        config["scaler"] = scaler
        env = SeniorEnvRllib(config, testing=True)
        env.reset(id=1)

        assert isinstance(env.scaler, MinMaxScaler)
        assert env.obs.shape == (1221,)
        max_obs_scaled = max(env.obs)
        assert max_obs>max_obs_scaled


    def test_run_next_action(self, test_paths_env):
        """
        Testing whether the run next action works
        """
        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)

        assert env.obs_grid2op.rho.max() >= 0.9

        # Checking whether or not the to_vect() was executed:
        assert env.obs.shape == (1429,)
        assert isinstance(env.obs, np.ndarray)


    def test_step_action_format(self, test_paths_env):
        "Testing whether the output of the step action is in correct format."
        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)
        obs, reward, done, info = env.step(2)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


    def test_reset(self, test_paths_env):
        "Testing whether the output of the step action is in correct format."
        test_env_path, test_action_path = test_paths_env
        config = {"action_space_path": test_action_path,
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)
        env.obs = np.zeros(1362)
        assert np.array_equal(env.obs, np.zeros(1362))
        env.reset()
        assert not np.array_equal(env.obs, np.zeros(1362))


    #########################################################################
    ########## Testing Tuple and Tripple Actions ############################

    def test_step_action_tuple_tripple(self, test_paths_env, test_action_paths):
        """
        Testing, whether tuple and tripple actions can be executed.
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {"action_space_path": [single_path, tuple_path, tripple_path],
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}
        env = SeniorEnvRllib(config, testing=True)

        # Note, the actions consist of 10 single, 10 tuple and 10 tripple actions
        # Execute a single action:

        obs, reward, done, info = env.step(2)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert info['is_illegal'] == False

        # Execute a single action:
        env.reset()
        step = env.single_env.nb_time_step
        obs, reward, done, info = env.step(12)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert info['is_illegal'] == False
        # Should be true
        assert step + 1 < env.single_env.nb_time_step

        # Execute a single action:
        env.reset()
        obs, reward, done, info = env.step(22)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert info['is_illegal'] == False

        # Should be true
        assert step + 2 < env.single_env.nb_time_step


    ########## Test filtered_obs ##############
    def test_filtered_obs(self, test_paths_env, test_action_paths):
        """
        Testing the filtered_obs
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {"action_space_path": [single_path, tuple_path, tripple_path],
                  "env_path": test_env_path,
                  "action_threshold": 0.9,
                  "filtered_obs": False}

        # Original
        env = SeniorEnvRllib(config,
                             testing=True)
        assert env.observation_space.shape == (1429,)
        obs, _, _, _ = env.step(1)
        assert obs.shape == (1429,)
        obs = env.run_next_action()
        assert obs.shape == (1429,)
        # Filtered
        config["filtered_obs"] = True
        env = SeniorEnvRllib(config,
                             testing=True)
        assert env.observation_space.shape == (1221,)
        obs, _, _, _ = env.step(1)
        assert obs.shape == (1221,)
        obs = env.run_next_action()
        assert obs.shape == (1221,)

        # Different Selection
        config["filtered_obs"] = list(range(1, 100))
        env = SeniorEnvRllib(config,
                             testing=True)
        assert env.observation_space.shape == (99,)
        obs, _, _, _ = env.step(1)
        assert obs.shape == (99,)
        obs = env.run_next_action()
        assert obs.shape == (99,)
