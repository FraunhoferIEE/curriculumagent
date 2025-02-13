import os
import pickle
import shutil
from pathlib import Path

import grid2op.Observation
import numpy as np
import pytest
import tensorflow as tf
from grid2op.Environment import Environment
from grid2op.Exceptions import Grid2OpException
from gymnasium.spaces import Discrete, Box
from sklearn.preprocessing import MinMaxScaler

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib


class TestEnvRllib:
    """
    Test Suite for the Rllib Env
    """

    def test_init_errors(self, test_paths_env, test_temp_save):
        """
        Testing whether the Errors are corretly raised.
        """
        # Clear directory:
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)

        test_env_path, _ = test_paths_env
        # Input completely wrong input!
        config = {
            "action_space_path": "Wrong Input",
            "env_path": "Wrong data path",
            "action_threshold": 0.9,
            "subset": False,
        }
        with pytest.raises(Grid2OpException):
            SeniorEnvRllib(config)

        config = {
            "action_space_path": "Wrong Input",
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
        }
        with pytest.raises(ValueError):
            SeniorEnvRllib(config)

        # Input file with no actions.npy in it
        config = {"action_space_path": test_env_path, "env_path": test_env_path, "subset": False}
        with pytest.raises(FileNotFoundError):
            SeniorEnvRllib(config)


    def test_init_check_import(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """
        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.8,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        assert isinstance(env, SeniorEnvRllib)
        assert isinstance(env.single_env, Environment)
        assert env.action_threshold == 0.8

    def test_init_check_action_space(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": [Path(test_action_path) / "actions62.npy", Path(test_action_path) / "actions146.npy"],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        assert isinstance(env.action_space, Discrete)
        assert env.action_space == Discrete(208)

    def test_init_check_action_tuple_tripple(self, test_paths_env, test_action_paths):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """
        single_path, tuple_path, tripple_path = test_action_paths
        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        assert env.action_space == Discrete(30)

    def test_init_check_actions_import(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        print(test_action_path)
        print(test_env_path)
        config = {
            "action_space_path": [Path(test_action_path) / "actions62.npy", Path(test_action_path) / "actions146.npy"],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)
        assert env.actions.shape == (208, 494)

        test_action_file = Path(test_action_path) / "actions62.npy"
        config = {
            "action_space_path": test_action_file,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)
        assert env.actions.shape == (62, 494)

    def test_env_kwargs(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False,
            "env_kwargs": {"opponent_attack_cooldown": 42}
        }
        env = SeniorEnvRllib(config, testing=True)
        assert env.single_env._opponent_attack_cooldown == 42

    def test_init_check_observation_space(self, test_paths_env):
        """
        Testing whether the init of the env works and returns the correct action space and observation space
        """

        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (1429,)

    def test_scaler(self, test_paths_env, test_submission_action_space):
        """
        Testing whether the standard scaler works. For this, we have to use the filtered observations !
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env

        config = {
            "action_space_path": test_submission_action_space,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)
        env.reset(options={"id": 1})
        max_obs = max(env.obs_rl)

        scaler_path = Path(__file__).parent.parent / "data" / "scaler_junior.pkl"
        with open(scaler_path, "rb") as fp:  # Pickling
            scaler = pickle.load(fp)

        config["scaler"] = scaler
        env = SeniorEnvRllib(config, testing=True)
        env.reset(options={"id": 1})

        assert isinstance(env.scaler, MinMaxScaler)
        assert env.obs_rl.shape == (1429,)
        max_obs_scaled = max(env.obs_rl)
        assert max_obs > max_obs_scaled

        # Now run wiht path:
        config["scaler"] = scaler_path
        env = SeniorEnvRllib(config, testing=True)
        env.reset(options={"id": 1})

        assert isinstance(env.scaler, MinMaxScaler)
        assert env.obs_rl.shape == (1429,)
        max_obs_scaled = max(env.obs_rl)
        assert max_obs > max_obs_scaled

    def test_run_next_action(self, test_paths_env):
        """
        Testing whether the run next action works
        """
        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        assert env.obs_grid2op.rho.max() >= 0.9

        # Checking whether the to_vect() was executed:
        assert env.obs_rl.shape == (1429,)
        assert isinstance(env.obs_rl, np.ndarray)

    def test_step_action_format(self, test_paths_env):
        "Testing whether the output of the step action is in correct format."
        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)
        obs, reward, terminated, truncated, info = env.step(2)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset(self, test_paths_env):
        "Testing whether the output of the step action is in correct format."
        test_env_path, test_action_path = test_paths_env
        config = {
            "action_space_path": test_action_path,
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)
        env.obs_rl = np.zeros(1362)
        assert np.array_equal(env.obs_rl, np.zeros(1362))
        env.reset()
        assert not np.array_equal(env.obs_rl, np.zeros(1362))

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

        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }
        env = SeniorEnvRllib(config, testing=True)

        # Note, the actions consist of 10 single, 10 tuple and 10 tripple actions
        # Execute a single action:

        obs, reward, terminated, truncated, info = env.step(2)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert info["is_illegal"] == False
        assert truncated is False
        assert terminated is False

        # Execute a single action:
        env.reset()
        step = env.single_env.nb_time_step
        obs, reward, terminated, truncated, info = env.step(12)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert info["is_illegal"] == False
        # Should be true
        assert step + 1 < env.single_env.nb_time_step

        # Execute a single action:
        env.reset()
        obs, reward, terminated, truncated, info = env.step(22)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert info["is_illegal"] == False

        # Should be true
        assert step + 2 < env.single_env.nb_time_step

    ########## Test subset ##############
    def test_filtered_obs(self, test_paths_env, test_action_paths):
        """
        Testing the subset
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }

        # Original
        env = SeniorEnvRllib(config, testing=True)
        assert env.observation_space.shape == (1429,)
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (1429,)
        obs = env.run_next_action()
        assert obs.shape == (1429,)
        # Filtered
        config["subset"] = True
        env = SeniorEnvRllib(config, testing=True)
        assert env.observation_space.shape == (1221,)
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (1221,)
        obs = env.run_next_action()
        assert obs.shape == (1221,)

        # Different Selection
        config["subset"] = list(range(1, 100))
        env = SeniorEnvRllib(config, testing=True)
        assert env.observation_space.shape == (99,)
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (99,)
        obs = env.run_next_action()
        assert obs.shape == (99,)

    def test_topo_env(self, test_paths_env, test_action_paths):
        """
        Testing the subset
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }

        env = SeniorEnvRllib(config, testing=True)
        assert env.topo == False
        env.single_env.seed(10)
        env.single_env.reset()
        env.reset()

        # Run multiple steps:
        env.step(15)
        assert env.single_env.nb_time_step >= 80
        steps_survived = env.single_env.nb_time_step

        obs_g2o: grid2op.Observation.BaseObservation = env.single_env.get_obs()
        assert sum(obs_g2o.topo_vect == 2) > 0
        number_of_sub_in_2 = sum(obs_g2o.topo_vect == 2)

        # Now lets run the topo network
        np.random.seed(42)
        tf.random.set_seed(42)
        config["topo"] = True
        env2 = SeniorEnvRllib(config, testing=True)
        assert env2.topo

        env2.single_env.seed(10)
        env2.single_env.reset()
        env2.reset()

        env.step(15)
        steps_survived_topo = env2.single_env.nb_time_step
        obs_g2o2: grid2op.Observation.BaseObservation = env2.single_env.get_obs()
        number_of_sub_in_2_topo = sum(obs_g2o2.topo_vect == 2)

        # Look wether a different result is present
        assert number_of_sub_in_2 > number_of_sub_in_2_topo

    def test_env_with_rllib_checker(self, test_paths_env, test_action_paths):
        """
        Testing the environment with rllib checker. This is quite usefull, given that with failure,
        it will also fail when running on cluster.
        """
        from ray.rllib.utils import check_env

        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }

        env = SeniorEnvRllib(config, testing=True)

        check_env(env)

    def test_truncated_and_terminated(self, test_paths_env, test_action_paths):
        """
        Testing if truncated or terminated returns something
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        test_env_path, _ = test_paths_env
        single_path, tuple_path, tripple_path = test_action_paths

        config = {
            "action_space_path": [single_path, tuple_path, tripple_path],
            "env_path": test_env_path,
            "action_threshold": 0.9,
            "subset": False,
            "topo": False
        }

        env = SeniorEnvRllib(config, testing=True)

        env.single_env.done = True

        obs, rew, term, trunc, info = env.step(15)
        assert rew == 0
        assert trunc is True
        assert term is False


class ImportRLib:
    """
    Test class to ensure that rllib works
    """

    def test_import_ray(self):
        """
        Testing import ray
        """
        import ray as ray
        assert ray.__version__ == "2.5.1"

    def test_import_rllib(self):
        """
        Testing rllib
        """

        from ray.rllib.env.base_env import BaseEnv
        env = BaseEnv()
        assert env
