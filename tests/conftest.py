"""
This file can be used to define extra pytest rules and fixtures
"""
import os
import pickle
import random
from pathlib import Path

import grid2op
import numpy as np
import pytest
import tensorflow as tf
from gymnasium.spaces import Box, Discrete
# We formulate two additional measures to assure that the run slow and ultra slow are triggered:
from lightsim2grid import LightSimBackend

from curriculumagent.junior.junior_student import load_dataset
from curriculumagent.senior.rllib_execution.alternative_rewards import PPO_Reward


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Tests will be slow. ")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "ultra_slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def test_paths_env():
    """Get the test path of the env, depending on if you run the code locally or all pytest through the console

    Returns:

    """
    test_env_path = Path(__file__).parent / "data" / "training_data_track1"

    test_actions_path = Path(__file__).parent/ "data" / "action_spaces" /"original"
    return test_env_path, test_actions_path

@pytest.fixture
def test_baseline_models():
    """ Returning two paths with one consisting of the junior model, the other of the senior+actions

    Returns:

    """
    junior_bl_path = Path(__file__).parent / "data" / "baseline" / "junior"
    senior_bl_path = Path(__file__).parent / "data" / "baseline" / "saved_model"

    return senior_bl_path,junior_bl_path

@pytest.fixture
def test_env(test_paths_env):
    """
    Returns a specifc grid2op Env with a specific scenario
    """
    backend = LightSimBackend()
    env_path, _ = test_paths_env
    data_path = str(env_path)
    scenario_path = str(env_path / "chronics")
    env = grid2op.make(
        dataset=data_path, chronics_path=scenario_path, backend=backend, reward_class=PPO_Reward, test=True
    )
    env.set_id(1)
    env.reset()
    return env

@pytest.fixture
def test_env_nonconverge():
    """
    Returns a "l2rpn_wcci_2022" grid2op Env with a specific scenario
    """

    backend = LightSimBackend()
    dataset = "l2rpn_wcci_2022"

    env = grid2op.make(
        dataset=dataset, backend=backend, test=True
    )
    np.random.seed(42)
    random.seed(42)
    env.set_id(19)
    return env


@pytest.fixture
def test_action_set():
    """
    Get all three types of actions. Tripple, Touple and Single action
    """
    test_env_path = Path(__file__).parent / "data" / "action_spaces"

    tripple_array = np.load(test_env_path / "test_tripple.npy")
    tuple_array = np.load(test_env_path / "test_tuple.npy")
    single_array = np.load(test_env_path / "test_single.npy")

    return single_array, tuple_array, tripple_array


@pytest.fixture
def test_action_paths():
    """
    Test paths of the actions
    """
    test_data_path = Path(__file__).parent / "data"
    single_path = test_data_path / "action_spaces" / "test_single.npy"
    tuple_path = test_data_path / "action_spaces" / "test_tuple.npy"
    tripple_path = test_data_path / "action_spaces" / "test_tripple.npy"
    return single_path, tuple_path, tripple_path

@pytest.fixture
def test_action_single():
    """
    Test paths of the actions
    """
    test_data_path = Path(__file__).parent / "data"
    single_path1 = test_data_path / "action_spaces" / "test_single01.npy"
    single_path2 = test_data_path / "action_spaces" / "test_single02.npy"
    return single_path1, single_path2

@pytest.fixture
def test_submission_action_space():
    """
    Test submission action spaces
    """
    test_data_path = Path(__file__).parent / "data"
    path1 = test_data_path / "action_spaces" / "submission" / "actionspace_nminus1.npy"
    path2 = test_data_path / "action_spaces" / "submission" / "actionspace_tuples.npy"
    return [path1, path2]


@pytest.fixture
def test_junior_input():
    """
    Junior files
    """
    tf.random.set_seed(42)
    np.random.seed(42)
    data_path = Path(__file__).parent / "data" / "junior_experience"
    s_tr, a_tr, s_v, a_v, s_te, a_te = load_dataset(dataset_path=data_path, dataset_name="test")
    return s_tr, a_tr, s_v, a_v, s_te, a_te


@pytest.fixture
def test_temp_save():
    """
    Temporary folder for saving data
    """
    data_path = Path(__file__).parent / "data" / "temporary_save"
    return data_path

@pytest.fixture
def test_path_data():
    """
    Temporary folder for saving data
    """
    data_path = Path(__file__).parent / "data"
    return data_path


@pytest.fixture
def test_submission_models():
    """
    Test paths of the original submission model
    """
    submission_dir = Path(__file__).parent / "data" / "submission"
    old_m = submission_dir / "old"
    ray_v1 = submission_dir / "ray_v1"
    ray_v24 = submission_dir / "ray_v24"
    return old_m, ray_v1, ray_v24


@pytest.fixture
def test_scaler():
    """
    Get MinMaxScaler
    """
    scaler_path = Path(__file__).parent / "data" / "scaler_junior.pkl"
    with open(scaler_path, "rb") as fp:  # Pickling
        scaler = pickle.load(fp)
    return scaler


@pytest.fixture
def test_action_possibilities(test_env):
    """
    Iterate over all action sets
    """
    actions_dict = {}
    for act in test_env.action_space.get_all_unitary_topologies_set(test_env.action_space):
        sub_effected = act.as_dict()["set_bus_vect"]["modif_subs_id"][0]

        if sub_effected in actions_dict:
            actions_dict[sub_effected].append(act)
        else:
            actions_dict[sub_effected] = [act]

    return actions_dict


@pytest.fixture
def test_sub_action(test_env):
    """
    Get one action that changes the substation
    """
    actions = test_env.action_space.get_all_unitary_topologies_set(test_env.action_space)
    act = actions[35]
    return act


@pytest.fixture
def obs_space():
    """
    Default obs space
    """
    return Box(low=-1.0, high=1.0, shape=(1429,), dtype=np.float32)


@pytest.fixture
def action_space():
    """
    Default action space
    """

    return Discrete(806)


@pytest.fixture
def custom_config():
    """
    Custom config for advanced junior model
    """
    default_params =  {'activation': 'relu',
                              'initializer': "Z",
                              'layer1': 1000,
                              'layer2': 1000,
                              'layer3': 1000,
                              'layer4': 1000}

    custom_config = {"model_path": Path(__file__).parent / "data"/"junior_experience" / "model",
                    "custom_config": default_params}
    return custom_config

@pytest.fixture
def senior_values(custom_config, test_paths_env, test_path_data, test_temp_save, test_action_single):
    """
    Values for the Senior Tests
    """
    env_path, act_path = test_paths_env
    actions_path = [act_path.parent / "submission" / "actionspace_nminus1.npy", act_path.parent / "submission" / "actionspace_tuples.npy"]
    path_to_junior = custom_config["model_path"]
    scaler = test_path_data / "scaler_junior.pkl"
    c_c = custom_config["custom_config"]
    return env_path,actions_path,path_to_junior,test_temp_save,c_c,scaler

@pytest.fixture
def rllib_ckpt(test_path_data):
    """
    Return the rllib checkpoint for the Senior Tests
    """
    ckpt = test_path_data / "PPO_SeniorEnvRllib" / "checkpoint_000001"
    return ckpt



