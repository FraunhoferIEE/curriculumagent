"""
This file can be used to define extra pytest rules and fixtures
"""
import os
import pickle
from pathlib import Path

import grid2op
import numpy as np
import pytest
import tensorflow as tf

from curriculumagent.junior.junior_student import load_dataset
from curriculumagent.senior.openai_execution.ppo_reward import PPO_Reward

# We formulate two additional measures to assure that the run slow and ultra slow are triggered:
from lightsim2grid import LightSimBackend

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: Tests will be slow. "
    )

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


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
    """ Get the test path of the env, depending if you run the code locally or all pytest through the console

    Returns:

    """
    test_env_path = Path(__file__).parent / 'training_data_track1'

    test_actions_path = Path(__file__).parent.parent / "curriculumagent" / 'action_space'
    return test_env_path, test_actions_path

@pytest.fixture
def test_paths_ieee_model():
    """ Get the test path of the env, depending if you run the code locally or all pytest through the console

    Returns:

    """
    model_path = Path(__file__).parent.parent / "curriculumagent" / "baseline" / "model_IEEE14"

    return model_path


@pytest.fixture
def test_path_rllib():
    """ Get the test path of rllib data, depending if you run the code locally or all pytest through the console

    Returns:

    """
    test_rllib_data = os.path.join(Path(__file__).parent / "data" / "senior_training_ckpt")

    return test_rllib_data


@pytest.fixture
def test_env(test_paths_env):
    """
    Returns a specifc grid2op Env with a specific scenario
    """
    backend = LightSimBackend()
    env_path, _ = test_paths_env
    data_path = str(env_path)
    scenario_path = str(env_path / 'chronics')
    env = grid2op.make(dataset=data_path, chronics_path=scenario_path, backend=backend,
                       reward_class=PPO_Reward, test=True)
    env.set_id(1)
    env.reset()
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
def test_junior_input():
    """
    Junior files
    """
    tf.random.set_seed(42)
    np.random.seed(42)
    data_path = Path(__file__).parent / "data" / "junior_data"
    s_tr, a_tr, s_v, a_v, s_te, a_te = load_dataset(dataset_path=data_path,
                                                    dataset_name="test")
    return s_tr, a_tr, s_v, a_v, s_te, a_te


@pytest.fixture
def test_temp_save():
    """
    Temporary folder for saving data
    """
    data_path = Path(__file__).parent / "data" / "temporary_save"
    return data_path


@pytest.fixture
def test_submission_models():
    """
    Test paths of the original submission model
    """
    submission_dir =  Path(__file__).parent / "data"  / "submission"
    old_m = submission_dir / "old"
    new_m = submission_dir / "new"
    return old_m, new_m

@pytest.fixture
def test_scaler():
    """
    Get MinMaxScaler
    """
    scaler_path = Path(__file__).parent / "data" /"scaler_junior.pkl"
    with open(scaler_path , "rb") as fp:  # Pickling
        scaler = pickle.load(fp)
    return scaler