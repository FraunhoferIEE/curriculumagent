"""
Script to run the senior model with rllib.

This script includes a check prior to the experiment to ensure that the environment and the model
work correctly.
"""

import logging
import os
import pickle
import random
import json
from pathlib import Path
from typing import Tuple

import sklearn
import tensorflow as tf
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.rllib_execution.senior_model_rllib import AdvancedCustomModel


def get_configs() -> Tuple[SeniorEnvRllib, AdvancedCustomModel]:
    """ Get default configs

    Returns: The Rllib Environment and a custom Model

    """
    Agent_Path = Path(__file__).parent.parent.parent
    actions = [Agent_Path / "action_space"/ "new" / "actionspace_208_poth.npy"]
    target = Agent_Path / 'senior'/"JuniorModel"

    e_c = {"action_space_path": actions,
           "data_path": Agent_Path.parent/"training_data_track1",
           "action_threshold": 0.9,
           'filtered_obs': True,
           'scaler_path': Agent_Path/"data"/"scaler_junior.pkl"}

    with open(Agent_Path/"data"/ 'junior_best_params.json') as json_file:
        best_params = json.load(json_file)

    best_params["epochs"] = 1000
    best_params["initializer"] = tf.keras.initializers.Orthogonal()
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        best_params[name] = np.round(best_params[name])

    m_c = {"path_to_junior": target,
           "custom_config": best_params}
    return e_c, m_c


def try_configs(env_c, model_c):
    """ Ensuring that both the environment and the model are working, before running the
    rllib optimization

    Args:
        env_c: RllibEnvrionment
        model_c: Custom Model

    Returns: None

    """

    with open(env_c["scaler_path"], "rb") as fp:  # Pickling
        scaler = pickle.load(fp)
    assert isinstance(scaler, sklearn.preprocessing.MinMaxScaler)

    # Test custom Env:
    env = SeniorEnvRllib(env_c)
    done = False
    a = None
    while done is False:
        act = random.choice(np.arange(env.action_space.n))

        a, _, done, _ = env.step(act)
        logging.info(a, done, env.step_in_env, max(a), min(a))

    logging.info("Environment does work!")

    model = AdvancedCustomModel(obs_space=env.observation_space,
                                action_space=env.action_space,
                                num_outputs=env.action_space.n,
                                model_config={},
                                path_to_junior=model_c["path_to_junior"],
                                custom_config=model_c["custom_config"],
                                name="Junior")
    logging.info(model.base_model.summary())

    obs = {"obs": a.reshape(1, -1)}
    model.forward(input_dict=obs, state=[1], seq_lens=None)
    logging.info("Model does work!")


def train_senior(env_c, model_c):
    """ Wrapper method to train the senior with ray. Please adjust accordingly.

    Returns: None, Saves ray checkpoints

    """

    # Init all variables and Ray
    NUM_TRIALS = 2
    NUM_CPUS = os.cpu_count()
    NUM_GPUS = len(tf.config.list_physical_devices('GPU'))

    if ray.is_initialized is False:
        ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)

    ModelCatalog.register_custom_model('binbinchen', AdvancedCustomModel)

    # Set population based training
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=2,
        resample_probability=0.5,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(3, 10),
            "vf_loss_coeff": lambda: random.uniform(0.5, 1),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "gamma": lambda: random.uniform(0.975, 1),
            "entropy_coeff": lambda: 10 ** -random.uniform(2, 5)
        })

    # Run Ray:
    tune.run(
        "PPO",
        checkpoint_freq=1,
        scheduler=pbt,
        keep_checkpoints_num=5,
        verbose=1,
        max_failures=3,
        num_samples=NUM_TRIALS,  # Adjust number of samples accordingly
        local_dir="~/bm",
        stop={"training_iteration": 5000},
        config={
            "env": SeniorEnvRllib,
            "env_config": env_c,
            "num_workers": (NUM_CPUS - NUM_TRIALS) / NUM_TRIALS,
            "num_envs_per_worker": 5,
            "lr": 5e-5,
            "num_gpus": NUM_GPUS / NUM_TRIALS,
            "num_cpus_per_worker": 1,
            "remote_worker_envs": False,
            # "model": {"use_lstm": False, "fcnet_hiddens": [1000, 1000, 1000, 1000], "fcnet_activation": 'relu',
            #           "vf_share_layers": True},
            "model": {"custom_model": "binbinchen", "custom_model_config": model_c},

        },
    )


if __name__ == '__main__':
    """
    Run Main Rllib training:
    """
    logging.basicConfig(level=logging.INFO)

    # Load configs
    env_config, model_config = get_configs()
    logging.info(env_config)
    logging.info(model_config)

    # test whether configs work
    try_configs(env_config, model_config)

    # Register

    ModelCatalog.register_custom_model('binbinchen', AdvancedCustomModel)
    logging.basicConfig(level=logging.INFO)

    train_senior(env_config, model_config)

    ray.shutdown()

