"""This is the advanced way to train the Senior model of the CurriculumAgent pipeline.
The Senior can trains the previous agent or Junior model an is based on the original work of Binbinchen.
Here the model uses RLlib to tune the hyper-parameter via Population Based Training.
"""
import os
from pathlib import Path
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModelTF, AdvancedCustomModel


def train_senior():
    """Wrapper method to train the senior with Ray. Please adjust accordingly.

    Args:
        None.

    Returns:
        None, Saves ray checkpoints.

    """

    # Init all variables and Ray
    NUM_CPUS = os.cpu_count()
    NUM_GPUS = 0  # Change if applicable
    NUM_TRIALS = 3

    DATA_PATH = Path("~/data_grid2op/l2rpn_neurips_2020_track1_small").expanduser()
    ACTION_SPACE_DIRECTORY = "../action_space"
    junior_model = Path(__file__).parent / "JuniorModel_new"

    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
    env_config = {
        "action_space_path": ACTION_SPACE_DIRECTORY,
        "env_path": DATA_PATH,
        "action_threshold": 0.9,
        "filtered_obs": True,
    }
    model_config = {"path_to_junior": junior_model}
    ModelCatalog.register_custom_model("Senior", Grid2OpCustomModelTF)

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
            "entropy_coeff": lambda: 10 ** -random.uniform(2, 5),
        },
    )

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
        stop={"training_iteration": 10},
        config={
            "env": SeniorEnvRllib,
            "env_config": env_config,
            "num_workers": (NUM_CPUS - NUM_TRIALS) / NUM_TRIALS,
            "num_envs_per_worker": 5,
            "lr": 5e-5,
            "num_gpus": NUM_GPUS / NUM_TRIALS,
            "num_cpus_per_worker": 1,
            "remote_worker_envs": False,
            # "model": {"use_lstm": False, "fcnet_hiddens": [1000, 1000, 1000, 1000], "fcnet_activation": 'relu',
            #           "vf_share_layers": True},
            "model": {"custom_model": "binbinchen", "custom_model_config": model_config},
        },
    )

