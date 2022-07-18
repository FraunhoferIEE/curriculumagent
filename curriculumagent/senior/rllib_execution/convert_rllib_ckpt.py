"""
This file consists of methods to collect the checkpoints from the rllib training. See the notebooks
for the detailed usage
"""

import logging
import os
import json
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
from ray.rllib.agents.ppo.ppo import PPOTrainer

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib

# TODO: Is this method necessary ? I recon, that we have to overwrite the config anyway. Thus,
# This might not be enought.
def collect_ckpt_from_ray_dir(folder: Union[Path, str],
                              save_path: Union[Path, str],
                              ckpt_nr: Optional[Union[int, str]] = None) -> None:
    """
    This method collects the checkpoints of the ray/rllib training and saves them in the
    readable tensorflow format of .pb

    Depending on the ckpt_nr, this method either collects all checkpoints (ckpt_nr=None), one specific
    checkpoint (e.g. ckpt_nr=42) or the latest checkpoint (ckpt_nr="latest").

    Note that the folder has to be the directory, where to find all checkpoints, as well as the
    params.jsons, tensorboard saves etc.

    Args:
        folder: path where to find the ray checkpoints, default is oftern ray_results
        save_path: path, where to save the checkpoints
        ckpt_nr: option to either load a specific checkpoint nr (int) or the latest checkpoint by
            implementing "latest".

    Returns: None, saving the checkpoints at requested directory.

    """

    checkpoint_numbers = [int(file.split("_")[-1]) for file in os.listdir(folder) if "checkpoint" in file]
    checkpoint_numbers = np.sort(checkpoint_numbers)

    if isinstance(folder, str):
        folder = Path(folder)

    rllib_config,_ = load_config(folder)

    out = ""
    if ckpt_nr is None and len(checkpoint_numbers) > 0:
        for ckpt_nr in checkpoint_numbers:
            ckpt_path = folder / f"checkpoint_{ckpt_nr:06}" / f"checkpoint-{ckpt_nr}"
            load_and_save_model(ckpt_path=ckpt_path,
                                config=rllib_config,
                                save_path=save_path,
                                ckpt_nr=ckpt_nr)

    if isinstance(ckpt_nr, int) and ckpt_nr in checkpoint_numbers:
        ckpt_path = folder / f"checkpoint_{ckpt_nr:06}" / f"checkpoint-{ckpt_nr}"
        load_and_save_model(ckpt_path=ckpt_path,
                            config=rllib_config,
                            save_path=save_path,
                            ckpt_nr=ckpt_nr)

    if ckpt_nr == "latest":
        latest_checkpoint_number = max(checkpoint_numbers)
        ckpt_path = folder / f"checkpoint_{latest_checkpoint_number:06}" / f"checkpoint-{latest_checkpoint_number}"
        load_and_save_model(ckpt_path=ckpt_path,
                            config=rllib_config,
                            save_path=save_path,
                            ckpt_nr=ckpt_nr)


def load_and_save_model(ckpt_path: Union[Path, str], config: dict, save_path: Union[Path, str],
                        ckpt_nr: Optional[int] = None) -> None:
    """ Method to load the rllib checkpoints into the PPOTrainer and then save them as .pb format.

    Args:
        ckpt_path: specific path of the checkpoint
        config: training config of rllib
        save_path: save path, where to save the checkpoint
        ckpt_nr: Optional number of ckpt for saving, if it should be renamed

    Returns:

    """
    config.update({'num_workers': 0, 'num_gpus': 0, 'num_envs_per_worker': 1,
                   'env': SeniorEnvRllib})

    config["env_config"]["scaler"] = None

    if isinstance(ckpt_path, Path):
        ckpt_path = str(ckpt_path)

    logging.info(f"Following config is initialized: {config}")

    evaluator = PPOTrainer(env=SeniorEnvRllib, config=config)

    evaluator.restore(ckpt_path)
    evaluator.export_model(export_formats="model", export_dir=save_path)

    # Delete access on the path
    del evaluator

    if ckpt_nr:
        old_model_path = Path(save_path) / "model"
        name = "ckpt_" + str(ckpt_nr)
        new_model_path = Path(save_path) / name
        assert old_model_path.is_dir()
        old_model_path.rename(new_model_path)
        assert new_model_path.is_dir()


def load_config(folder: Union[Path, str], latest: bool = False) -> Tuple[dict, list]:
    """ Loading the rllib config file and additionally return a list of the checkpoint direcories

    Args:
        folder: path, where to find the config
        latest: Whether or not only the lates checkpoint should be returned

    Returns: dictionary

    """
    config_path = Path(folder) / "params.json"
    with open(config_path) as json_file:
        config = json.load(json_file)

    if isinstance(config['env_config']['action_space_path'], str):
        config['env_config']['action_space_path'] = Path(config['env_config']['action_space_path'])
    if isinstance(config['env_config']['env_path'], str):
        config['env_config']['env_path'] = Path(config['env_config']['env_path'])

    if latest:
        checkpoint_list = path_to_latest_checkpoint(folder)
    else:
        checkpoint_list = paths_to_checkpoints(folder)

    return config, checkpoint_list


def paths_to_checkpoints(folder: Union[str, Path]) -> List:
    """ Loading all Checkpoints of a Path. Usfull to have a list of the Rllib experiments

    Args:
        folder: directories

    Returns: List

    """
    files = os.listdir(folder)
    checkpoint_numbers = [int(file.split("_")[-1]) for file in files if "checkpoint" in file]
    checkpoint_numbers = np.sort(checkpoint_numbers)
    if len(checkpoint_numbers) > 0:
        return [os.path.join(folder, f"checkpoint_{str(chk_nr).zfill(6)}", f"checkpoint-{chk_nr}") for chk_nr in
                checkpoint_numbers]

    return ""


def path_to_latest_checkpoint(folder: Union[str, Path]) -> List:
    files = os.listdir(folder)
    checkpoint_numbers = [int(file.split("_")[-1]) for file in files if "checkpoint" in file]
    if len(checkpoint_numbers) > 0:
        latest_checkpoint_number = max(checkpoint_numbers)

        # print("LATEST CHKP: ", latest_checkpoint_number )
        # latest_checkpoint_number = 600
        return os.path.join(folder, f"checkpoint_{str(latest_checkpoint_number).zfill(6)}", f"checkpoint-{latest_checkpoint_number}")

    return ""
