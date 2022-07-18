"""
This file consist of the two tutor approaches. The general_tutor method is the more general approach
that enables multiple tweaks in order to enhance the training.
The original_tutor method is the original approach by @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution

"""

from pathlib import Path
from typing import Union, Optional, List

from curriculumagent.tutor.tutors.general_tutor import generate_tutor_experience
from curriculumagent.tutor.tutors.original_tutor import generate_original_tutor_data


def general_tutor(env_name_path: Union[Path, str],
                  save_path: Union[Path, str],
                  action_paths: Union[Path, List[Path]],
                  num_chronics: Optional[int] = None,
                  num_sample: Optional[int] = None,
                  jobs: int = -1,
                  seed: Optional[int] = None):
    """ Method to run the general Tutor in parallel

    Args:
        env_name_path: Path to grid2op dataset or the standard name of it.
        save_path: Where to save the experience
        action_paths: List of action sets (in .npy format)
        num_chronics: Total numer of chronics
        num_sample: length of sample from the num_chronics. With replacement!
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments

    Returns: None, Saves results as numpy file.

    """
    generate_tutor_experience(env_name_path=env_name_path,
                              save_path=save_path,
                              action_paths=action_paths,
                              num_chronics=num_chronics,
                              num_sample=num_sample,
                              jobs=jobs,
                              seed=seed)


def original_tutor(env_name_path: Union[Path, str],
                   save_path: Union[Path, str],
                   action_paths: Path,
                   num_chronics: Optional[int] = None,
                   seed: Optional[int] = None):
    """ Method to run the original tutor, similar to the general tutor. Note: The original tutor
    is deprecated and will be removed in later versions.

    Args:
        env_name_path: Path to grid2op dataset or the standard name of it.
        save_path: Where to save the experience
        action_paths: List of action sets (in .npy format)
        num_chronics: Total numer of chronics
        seed: Whether to set a seed to the sampling of environments

    Returns: Experience of the tutor

    """
    generate_original_tutor_data(env_name_path=env_name_path,
                                 save_path=save_path,
                                 action_paths=action_paths,
                                 num_chronics=num_chronics,
                                 seed=seed)
