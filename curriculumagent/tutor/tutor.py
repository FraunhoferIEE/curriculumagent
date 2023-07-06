"""This file consist of the two tutor approaches. The general_tutor method is the more general approach.
that enables multiple tweaks in order to enhance the training.
The original_tutor method is the original approach by @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution

"""

from pathlib import Path
from typing import Union, Optional, List

from curriculumagent.tutor.collect_tutor_experience import generate_tutor_experience
from curriculumagent.tutor.tutors.n_minus_one_tutor import NminusOneTutor


def general_tutor(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        seed: Optional[int] = None,
):
    """Method to run the general Tutor in parallel

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the teacher_experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. With replacement!
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments.

    Returns:
        None, saves results as numpy file.

    """
    generate_tutor_experience(
        env_name_path=env_name_path,
        save_path=save_path,
        action_paths=action_paths,
        num_chronics=num_chronics,
        num_sample=num_sample,
        jobs=jobs,
        seed=seed,
    )


def n_minus_1_tutor(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        seed: Optional[int] = None,
        revert_to_original_topo: Optional[bool] = False,
):
    """Method to run the general Tutor in parallel.

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the teacher_experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. With replacement!
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments.
        revert_to_original_topo: Whether the N-1 Tutor should revert to original topology
        within the optimization process.
.
    Returns:
        None, saves results as numpy file.

    """

    if revert_to_original_topo:
        tutor_kwargs = {"revert_to_original_topo": True}
    else:
        tutor_kwargs = None

    generate_tutor_experience(
        env_name_path=env_name_path,
        save_path=save_path,
        action_paths=action_paths,
        num_chronics=num_chronics,
        num_sample=num_sample,
        jobs=jobs,
        seed=seed,
        TutorAgent=NminusOneTutor,
        tutor_kwargs=tutor_kwargs,
    )
