"""
This file contains the different teacher types of the CurriculumAgent. The attacking teacher and the
general teacher are based on the original submission of @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution.

The remaining two teacher were created in the project AI2Go.
"""


import os
from pathlib import Path
from typing import Union, Optional

from curriculumagent.teacher.teachers.teacher1 import run_attacking_teacher
from curriculumagent.teacher.teachers.teacher2 import run_general_teacher


def attacking_teacher(save_path: Path,
                      env_name_path: Union[Path, str] = 'l2rpn_neurips_2020_track1_small',
                      *, n_episodes: int = 100, top_k: int = 125,
                      jobs: int = os.cpu_count(),
                      seed: Optional[int] = None):
    """This teacher is trying to collect useful actions by attacking/disconnecting specific lines at random timesteps
    and trying to find the best action to resolve that problem.

    Args:
        save_path: Path to save the experience file to.
        env_name_path: Path to grid2op dataset or the standard name of it.
        n_episodes: Number of episodes(run through all scenarios) to run.
        top_k: How many of the top best actions should be saved in the experience file at save_path.
        jobs: Number of jobs to use for parallel experience collection.
        seed: Optional seed for the Grid2Op Environment and random sampling
    """

    run_attacking_teacher(save_path=save_path,
                          env_name_path=env_name_path,
                          n_episodes=n_episodes,
                          top_k=top_k,
                          jobs=jobs,
                          seed=seed)


def general_teacher(save_path: Path,
                    env_name_path: Union[Path, str] = 'l2rpn_neurips_2020_track1_small',
                    rho_acting_threshold: float = 0.925,
                    *, n_episodes: int = 100, limit_chronics: int = None, top_k: int = 125,
                    jobs: int = os.cpu_count(), seed: int = 42):
    """This teacher is trying to collect useful actions by doing topology search while running an
    environment when an overflow occurs.

    Args:
        save_path: Path to save the experience file to.
        env_name_path: Path to grid2op dataset or the standard name of it.
        rho_acting_threshold: When the environments max_rho gues above this threshold the agent begins to act.
        n_episodes: Number of episodes(run through all scenarios) to run.
        limit_chronics: Limit the number of chronics to run through.
        top_k: How many of the top best actions should be saved in the experience file at save_path.
        jobs: Number of jobs to use for parallel experience collection.
        seed: The random seed to use.
    """
    run_general_teacher(save_path=save_path,
                        env_name_path=env_name_path,
                        rho_acting_threshold=rho_acting_threshold,
                        n_episodes=n_episodes,
                        limit_chronics=limit_chronics,
                        top_k=top_k,
                        jobs=jobs,
                        seed=seed)

