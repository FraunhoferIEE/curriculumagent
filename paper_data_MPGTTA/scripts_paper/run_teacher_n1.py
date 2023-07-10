import logging
import multiprocessing
import os

import grid2op
from lightsim2grid import LightSimBackend

from curriculumagent.teacher.teachers.teacher_n_minus_1 import NMinusOneTeacher


def main():
    """
    Run N-1 Teacher
    Returns: Nothing, values are saved in directory.

    """
    backend = LightSimBackend()
    env = grid2op.make("l2rpn_neurips_2020_track1_small", backend=backend)
    obs = env.get_obs()

    n1_agent = NMinusOneTeacher(rho_n0_threshold=0.9,
                                rho_max_threshold=1.0)

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    log_format = '(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]'
    logging.basicConfig(level=logging.INFO, format=log_format)

    n1_agent.collect_n_minus_1_experience(save_path='N1',
                                          env_name_path="l2rpn_neurips_2020_track1_small",
                                          number_of_years=None, jobs=os.cpu_count(),
                                          save_greedy=False,
                                          active_search=False,
                                          disable_opponent=True)


if __name__ == "__main__":
    main()
