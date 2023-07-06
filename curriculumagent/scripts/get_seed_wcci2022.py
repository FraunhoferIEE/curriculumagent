"""
This file is running through various seeds, to check the configurations fo different seed s
"""

import pickle

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from lightsim2grid import LightSimBackend
import grid2op
import sys
import logging
from typing import Optional, Tuple

from l2rpn_baselines.ExpertAgent.expertAgent import ExpertAgent
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import BaseEnv
from curriculumagent.common.score_agent import load_or_run, render_report
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor
from curriculumagent.submission.my_agent import MyAgent
from curriculumagent.tutor.tutors.n_minus_one_tutor import NminusOneTutor

plt.rcParams['figure.figsize'] = [15, 5]
date_strftime_format = "%Y-%m-%y %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt=date_strftime_format)


def run_evaluation_of_env(agent_dict: dict, env: BaseEnv, out_path: Path, seed: Optional[int]) -> Tuple[dict, dict]:
    """ Calculate the performance of the provided agents (from the agents dicts)

    Note: The do nothing agent is always provided!

    Args:
        agent_dict: Dictionary containing multiple agents
        env: Grid2Op environment
        out_path: path, where to save the results and cached results
        seed: Optional seed for execution.

    Returns: None

    """
    if out_path.is_dir() is False:
        os.mkdir(out_path)

    number_of_runs = len(os.listdir(env.chronics_handler.path))

    do_nothing_agent = DoNothingAgent(env.action_space)
    # We use the reinit method to load 
    dn_report = load_or_run(agent=do_nothing_agent, env=env, output_path=out_path, name="DoNothing",
                            number_episodes=number_of_runs, seed=seed, reinit=True)  # Very Important, reinit True!!!
    print(f"The Do-Nothing agent has the scores of: {dn_report.score_data['all_scores']}")

    agent_res = []

    for name, agent in agent_dict.items():
        if ("Senior" in name) or ("Expert" in name):
            print("Run Senior Original with nb_process 1")
            agent_res.append(load_or_run(agent, env=env, output_path=out_path, name=name,
                                         nb_processes=1, number_episodes=number_of_runs, seed=seed))
        else:
            agent_res.append(load_or_run(agent, env=env, output_path=out_path, name=name,
                                         nb_processes=os.cpu_count(), number_episodes=number_of_runs, seed=seed))

    render_report(out_path / 'report.md', dn_report, agent_res)

    # collect the mean of the overall scores: 
    res_dict = {dn_report.agent_name: dn_report.avg_score}
    for agent in agent_res:
        res_dict[agent.agent_name] = agent.avg_score

    # Collect the surviving time
    surv_time = {dn_report.agent_name: dn_report.score_data['ts_survived']}
    for agent in agent_res:
        surv_time[agent.agent_name] = agent.score_data['ts_survived']

    return res_dict, surv_time


def create_agents_and_env():
    """ Simple Method to initialize the agents in order to make Pooling work

    Returns: dictionary of agent

    """
    backend = LightSimBackend()
    # Note we are only interested in the test env, thus number of episodes = 24
    env = grid2op.make("l2rpn_wcci_2022_valscenario", backend=backend)

    n_1_actions = [Path("/home/mlehna/AI2Go/l2rpn_binbinchen_iee/Update2023/wcci_2022_actions.npy"),
                   Path("/home/mlehna/AI2Go/l2rpn_binbinchen_iee/Update2023/maze_actions.npy")]
    disc_lines = [136, 146, 149, 156, 155, 180, 83, 154, 121, 93, 138, 88,
                  148, 86, 110, 106, 68, 162, 141, 79, 81, 150, 152, 153,
                  160, 131, 132, 126, 108, 62, 89, 61, 125, 91, 133, 82,
                  51, 175, 127, 84, 120, 76]

    tutor_n1 = NminusOneTutor(action_space=env.action_space,
                              action_space_file=n_1_actions,
                              do_nothing_threshold=0.9,
                              best_action_threshold=0.99,
                              rho_greedy_threshold=0.99,
                              lines_to_check=disc_lines,
                              return_status=True,
                              revert_to_original_topo=True)

    tutor_original = GeneralTutor(action_space=env.action_space,
                                  action_space_file=n_1_actions)

    agents = {"T Original": tutor_original,
              "T N-1": tutor_n1,
              }

    return agents, env


def run_experiment_with_seed(seed: int):
    """ Execute the experiment with the given seed

    Args:
        seed:

    Returns:

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fixed_path = Path("/share/data1/GYM/wcci2022/seeds") / str(seed)

    agents, env = create_agents_and_env()
    logging.info(f"Run evaluation of seed {seed}.")
    res = run_evaluation_of_env(agent_dict=agents,
                                env=env,
                                out_path=fixed_path,
                                seed=seed)
    logging.info(f"Seed {seed} done with score {res}")

    return res, seed


if __name__ == "__main__":
    """
    Create the different agents e.g. Tutors and Seniors
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(8888)
    seeds = list(np.random.randint(0, 10000, 10))
    logging.info(f"length of tasks:{len(seeds)}")

    #     # Save the results
    collect_scores = {}

    agents, env = create_agents_and_env()

    collect_survival_time = {}

    for seed in seeds:
        res, surv_time = run_evaluation_of_env(agent_dict=agents,
                                               env=env,
                                               out_path=Path("/share/data1/GYM/wcci2022/seeds") / str(seed),
                                               seed=seed)
        collect_scores[seed] = res
        collect_survival_time[seed] = surv_time

    with open('/share/data1/GYM/wcci2022/seeds/seed_res.pkl', 'wb') as handle:
        pickle.dump(collect_scores, handle)

    with open('/share/data1/GYM/wcci2022/seeds/collect_survival_time.pkl', 'wb') as handle:
        pickle.dump(collect_survival_time, handle)
