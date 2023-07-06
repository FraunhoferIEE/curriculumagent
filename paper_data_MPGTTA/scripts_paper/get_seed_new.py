"""
This file is running through various seeds, to check the configurations fo different seed s
"""
import pickle

import sys
import logging
from typing import Optional

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from l2rpn_baselines.ExpertAgent.expertAgent import ExpertAgent
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import BaseEnv
from curriculumagent.common.score_agent import load_or_run, render_report
from curriculumagent.tutor.tutors import general_tutor as general_tutor
from curriculumagent.submission.my_agent import MyAgent
from curriculumagent.tutor.tutors.n_minus_one_tutor import NminusOneTutor

plt.rcParams['figure.figsize'] = [15, 5]
date_strftime_format = "%Y-%m-%y %H:%M:%S"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt=date_strftime_format)


def run_evaluation_of_env(agent_dict: dict, env: BaseEnv, out_path: Path, seed: Optional[int]) -> dict:
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
                            number_episodes=number_of_runs,seed = seed, reinit=True) # Very Important, reinit True!!!
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
    surv_time ={dn_report.agent_name: dn_report.score_data['ts_survived']}
    for agent in agent_res:
        surv_time[agent.agent_name] = agent.score_data['ts_survived']

    return res_dict, surv_time


def create_agents_and_env():
    """ Simple Method to initialize the agents in order to make Pooling work

    Returns: dictionary of agent

    """
    backend = LightSimBackend()
    # Note we are only interested in the test env, thus number of episodes = 24
    env = grid2op.make("l2rpn_neurips_2020_test", backend=backend)
    action_space_dir = Path("action_spaces_paper")

    assert action_space_dir.is_dir()

    n_1_actions = [action_space_dir / "n1_actions.npy",
                   action_space_dir  / "actions62.npy",
                   action_space_dir  / "actions146.npy"]

    tutor_n1 = NminusOneTutor(action_space=env.action_space,
                              action_space_file=n_1_actions,
                              do_nothing_threshold=0.9,
                              best_action_threshold=0.99,
                              rho_greedy_threshold=0.99,
                              lines_to_check=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
                              return_status=True)

    tutor_n1_topo = NminusOneTutor(action_space=env.action_space,
                                   # action_space_file=ACTION_SPACE_DIRECTORY /"n1_actions.npy",
                                   action_space_file=n_1_actions,
                                   do_nothing_threshold=0.9,
                                   best_action_threshold=0.99,
                                   rho_greedy_threshold=0.99,
                                   lines_to_check=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
                                   return_status=True,
                                   revert_to_original_topo=True)

    tutor_general = general_tutor.Tutor(action_space=env.action_space,
                                          old_actionspace_path=action_space_dir )

    with open('scaler_junior.pkl', "rb") as fp:  # Pickling
        scaler = pickle.load(fp)
  
    expert_agent = ExpertAgent(action_space = env.action_space,
                           observation_space = env.observation_space,
                           name= "Expert",
                           gridName="IEEE118_R2")
  
    # Agents:
    agents = {"T Original": tutor_general,
              "T N-1": tutor_n1,
              "T N-1 Topo": tutor_n1_topo,
              "ExpertAgent": expert_agent,
              }

    my_agent_ckpt = MyAgent(
    action_space=env.action_space,
    model_path=Path("checkpoint"),
    action_space_path=n_1_actions,
    scaler=scaler,
    best_action_threshold=0.95,
    topo=True)

    agents["Senior N1"] = my_agent_ckpt

    return agents,env


def run_experiment_with_seed(seed: int):
    """ Execute the experiment with the given seed

    Args:
        seed:

    Returns:

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fixed_path = Path("seeds") / str(seed)

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
    Create the three main tutors
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(8888)
    seeds = list(np.random.randint(0, 10000, 30))  #

    logging.info(f"length of tasks:{len(seeds)}")
    
    #     # Save the results
    collect_scores = {}
      
    agents, env = create_agents_and_env()
    
    collect_survival_time = {}
    
    for seed in seeds:
        res, surv_time = run_evaluation_of_env(agent_dict=agents,
                                    env=env,
                                    out_path=Path("seeds") / str(seed),
                                    seed=seed)
        collect_scores[seed] = res
        collect_survival_time[seed] = surv_time

    with open('seed_res.pkl', 'wb') as handle:
        pickle.dump(collect_scores, handle)
        
    with open('collect_survival_time.pkl', 'wb') as handle:
        pickle.dump(collect_survival_time, handle)