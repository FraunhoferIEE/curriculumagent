# -*- coding: utf-8 -*-
"""
Created on Fri October 28 16:57:47 2022
@author: fmarten
Goal: to re-run agent with (or without) saved state variables to test if its
actions during a specific Grid2Op time step can be exactly reproduced.
"""
import copy
import grid2op
from my_agent_savestate_rosalie import make_agent
import pandas as pd
import time
import os

'''
User defined methods
'''


def grid2op_loop(max_iter=0):
    # Perform a grid2op step loop until max_iter is reached
    done = False
    obs = env.current_obs
    reward = 0.0
    step = 0
    action_sum = 0
    while done is False and step < (max_iter+1):
        agent_action = binbinchen_agent.act(obs, reward, done)
        action_sum = sum(agent_action.set_line_status)
        obs, reward, done, info = env.step(agent_action)
        step += 1
        if action_sum > 0:
            print("Agent performed at least one action")
    return obs, reward, done, info, agent_action


def re_run_agent_action(iteration, obs_previous, reward_previous,
                        done_previous, time_step_before_action):
    # Re-load the GridOp and Agent state at a specific time step
    # and repeat the agent action multiple times, to test if identical.
    df_result = pd.DataFrame(
        columns=['switch_bus_ids', 'switch_bus_types', 'switch_bus_subs_ids',
                 'lines_connected', 'lines_disconnected'],
        index=range(iteration))

    # Loop over the same Grid2op time step N times and record agent actions
    for i in df_result.index.values:
        binbinchen_agent.load_state()
        env_previous = copy.deepcopy(env)
        agent_action_next = binbinchen_agent.act(obs_previous, reward_previous,
                                                 done_previous)
        obs_next, reward_next, done_next, info_next = env_previous.step(
            agent_action_next)
        # Save agent action into a list during each iteration
        adict = agent_action_next.as_dict()
        if 'change_bus_vect' in adict:
            df_result.at[i, 'switch_bus_ids'] = list(
                adict['change_bus_vect']['0'])
            df_result.at[i, 'switch_bus_types'] = [
                d['type'] for d in adict['change_bus_vect']['0'].values()]
            df_result.at[i, 'switch_bus_subs_ids'] = adict[
                'change_bus_vect']['modif_subs_id']

        if 'set_line_status' in adict:
            df_result.at[i, 'lines_connected'] = adict[
                'set_line_status']['connected_id']
            df_result.at[i, 'lines_disconnected'] = adict[
                'set_line_status']['disconnected_id']

    return agent_action_next, df_result


def df_compare(df_result):
    # Compare each rows of dataframe for similarity
    df_result = df_result.fillna(-1)
    df_result = df_result.astype(str)  # lists to strings, comparison is easier

    if ((df_result['switch_bus_ids'].values == df_result[
            'switch_bus_ids'][0]).all()
        and (df_result['switch_bus_types'].values == df_result[
            'switch_bus_types'][0]).all()
        and (df_result['switch_bus_subs_ids'].values == df_result[
            'switch_bus_subs_ids'][0]).all()):
        print('All switch bus actions were equal!')
    else:
        print('At least one switch bus action was different!')

    if ((df_result['lines_connected'].values == df_result[
            'lines_connected'][0]).all()
        and (df_result['lines_disconnected'].values == df_result[
            'lines_disconnected'][0]).all()):
        print('All line connection actions were equal!')
    else:
        print('All line connection action was different!')


'''
Main program
'''
# Create a new seeded environment
env = grid2op.make(dataset="l2rpn_neurips_2020_track1_small")
agent_seed = 1
env.seed(3)
time_step_before_action = 339  # 226 or 339 for exemplar times of an action

# Create a seeded binbinchen agent and a path to store its state variables
if not os.path.exists(r'agent_state'):
    os.mkdir(r'agent_state')

binbinchen_agent = make_agent(
    env=env,
    this_directory_path=r'',
    savestate_path=r'agent_state')
binbinchen_agent.seed(agent_seed)

# Run grid2op until just before the agent performs an action, save state
obs_before, reward_before, done_before,\
    info_before, agent_action_before = grid2op_loop(
        max_iter=time_step_before_action)
binbinchen_agent.save_state()

# Repeat the Grid2Op time step with re-loaded agent state, compare actions
t0 = time.time()
agent_action_next, df_result = re_run_agent_action(
    iteration=5,
    obs_previous=obs_before,
    reward_previous=reward_before,
    done_previous=done_before,
    time_step_before_action=time_step_before_action)

t1 = time.time()
print("Runtime of repeated Grid2Op time step loop was %d seconds." % (t1-t0))
df_compare(df_result)
