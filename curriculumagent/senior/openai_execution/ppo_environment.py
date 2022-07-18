"""
In this file, a multi-process training for PPO model is designed.

Training process:

    The environment steps “do nothing” action (except reconnection of lines)
    until encountering a dangerous scenario, then its observation is sent to
    the Senior Student to get a “do something” action. After stepping this
    action, the reward is calculated and fed back to the Senior Student for
    network updating.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
import os
from pathlib import Path
from typing import Union, Optional, List
import tensorflow as tf
import numpy as np
from grid2op.Environment import SingleEnvMultiProcess
from grid2op.Observation import BaseObservation
from curriculumagent.common.utilities import array2action
from curriculumagent.senior.openai_execution.ppo import PPO

class Run_env():
    def __init__(self, envs: SingleEnvMultiProcess, agent: PPO, num_core: int, action_threshold: float,
                 n_steps: int = 2000, gamma: float = 0.99, lam: float = 0.95,
                 action_space_path: Union[str, Path] = '../'):
        """ Class to run the environment in parallel for the optimization of the Senior Student.

        Within the Run_env class the environment is executed in parallel with the following process:
        As long as no dangerous scenario occurs, the environment automatically does "do nothing" as
        action, as well as reconnects lines (when possible). In case a dangerous scenario happends, the
        observation is then send to the Senior Student to train a topology action.
        After the execution of the action, the reward is calculated and all information of the step
        are fed to the PPO algorithm for network updating.

        Args:
            envs: Multiple Grid2op environment created by the SingleEnvMultiProcess class for calculating
            agent: PPO model to compute the actions
            num_core: number of cpu cores
            action_threshold: Threshold between 0.0 and 1.0, indicating when the dangerous scenario starts.
                For example: if the threshold is set to 0.9, the agent has to act, when the max rho is larger
                or equal to 0.9
            n_steps: number of steps to run through the environment.
            gamma: Variable showing the future discount factor
            lam: lambda of the PPO algorithm
            action_space_path: path of the action space for the Senior student.
        """
        self.envs = envs
        self.agent = agent
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(
            range(1164, 1223))
        assert num_core > 0
        assert 0.0 < action_threshold <= 1.0
        self.num_core = num_core
        self.action_threshold = action_threshold
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)
        self.actions62 = np.load(os.path.join(action_space_path, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(action_space_path, 'actions146.npy'))
        self.actions = np.concatenate((self.actions62, self.actions146), axis=0)
        self.batch_reward_records = []
        self.aspace = self.envs.action_space[0]
        self.rec_rewards = []
        self.worker_alive_steps = np.zeros(self.num_core)
        self.alive_steps_record = []

    def run_n_steps(self, n_steps: Optional[int] = None) -> List:
        """ Method to run n steps within the simulation. The default is set to 2000 by the __init__
        method. For each step the agent runs through multiple instances of the environment and the
        results are returned in a combined list

        This function returns the combined:
            observations, returns (i.e. rewards), Information whether the environment is done,
             the combined actions, the combined values of the value functions, the combined log-likelihood
             and the average return(i.e. reward).

        Args:
            n_steps: Number of steps.

        Returns: List containing the values described above

        """

        def swap_and_flatten(arr):
            shape = arr.shape
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

        self.n_steps = n_steps if n_steps is not None else self.n_steps
        # mb for mini-batch

        # Assign empty list
        mb_obs = [[] for _ in range(self.num_core)]
        mb_rewards = mb_obs.copy()
        mb_actions = mb_obs.copy()
        mb_values = mb_obs.copy()
        mb_dones = mb_obs.copy()
        mb_neg_log_p = mb_obs.copy()

        # start sampling
        obs_objs = self.envs.get_obs()
        obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])  # (12, 1221,)
        dones = np.asarray([False for _ in range(self.num_core)])  # (12,)
        agent_step_rs = np.asarray([0 for _ in range(self.num_core)], dtype=np.float64)  # (12,)
        for _ in range(self.n_steps):
            self.worker_alive_steps += 1
            actions = np.asarray([None for _ in range(self.num_core)])  # 均为shape=(12,)的np array
            values = np.asarray([None for _ in range(self.num_core)])
            neg_log_ps = np.asarray([None for _ in range(self.num_core)])
            for id in range(self.num_core):
                if obss[id, 654:713].max() >= self.action_threshold:
                    actions[id], values[id], neg_log_ps[id], _ = map(lambda x: x._numpy(),
                                                                     self.agent.step(tf.constant(obss[[id], :])))
                    if dones[id] is False and len(mb_obs[id]) > 0:
                        mb_rewards[id].append(agent_step_rs[id])
                    agent_step_rs[id] = 0
                    mb_obs[id].append(obss[[id], :])
                    mb_dones[id].append(dones[[id]])
                    dones[id] = False
                    mb_actions[id].append(actions[id])
                    mb_values[id].append(values[id])
                    mb_neg_log_p[id].append(neg_log_ps[id])
                else:
                    pass
            actions_array = [
                array2action(self.aspace, self.actions[i][0]) if i is not None else \
                array2action(self.aspace, np.zeros(494), Run_env.reconnect_array(obs_objs[idx])) \
                for idx, i in enumerate(actions)
            ]
            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)
            obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])
            for id in range(self.num_core):
                if env_dones[id]:
                    # death or end
                    self.alive_steps_record.append(self.worker_alive_steps[id])
                    self.worker_alive_steps[id] = 0
                    if 'GAME OVER' in str(infos[id]['exception']):
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] - 300)  # 上一个agent step的reward
                    else:
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] + 500)
            agent_step_rs += rs
        # end sampling

        # batch to trajectory
        for id in range(self.num_core):
            if mb_obs[id] == []:
                continue
            if dones[id]:
                mb_dones[id].append(np.asarray([True]))
                mb_values[id].append(np.asarray([0]))
            else:
                mb_obs[id].pop()
                mb_actions[id].pop()
                mb_neg_log_p[id].pop()
        obs2ret, done2ret, action2ret, value2ret, neglogp2ret, return2ret = ([] for _ in range(6))
        for id in range(self.num_core):
            if mb_obs[id] == []:
                continue
            mb_obs_i = np.asarray(mb_obs[id], dtype=np.float32)
            mb_rewards_i = np.asarray(mb_rewards[id], dtype=np.float32)
            mb_actions_i = np.asarray(mb_actions[id], dtype=np.float32)
            mb_values_i = np.asarray(mb_values[id][:-1], dtype=np.float32)
            mb_neg_log_p_i = np.asarray(mb_neg_log_p[id], dtype=np.float32)
            mb_dones_i = np.asarray(mb_dones[id][:-1], dtype=np.bool)
            last_done = mb_dones[id][-1][0]
            last_value = mb_values[id][-1][0]

            # calculate R and A
            mb_advs_i = np.zeros_like(mb_values_i)
            last_gae_lam = 0
            for t in range(len(mb_obs[id]))[::-1]:
                if t == len(mb_obs[id]) - 1:
                    # last step
                    next_non_terminal = 1 - last_done
                    next_value = last_value
                else:
                    next_non_terminal = 1 - mb_dones_i[t + 1]
                    next_value = mb_values_i[t + 1]
                # calculate delta：r + gamma * v' - v
                delta = mb_rewards_i[t] + self.gamma * next_value * next_non_terminal - mb_values_i[t]
                mb_advs_i[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            mb_returns_i = mb_advs_i + mb_values_i
            obs2ret.append(mb_obs_i)
            action2ret.append(mb_actions_i)
            value2ret.append(mb_values_i)
            done2ret.append(mb_dones_i)
            neglogp2ret.append(mb_neg_log_p_i)
            return2ret.append(mb_returns_i)
        obs2ret = np.concatenate(obs2ret, axis=0)
        action2ret = np.concatenate(action2ret, axis=0)
        value2ret = np.concatenate(value2ret, axis=0)
        done2ret = np.concatenate(done2ret, axis=0)
        neglogp2ret = np.concatenate(neglogp2ret, axis=0)
        return2ret = np.concatenate(return2ret, axis=0)
        self.rec_rewards.append(sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])

        output = list(map(swap_and_flatten, (obs2ret, return2ret, done2ret, action2ret, value2ret, neglogp2ret)))
        output.append(sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])
        return output

    @staticmethod
    def reconnect_array(obs: BaseObservation) -> np.ndarray:
        """ Method to reconnect lines within the environment

        Args:
            obs: observation of Grid2Op

        Returns: an array, containing the new line status

        """
        new_line_status_array = np.zeros_like(obs.rho, dtype=np.int)
        disconnected_lines = np.where(obs.line_status is False)[0]
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                line_to_reconnect = line  # reconnection
                new_line_status_array[line_to_reconnect] = 1
                break
        return new_line_status_array
