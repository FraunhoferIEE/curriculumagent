import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float


class TopoRew(BaseReward):
    """This class executes a specific topological reward. The idea is that the agent receives a
    positive reward for each step and a "strong" positive reward, if the grid is in its original state.
    """

    def __init__(self, min_rew: float = 0.0, max_rew: float = 1.0, proportion: float = 0.5, logger=None):
        """The reward is calculated as following:
        If something illegal happened or game over is achieved, the agent receives the min_rew.
        If the agent survives the step, it receives the max_rew*proportion reward.
        If the agent survives the step and is also in the original topology, the agent receives max_rew.

        Args:
            min_rew: Minimal reward for the agent.
            max_rew: Maximal reward for the agent.
            proportion: Proportion between the reward and the topo reward. We multiply with the proportion
            logger: Logger.

        Returns:
            None.
        """
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(min_rew)
        self.reward_max = dt_float(max_rew)
        self.proportion = proportion

    def __call__(self, action: grid2op.Action.BaseAction,
                 env: grid2op.Environment.Environment,
                 has_error,
                 is_done,
                 is_illegal,
                 is_ambiguous) -> float:
        """Here the reward is called. Note that we differentiate between the three different
        is_done options. If is_done=False, then the agent receives the reward. If the agent completed
        the whole env, it also receives the max_rew. If the agent does not complete the env, thus
        is_done = True, it receives the min_rew.

        Args:
            action: Action of the agent.
            env: Grid2Op Environment.
            has_error: Does the Environment Return an Error.
            is_done: Is the Environment Done.
            is_illegal: Is the action illegal.
            is_ambiguous: Is the action of the agent ambiguous.

        Returns:
            Float.

        """
        # check for errors
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Check dones:
        if is_done:
            if env.nb_time_step == env.max_episode_duration():
                return self.reward_max
            else:
                return self.reward_min

        # Check topology:
        if all(env.current_obs.topo_vect == 1):
            return self.reward_max
        else:
            return self.reward_max * self.proportion


class PPO_Reward(BaseReward):
    def __init__(self):
        """
        PPO_Reward class, based on the BaseReward from Grid2Op
        """
        BaseReward.__init__(self)
        self.reward_min = -10
        self.reward_std = 2

    def __call__(
            self, action: BaseAction, env: Environment, has_error: bool, is_done: bool, is_illegal: bool,
            is_ambiguous: bool
    ) -> float:
        """Call of the Reward class. Here the BaseReward is overwritten.

        Compute the reward based on:

        If the action is valid, the reward is calculated as followed:
                2 - rho_max * (1 if rho_max < 0.95 else 2)

        Args:
            action: action of the environment in Grid2Op format
            env: Grid2Op Environment
            has_error: Has there been an error, for example a :class:`grid2op.DivergingPowerFlow` be thrown when the
                action has been implemented in the environment.
            is_done: Is the episode over (either because the agent has reached the end, or because
                there has been a game over)
            is_illegal: Has the action submitted by the BaseAgent raised an
             :class:`grid2op.Exceptions.IllegalAction` exception. In this case it has been replaced by "do nohting" by
             the environment. **NB** an illegal action is NOT an ambiguous action. See the description of the Action
             module: :ref:`Illegal-vs-Ambiguous` for more details. is_ambiguous:  Has the action submitted by the
             BaseAgent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception. In this case it has been
              replaced by "do nothing" by the environment. **NB** an illegal action is NOT an ambiguous action.
              See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.

        Returns: Return, based on the formular

        """
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2)


class RhoReward(BaseReward):
    def __init__(self, max_steps: int = 8064, complex:bool = False):
        """
        Rho Reward based on BaseReward from Grid2Op

        If complex is set to True, than we also give a reward for staying in the starting grid and
        not changing to many actions !
        """
        BaseReward.__init__(self)
        self.reward_min = -20
        self.reward_max = 10
        self.max_steps = max_steps
        self.complex = complex
        self.prev_action = None

    def __call__(
            self, action: BaseAction, env: Environment, has_error: bool, is_done: bool, is_illegal: bool,
            is_ambiguous: bool
    ) -> float:
        """Call of the Reward class. Here the BaseReward is overwritten.

        We look at the rho values: If below 1 then we give the reward 1-rho.
        If above 1.0, we return 1-sum((rho>1)^2). The quadratic value is there to penalize high rhos!

        Args:
            action: action of the environment in Grid2Op format
            env: Grid2Op Environment
            has_error: Has there been an error, for example a :class:`grid2op.DivergingPowerFlow` be thrown when the
                action has been implemented in the environment.
            is_done: Is the episode over (either because the agent has reached the end, or because
                there has been a game over)
            is_illegal: Has the action submitted by the BaseAgent raised an
             :class:`grid2op.Exceptions.IllegalAction` exception. In this case it has been replaced by "do nohting" by
             the environment. **NB** an illegal action is NOT an ambiguous action. See the description of the Action
             module: :ref:`Illegal-vs-Ambiguous` for more details. is_ambiguous:  Has the action submitted by the
             BaseAgent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception. In this case it has been
              replaced by "do nothing" by the environment. **NB** an illegal action is NOT an ambiguous action.
              See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.

        Returns: Return, based on the formular

        """
        reward = 0
        if is_done:
            if env.nb_time_step == self.max_steps:
                # Terminated:
                reward += self.reward_max
            else:
                # Truncated:
                reward = self.reward_min

            self.prev_action = None

        # Illegal action:
        if is_illegal or is_ambiguous or has_error:
            reward = self.reward_min

        obs = env.get_obs()
        topo_vect = obs.topo_vect

        if self.complex:
            # Add a small bonus, when action remains the same:
            if self.prev_action == action:
                reward +=0.1
            self.prev_action = action

            # Add a larger bonus if grid in original state:
            if np.all(obs.topo_vect == np.ones(len(
                obs.topo_vect))):
                reward += 0.2

        # Now rho value:

        rho = obs.rho

        if rho.max() < 1.0:
            reward += 1 - rho.max()

        else:
            reward += - np.sum(rho[rho >= 1.0] ** 2)

        return reward
