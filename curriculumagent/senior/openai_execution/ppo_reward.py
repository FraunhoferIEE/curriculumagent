"""
In this file, the original PPO reward of @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
is defined.
"""
from grid2op.Reward.BaseReward import BaseReward
from grid2op.Environment import Environment
from grid2op.Action import BaseAction

class PPO_Reward(BaseReward):
    def __init__(self):
        """
        PPO_Reward class, based on the BaseReward from Grid2Op
        """
        BaseReward.__init__(self)
        self.reward_min = -10
        self.reward_std = 2

    def __call__(self, action: BaseAction, env: Environment, has_error: bool,
                 is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        """ Call of the Reward class. Here the BaseReward is overwritten.

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
            is_illegal: Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.IllegalAction` exception.
                In this case it has been replaced by "do nohting" by the environment. **NB** an illegal action is NOT
                an ambiguous action. See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.
            is_ambiguous:  Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception.
                In this case it has been replaced by "do nothing" by the environment. **NB** an illegal action is NOT
                an ambiguous action. See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.

        Returns: Return, based on the formular

        """
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2)
