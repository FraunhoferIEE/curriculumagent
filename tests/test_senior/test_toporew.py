import grid2op
import numpy as np
from lightsim2grid import LightSimBackend

from curriculumagent.senior.rllib_execution.alternative_rewards import TopoRew


class TestTopoRew:
    """
    Testing the topo reward
    """

    def test_init(self, test_paths_env):
        """
        Testing, whether we can add the reward
        Args:
            test_paths_env:

        Returns:

        """
        backend = LightSimBackend()

        env_path, _ = test_paths_env
        test_env = grid2op.make(dataset=env_path, backend=backend,
                                reward_class=TopoRew, test=True)

        assert test_env.reward_range == (0,1)
        assert isinstance(test_env.get_reward_instance(),TopoRew)

    def test_run_agent(self, test_paths_env, test_sub_action):
        """
        Test the different returns

        Args:
            test_paths_env:

        Returns:

        """
        backend = LightSimBackend()

        env_path, _ = test_paths_env
        test_env = grid2op.make(dataset=env_path, backend=backend,
                                reward_class=TopoRew, test=True)

        # Do not change topology:
        obs,rew,done,info = test_env.step(test_env.action_space())
        assert rew == 1.0
        # Change topology:

        obs,rew , done, info = test_env.step(test_sub_action)
        # Check if the grid is stable and any bus was changed:
        assert np.any(obs.topo_vect == 2)
        assert rew == 0.5

