import random
from pathlib import Path

import grid2op
import numpy as np
import pytest
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend

from curriculumagent.common.utilities import (
    extract_action_set_from_actions,
    split_and_execute_action,
    is_legal,
    split_action_and_return,
    find_best_line_to_reconnect,
    simulate_action, revert_topo, map_actions, change_bus_from_topo_vect, set_bus_from_topo_vect, check_convergence,
)
from curriculumagent.teacher.submodule.common import affected_substations

class TestExecuteAction:
    """
    Test suite for the combined execute action
    """

    def test_extract_action_set_from_actions_single(self, test_action_set, test_env):
        """
        Testing the single action
        """
        single_a, _, _ = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space, action_vect=single_a)

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], grid2op.Action.BaseAction)

        # Execute action
        _, _, _, info = test_env.step(out[0])
        assert info["is_illegal"] is False

    def test_extract_action_set_from_actions_tuple(self, test_action_set, test_env):
        """
        Testing the single action
        """
        _, tuple_a, _ = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space, action_vect=tuple_a)

        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], grid2op.Action.BaseAction)
        assert isinstance(out[1], grid2op.Action.BaseAction)

        # Execute action

        # First the original, which should be illegal:
        _, _, _, info = test_env.step(test_env.action_space.from_vect(tuple_a))
        assert info["is_illegal"]
        _, _, _, info = test_env.step(out[0])
        assert info["is_illegal"] is False
        _, _, _, info = test_env.step(out[1])
        assert info["is_illegal"] is False

    def test_extract_action_set_from_actions_tripple(self, test_action_set, test_env):
        """
        Testing the single action
        """
        _, _, tripple = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space, action_vect=tripple)

        assert isinstance(out, list)
        assert len(out) == 3
        for o in out:
            assert isinstance(o, grid2op.Action.BaseAction)
        # Execute action

        # First the original, which should be illegal:
        _, _, _, info = test_env.step(test_env.action_space.from_vect(tripple))
        assert info["is_illegal"]
        for o in out:
            _, _, _, info = test_env.step(o)
        assert info["is_illegal"] is False

    def test_extract_action_set_dn(self,  test_env):
        """
        Testing the single action
        """
        d_n = test_env.action_space({})
        d_n_vect = d_n.to_vect()

        out = extract_action_set_from_actions(action_space=test_env.action_space, action_vect=d_n_vect)

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], grid2op.Action.BaseAction)
        assert d_n == out[0]


    def test_execute_action_single(self, test_env, test_action_set):
        """
        Testing the execute action set.
        """
        single_a, _, _ = test_action_set

        # Execute all three types:
        test_env.set_id(1)
        test_env.reset()
        obs = test_env.get_obs()
        assert pytest.approx(obs.rho.max()) == 0.9180667
        assert test_env.nb_time_step == 0

        obs, _, done, info = split_and_execute_action(env=test_env, action_vect=single_a)

        assert pytest.approx(obs.rho.max()) == 1.0084294080734253
        assert test_env.nb_time_step == 1
        assert done is False
        assert info["is_illegal"] is False

    def test_execute_action_tuple(self, test_env, test_action_set):
        """
        Testing the execute action set.
        """
        _, tuple_a, _ = test_action_set

        # Now two steps:
        test_env.set_id(1)
        test_env.reset()
        obs, _, done, info = split_and_execute_action(env=test_env, action_vect=tuple_a)

        assert pytest.approx(np.round(obs.rho.max(),4)) == 1.0546
        assert test_env.nb_time_step == 2
        assert done is False
        assert info["is_illegal"] is False

    def test_execute_action_tripple(self, test_env, test_action_set):
        """
        Testing whether the Do nothing action is returned
        """
        _, _, triple_a = test_action_set

        # Now two steps:
        test_env.set_id(1)
        test_env.reset()
        obs, _, done, info = split_and_execute_action(env=test_env, action_vect=triple_a)

        assert test_env.nb_time_step == 3
        assert done is False
        assert info["is_illegal"] is False


class TestLineReconnect:
    """Testing the line reconnect method"""

    def test_line_reconnect(self, test_env):
        """
        Check, whether line reconnect works
        """
        obs = test_env.get_obs()
        # Disconnect line:
        obs, rew, done, info = test_env.step(test_env.action_space({"set_line_status": [(5, -1)]}))
        # Two rounds do nothing
        dn = test_env.action_space({})
        assert sum(obs.time_before_cooldown_line) >= 3
        for _ in range(3):
            obs, rew, done, info = test_env.step(dn)

        # Now check, if line is reconnected
        assert sum(obs.time_before_cooldown_line) == 0
        action = find_best_line_to_reconnect(obs, dn)
        assert "set_line_status" in action.as_dict().keys()


class TestIsLegal:
    """
    Test suite for Best Line Reconnect and is legal.
    """

    def test_is_legal(self, test_action_set, test_env):
        """
        Testing whether a triple action is a valid input
        """
        _, _, tripple_a = test_action_set
        act = test_env.action_space.from_vect(tripple_a)
        assert is_legal(act, test_env.get_obs())

    def test_do_nothing(self, test_env):
        """
        Do nothing should be legal
        """
        assert is_legal(test_env.action_space({}), test_env.get_obs())

    def test_line_reconnect(self, test_env):
        """
        Do nothing should be legal
        """
        obs = test_env.get_obs()
        new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
        new_line_status_array[1] = -1
        action = test_env.action_space({"set_line_status": new_line_status_array})
        obs, _, _, _ = test_env.step(action)

        new_line_status_array[1] = 1

        action = test_env.action_space({"set_line_status": new_line_status_array})
        assert not is_legal(action, test_env.get_obs())

        obs = test_env.reset()
        assert obs.time_before_cooldown_line[1] == 0
        assert is_legal(action, test_env.get_obs())

    def test_converging_action(self, test_env_nonconverge):

        action = test_env_nonconverge.action_space({})
        test_env_nonconverge.seed(19)
        obs = test_env_nonconverge.reset()

        for i in range(164):
            obs, _, _, _ = test_env_nonconverge.step(action)

        assert is_legal(action, obs)

        simulator = obs.get_simulator()

        load_p_stressed = obs.load_p * 1.25
        gen_p_stressed = obs.gen_p * 1.25

        simulator_stressed = simulator.predict(act=action, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed)

        assert not simulator_stressed.converged
class TestCheckConvergence:
    """
    Test class for check_convergence function.
    """

    def test_converging(self, test_env_obs, sandbox_actions):
        """Test that specific converging action IDs lead to convergence."""
        env, obs = test_env_obs
        actions = np.load(sandbox_actions)
        # Converging action IDs based on your results
        converging_action_ids = [0, 12]  # Example: Test action IDs 0 and 12

        for action_id in converging_action_ids:
            # Use actions from sandbox_actions loaded from the .npy file
            action_vect = actions[action_id]
            action = env.action_space.from_vect(action_vect)

            # Check that the action leads to convergence
            assert check_convergence(action, obs), f"Action ID {action_id} should converge but did not."

    def test_non_converging(self, test_env_obs, sandbox_actions):
        """Test that specific non-converging action IDs do not lead to convergence."""
        env, obs = test_env_obs
        actions = np.load(sandbox_actions)
        # Non-converging action IDs based on your results
        non_converging_action_ids = [39, 125]  # Example: Test action IDs 39 and 125

        for action_id in non_converging_action_ids:
            # Use actions from sandbox_actions loaded from the .npy file
            action_vect = actions[action_id]
            action = env.action_space.from_vect(action_vect)

            # Check that the action does not lead to convergence
            assert not check_convergence(action, obs), f"Action ID {action_id} should not converge but did."

class TestSplitAction:
    """
    Test suite splitting combined actions similar to :class:`TextExecuteAction`
    """

    def test_split_do_nothing(self, test_paths_env):
        # Test Setup
        env_path, _ = test_paths_env
        backend = LightSimBackend()
        env: BaseEnv = grid2op.make(dataset=str(env_path), backend=backend)
        env.seed(42)
        np.random.seed(42)

        obs = env.reset()
        do_nothing = env.action_space({})
        actions = split_action_and_return(obs, env.action_space, do_nothing.to_vect())
        action = next(actions)
        assert (action == do_nothing) # "Splitting a do nothing should result in do nothing"
        with pytest.raises(StopIteration):
            action = next(actions)

    def test_split_generator(self, test_paths_env):
        # Test Setup
        env_path, _ = test_paths_env
        test_as_path = Path(__file__).parent.parent.parent / "tests" / "data" / "action_spaces"
        triple_actionspace_path = test_as_path / "test_tripple.npy"
        triple_actionspace = np.load(str(triple_actionspace_path))
        backend = LightSimBackend()
        env: BaseEnv = grid2op.make(dataset=str(env_path), backend=backend)
        env.seed(42)
        np.random.seed(42)

        # Given a triple action
        triple_action = triple_actionspace[0]
        env.action_space.from_vect(triple_action)

        # When splitting it up, all actions should be unitary
        obs = env.reset()
        collected_actions = []
        collected_ri = []
        collected_ri.append(obs.rho.max())
        for action in split_action_and_return(obs, env.action_space, triple_action):
            collected_actions.append(action)
            obs_, _, _, info = env.step(action)
            # collected_ri.append(obs.rho.max() - obs_.rho.max())
            collected_ri.append(obs_.rho.max())

            # Then the following assertions should hold
            assert not info["is_illegal"], "Returned action is legal"
            assert len(affected_substations(action)) == 1, "Returned action is unitary"

        assert len(collected_actions) == 3, "We actually got three actions out of a triple one"

        assert pytest.approx(collected_ri) == [0.918066680431366, 0.9049604535102844, 0.9032496809959412, 1.3328317403793335], (
            "Actions are executed best to " "worst"
        )


class TestClassSimulateAction:
    def test_simulate_action(self, test_env, test_action_set):
        """
        Test of the simulate action method.
        """
        single_a, _, tripple_a = test_action_set
        test_env.set_id(1)
        test_env.seed(1)
        obs = test_env.reset()

        while obs.rho.max() < 0.6:
            obs, _, _, _ = test_env.step(test_env.action_space({}))

        obs_rho, valid_action = simulate_action(action_space=test_env.action_space, obs=obs, action_vect=tripple_a)

        assert valid_action

        # First assert that the "normal" simulation of illegal values just does nothing:
        b = test_env.action_space.from_vect(tripple_a)
        b = find_best_line_to_reconnect(obs=obs, original_action=b)
        illegal_obs, _, _, _ = obs.simulate(b)
        obs_d_n, _, _, info = obs.simulate(test_env.action_space({}))
        assert np.round(illegal_obs.rho.max(), 4) == np.round(obs_d_n.rho.max(), 4)

        # Now check that this rho is different:
        assert obs_rho != illegal_obs.rho.max()


class TestRevertTopo:
    """
    Test suite for the revert topo agent
    """

    def test_revert_topo_rho_limit(self, test_env, test_sub_action):
        """
        We test two things:
        1. When the limit is low (0.8) then the reversion should not help (reversion ~0.85 rho)
        2. When the limit is about 0.9 we should be able to revert the grid
        """

        test_env.set_id(1)
        test_env.seed(1)
        test_env.reset()
        print(test_sub_action)
        obs, _, done, info = test_env.step(test_sub_action)
        # Check if the grid is stable and any bus was changed:
        assert np.any(obs.topo_vect == 2)

        # Wait some time to manipulate the grid again
        for _ in range(10):
            obs, _, done, info = test_env.step(test_env.action_space())
        assert done is False
        # now let's apply the revert topo
        action_array = revert_topo(test_env.action_space, obs)

        # Check if action is zero
        assert np.sum(action_array) == 0

        # Now let's rais the min rho
        action_array = revert_topo(test_env.action_space, obs, 0.9)
        assert np.sum(action_array) > 0

        topo_act = test_env.action_space.from_vect(action_array)
        obs, _, done, info = test_env.step(topo_act)
        # Now, this test should still be positive, because it returns to the rho limit
        assert np.all(obs.topo_vect == 1)


class Test_map_actions:
    """
    Testing the new mapping method
    """

    def test_map(self):
        """
        Testing whether the indexing works
        """
        a = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([[2, 1], [2, 2], [2, 3]])

        out = map_actions([a,b])
        assert isinstance(out,list)
        assert isinstance(out[0],dict)
        #test first
        for k in range(3):
            assert all(out[0][k] == a[k])

        # test second:
        for k in range(3, 6):
            assert all(out[1][k] == b[k - 3])


class TestGetFromTopoVect():
    "Testing the change_bus_from_topo_vect and set_bus_from_topo_vect"
    def test_change_bus_from_topo_vect(self,sandbox_env,sandbox_actions):
        """
        Testing the change bus
        """
        topo2 = np.array([1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                          2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        obs = sandbox_env.get_obs()

        # We take a random action
        a = change_bus_from_topo_vect(obs.topo_vect,topo2,sandbox_env.action_space)

        assert isinstance(a,grid2op.Action.BaseAction)
        assert a.as_dict()["change_bus_vect"]["modif_subs_id"] == ["1","4"]
        obs2,rew,done,info = split_and_execute_action(sandbox_env,a.to_vect())

        assert np.all(obs2.topo_vect==topo2)

    def test_set_bus_from_topo_vect(self,sandbox_env,sandbox_actions):
        """
        Testing the change bus
        """
        topo2 = np.array([1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                          2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        obs = sandbox_env.get_obs()

        # We take a random action
        a = set_bus_from_topo_vect(obs.topo_vect,topo2,sandbox_env.action_space)

        assert isinstance(a,grid2op.Action.BaseAction)
        assert a.as_dict()["set_bus_vect"]["modif_subs_id"] == ["1","4"]
        obs2,rew,done,info = split_and_execute_action(sandbox_env,a.to_vect())

        assert np.all(obs2.topo_vect==topo2)

    def test_both_together(self,sandbox_env,sandbox_actions):
        """
        Testing, whether we get the same results
        """

        topo3 = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        obs = sandbox_env.get_obs()

        # We take a random action
        a = set_bus_from_topo_vect(obs.topo_vect, topo3, sandbox_env.action_space)
        b = change_bus_from_topo_vect(obs.topo_vect, topo3, sandbox_env.action_space)

        # Should be different actions
        assert a!=b

        # But with same results (needs unitary action for this, else it is illegal):
        o1 = obs + a
        o2 = obs +b
        assert np.all(o1.topo_vect==o2.topo_vect)

    def test_change_bus_from_topo_vect_backwards_to_one(self, sandbox_env, sandbox_actions):
            """
            Testing the change bus
            """
            topo1 = np.array([1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                              2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            obs = sandbox_env.get_obs()
            topo2 = obs.topo_vect
            # We take a random action
            a = change_bus_from_topo_vect(topo1, topo2, sandbox_env.action_space)

            assert isinstance(a, grid2op.Action.BaseAction)
            assert a.as_dict()["change_bus_vect"]["modif_subs_id"] == ["1", "4"]


    def test_set_bus_from_topo_vect_back_to_one(self,sandbox_env,sandbox_actions):
        """
        Testing the change bus
        """
        topo1 = np.array([1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                          2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        obs = sandbox_env.get_obs()
        topo2 = obs.topo_vect
        # We take a random action
        a = set_bus_from_topo_vect(topo1,topo2,sandbox_env.action_space)

        assert isinstance(a,grid2op.Action.BaseAction)
        assert a.as_dict()["set_bus_vect"]["modif_subs_id"] == ["1","4"]
        obs2,rew,done,info = split_and_execute_action(sandbox_env,a.to_vect())

        assert np.all(obs2.topo_vect==topo2)