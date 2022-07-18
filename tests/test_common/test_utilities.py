from pathlib import Path

import grid2op
import numpy as np
import pytest
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend

from curriculumagent.common.utilities import array2action, get_from_dict_set_bus, extract_action_set_from_actions, \
    split_and_execute_action, is_legal, split_action_and_return, find_best_line_to_reconnect
from curriculumagent.teacher.submodule.common import affected_substations


class TestArray2Action:
    """Test Suite for array2action method"""

    def test_array2action(self, test_env):
        """
        Testing that a do nothing action is returned if np.zeros are inputed
        """
        empty_action = np.zeros(len(test_env.action_space({}).to_vect()))
        converted_action = array2action(test_env.action_space, empty_action)
        assert converted_action == test_env.action_space({})

    def test_array2action_lines(self, test_env):
        """
        Testing, whether the conversion works
        """
        line_action = np.zeros(len(test_env.action_space({}).to_vect()))
        line_action[0] = -1
        converted_action = array2action(test_env.action_space, line_action)
        action_dict = converted_action.as_dict()
        assert action_dict['set_line_status']['nb_connected'] == 0
        assert action_dict['set_line_status']['nb_disconnected'] == 1

    def test_array2action_sub(self, test_env, test_action_set):
        """
        Testing the substations action
        """
        single_a, _, _ = test_action_set
        converted_action = array2action(test_env.action_space, single_a[0]).as_dict()
        assert converted_action['set_bus_vect']['nb_modif_objects'] == 3
        assert converted_action['set_bus_vect']['modif_subs_id'] == ["11"]

    def test_array2action_sub_plus_reconnect(self, test_env, test_action_set):
        """
        Testing the substations action plus reconnect lines

        Due to the fact that all lines are still connected, we instead disconnect a line

        """
        single_a, _, _ = test_action_set
        converted_action = array2action(test_env.action_space, single_a[0]).as_dict()
        assert converted_action['set_bus_vect']['nb_modif_objects'] == 3
        assert converted_action['set_bus_vect']['modif_subs_id'] == ["11"]
        assert 'set_line_status' not in converted_action.keys()

        # Add reconnect lines
        line_action = np.zeros(len(test_env.get_obs().rho), dtype=int)
        line_action[0] = -1
        converted_action = array2action(test_env.action_space, single_a[0], line_action).as_dict()
        assert converted_action['set_bus_vect']['nb_modif_objects'] == 3
        assert converted_action['set_bus_vect']['modif_subs_id'] == ["11"]
        assert converted_action['set_line_status']['nb_connected'] == 0
        assert converted_action['set_line_status']['nb_disconnected'] == 1

    def test_array2action_zeros(self, test_env):
        """
        Testing, what happens if empty array is supplied
        """
        converted_action = array2action(test_env.action_space, np.zeros(494, dtype=int))
        assert converted_action == test_env.action_space({})


class TestExecuteAction:
    """
    Test suite for the combined execute action
    """

    def test_get_from_dict_set_bus_tuple(self):
        """
        Test for the get_from_dict
        """
        example_dict = {'3': {'type': 'load', 'new_bus': 1},
                        '4': {'type': 'line (origin)', 'new_bus': 1},
                        '12': {'type': 'line (origin)', 'new_bus': 1},
                        '0': {'type': 'generator', 'new_bus': 1},
                        '2': {'type': 'load', 'new_bus': 1}}

        out = get_from_dict_set_bus(original=example_dict)

        assert list(out.keys()) == ["set_bus"]
        for key in out["set_bus"]:
            assert key in ['lines_or_id', 'lines_ex_id', 'loads_id', 'generators_id']

        assert out["set_bus"]["lines_or_id"][0] == (4, 1)
        assert out["set_bus"]["lines_or_id"][1] == (12, 1)

    def test_extract_action_set_from_actions_single(self, test_action_set, test_env):
        """
        Testing the single action
        """
        single_a, _, _ = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space,
                                              action=single_a[0])

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], grid2op.Action.BaseAction)

        # Execute action
        _, _, _, info = test_env.step(out[0])
        assert info['is_illegal'] is False

    def test_extract_action_set_from_actions_tuple(self, test_action_set, test_env):
        """
        Testing the single action
        """
        _, tuple_a, _ = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space,
                                              action=tuple_a[0])

        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], grid2op.Action.BaseAction)
        assert isinstance(out[1], grid2op.Action.BaseAction)

        # Execute action

        # First the original, which should be illegal:
        _, _, _, info = test_env.step(array2action(test_env.action_space, tuple_a[0]))
        assert info['is_illegal']
        _, _, _, info = test_env.step(out[0])
        assert info['is_illegal'] is False
        _, _, _, info = test_env.step(out[1])
        assert info['is_illegal'] is False

    def test_extract_action_set_from_actions_tripple(self, test_action_set, test_env):
        """
        Testing the single action
        """
        _, _, tripple = test_action_set

        out = extract_action_set_from_actions(action_space=test_env.action_space,
                                              action=tripple[0])

        assert isinstance(out, list)
        assert len(out) == 3
        for o in out:
            assert isinstance(o, grid2op.Action.BaseAction)
        # Execute action

        # First the original, which should be illegal:
        _, _, _, info = test_env.step(array2action(test_env.action_space, tripple[0]))
        assert info['is_illegal']
        for o in out:
            _, _, _, info = test_env.step(o)
        assert info['is_illegal'] is False

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

        obs, _, done, info = split_and_execute_action(env=test_env,
                                                      action=single_a[0])

        assert pytest.approx(obs.rho.max()) == 0.9167755
        assert test_env.nb_time_step == 1
        assert done is False
        assert info['is_illegal'] is False

    def test_execute_action_tuple(self, test_env, test_action_set):
        """
        Testing the execute action set.
        """
        _, tuple_a, _ = test_action_set

        # Now two steps:
        test_env.set_id(1)
        test_env.reset()
        obs, _, done, info = split_and_execute_action(env=test_env,
                                                      action=tuple_a[0])

        assert pytest.approx(obs.rho.max()) == 0.9020909
        assert test_env.nb_time_step == 2
        assert done is False
        assert info['is_illegal'] is False

    def test_execute_action_tripple(self, test_env, test_action_set):
        """
        Testing the execute action set.
        """
        _, _, triple_a = test_action_set

        # Now two steps:
        test_env.set_id(1)
        test_env.reset()
        obs, _, done, info = split_and_execute_action(env=test_env,
                                                      action=triple_a[0])

        assert test_env.nb_time_step == 3
        assert done is False
        assert info['is_illegal'] is False


class TestLineReconnectAndIsLegal():
    """
    Test suite for Best Line Reconnect and is legal.
    """

    def test_is_legal(self, test_action_set, test_env):
        """
        Testing whether a triple action is a valid input
        """
        _, _, tripple_a = test_action_set
        act = array2action(test_env.action_space, tripple_a[0])
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
        assert (action.to_vect() == do_nothing.to_vect()).all(), "Splitting a do nothing should result in do nothing"
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
        array2action(env.action_space, triple_action)

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
            assert not info['is_illegal'], "Returned action is legal"
            assert len(affected_substations(action)) == 1, "Returned action is unitary"

        assert len(collected_actions) == 3, "We actually got three actions out of a triple one"
        # TODO: This fails at the moment, fix or remove this assumption!

        assert pytest.approx(collected_ri) == [0.91806668,
                                               0.90394872,
                                               0.90209091,
                                               1.33232236], "Actions are executed best to " \
                                                            "worst"
