import os
from pathlib import Path

import grid2op
import pytest

from curriculumagent.teacher.teachers.teacher_n_minus_1 import NMinusOneTeacher


class TestN1Agent:
    """
    Test Suite for the N-1 Agent.
    """

    def test_check_init(self):
        """
        Testing the init
        """
        agent = NMinusOneTeacher()
        assert agent.rho_n0 == 0.9
        assert agent.rho_threshold == 1.0
        assert agent.lines_to_attack == [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]

        agent = NMinusOneTeacher(lines_to_attack=[42], rho_n0_threshold=0.8, rho_max_threshold=2.0)
        assert agent.rho_n0 == 0.8
        assert agent.rho_threshold == 2.0
        assert agent.lines_to_attack == [42]

    def test_do_nothing_lines(self, test_env):
        """
        Testing, whether the do nothing and n-1 action works

        This means:
        1. Check whether the one line is found and disconnected.
        2. Check whether the iteration continues
        3. Check whether the do_nothing_and_run returns a do nothing action if rho>1.0
        """
        agent = NMinusOneTeacher(lines_to_attack=[16])
        assert isinstance(test_env, grid2op.Environment.BaseEnv)

        test_env.set_id(1)
        test_env.reset()

        obs = test_env.get_obs()
        action = agent.do_nothing_and_run_through_lines_action(env=test_env, obs=obs)
        assert isinstance(action, grid2op.Action.BaseAction)
        act_json = action.to_json()
        assert all([a == 0 for a in act_json["_set_topo_vect"]])
        print(act_json["_set_line_status"])
        assert any([a == -1 for a in act_json["_set_line_status"]])
        assert act_json["_set_line_status"][16] == -1
        # Now run through do nothing for 35 steps:
        steps = 0
        while obs.rho.max() < 1.0:
            action = agent.do_nothing_and_run_through_lines_action(env=test_env, obs=obs)
            obs, _, _, _ = test_env.step(action)
            steps += 1
        assert steps > 1
        assert obs.rho.max() >= 1.0

        # Check whether any line gets disconnected after rho>1.0
        action = agent.do_nothing_and_run_through_lines_action(env=test_env, obs=obs)
        act_json = action.to_json()
        assert not any([a == -1 for a in act_json["_set_line_status"]])

    def test_run_through_lines(self, test_env):
        """
        Testing wether the maximum rho is returned.
        """

        agent = NMinusOneTeacher(lines_to_attack=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39])

        test_env.set_id(1)
        test_env.reset()
        obs = test_env.get_obs()

        out = agent.calculate_attacked_max_rho(obs=obs, action=test_env.action_space({}))
        assert pytest.approx(out) == 2.1300917

    @pytest.mark.slow
    def test_search_best_n_minus_one_action(self, test_env):
        agent = NMinusOneTeacher(lines_to_attack=[16])
        assert isinstance(test_env, grid2op.Environment.BaseEnv)

        test_env.set_id(1)
        test_env.reset()
        obs = test_env.get_obs()

        act, n_1_bool = agent.search_best_n_minus_one_action(env=test_env, obs=obs)
        assert n_1_bool
        assert isinstance(act, grid2op.Action.BaseAction)

        # Check the result:
        last_rho = obs.rho.max()
        obs, _, done, _ = test_env.step(act)
        print(last_rho, obs.rho.max())
        assert not done
        assert last_rho > obs.rho.max()

    @pytest.mark.ultra_slow
    @pytest.mark.slow
    def test_n_minus_one_agent(self, test_paths_env):
        """
        Testing, whether the n_minus_one_agent returns some variables
        """
        env_path, _ = test_paths_env
        data_path = str(env_path)
        scenario_path = str(env_path / "chronics")
        path_variable = Path(__file__).parent.parent / "data" / "test_n_minus_1.csv"
        print(path_variable)
        assert not path_variable.is_file()

        agent = NMinusOneTeacher(lines_to_attack=[16])

        agent.n_minus_one_agent(
            env_path=data_path,
            chronics_path=scenario_path,
            chronics_id=1,
            save_path=path_variable,
            save_greedy=False,
            active_search=True,
        )

        assert path_variable.is_file()

        if os.path.exists(path_variable):
            os.remove(str(path_variable))
