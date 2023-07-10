import os
from pathlib import Path

import pytest
import grid2op
from lightsim2grid import LightSimBackend
from curriculumagent.teacher.submodule.topology_action_search import topology_search_sequential_x_steps
from curriculumagent.teacher.teachers.teacher_sequential_actions import collect_sequential_experience


class TestTopologySequential:
    """
    Test class to collect the sequential steps of the teacher.
    """

    @pytest.mark.slow
    def test_tuple_actions(self, test_env, test_action_possibilities):
        """
        Testing, whether two tuple actions are selected
        """
        # For testing we exclude the substation 16 in order to speed up the results:
        sub_dict = {k: v for k, v in test_action_possibilities.items() if k != "16"}
        assert len(sub_dict.keys()) == 34
        test_env.set_id(1)
        obs = test_env.reset()
        done = False
        while obs.rho.max() < 0.9 and done == False:
            obs, _, done, _ = test_env.step(test_env.action_space({}))

        current_rho = obs.rho.max()
        # Now run for steps 2:
        action_set: grid2op.Action.BaseAction = topology_search_sequential_x_steps(
            env=test_env, sub_action_set=sub_dict, steps=2
        )

        assert action_set.as_dict()["set_bus_vect"]["modif_subs_id"] == ["23", "26"]
        new_rho = test_env.get_obs().rho.max()
        # Should be quite some improvement.
        assert current_rho - new_rho > 0

    @pytest.mark.slow
    def test_tripple_actions(self, test_env, test_action_possibilities):
        """
        Testing, whether two tuple actions are selected
        """
        # For testing we exclude the substation 16 in order to speed up the results:
        sub_dict = {k: v for k, v in test_action_possibilities.items() if k != "16"}
        test_env.set_id(1)
        obs = test_env.reset()
        done = False
        while obs.rho.max() < 0.9 and done == False:
            obs, _, done, _ = test_env.step(test_env.action_space({}))

        current_rho = obs.rho.max()
        # Now run for steps 3:
        action_set: grid2op.Action.BaseAction = topology_search_sequential_x_steps(
            env=test_env, sub_action_set=sub_dict, steps=3
        )
        assert action_set.as_dict()["set_bus_vect"]["modif_subs_id"] == ["12", "23", "26"]
        new_rho = test_env.get_obs().rho.max()
        # Should be quite some improvement.
        assert current_rho - new_rho > 0

    @pytest.mark.ultra_slow
    @pytest.mark.slow
    def test_tripple_actions(self, test_paths_env, test_action_possibilities):
        """
        Testing, whether two tuple actions are selected
        """
        # For testing we exclude the substation 16 in order to speed up the results:
        env_path, _ = test_paths_env
        sub_dict = {k: v for k, v in test_action_possibilities.items() if k != "16"}

        data_path = Path(__file__).parent.parent / "data" / "temporary_save"
        if not data_path.is_dir():
            data_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(data_path)) == 0

        collect_sequential_experience(
            save_path=data_path,
            env_name_path=env_path,
            chronics_path=str(env_path / "chronics"),
            steps=2,
            seed=42,
            rho_acting_threshold=0.9,
            actions_dict=sub_dict,
            chronic_limit=1,
        )
        path_res = data_path / "teacher_experience.csv"
        assert (path_res).is_file()

        if os.path.exists(path_res):
            os.remove(str(path_res))

        assert len(os.listdir(data_path)) == 0
