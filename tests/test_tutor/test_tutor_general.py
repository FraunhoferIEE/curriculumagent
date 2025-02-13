import os
import random
import shutil
from pathlib import Path

import numpy as np

from curriculumagent.tutor.tutors.general_tutor import GeneralTutor
from curriculumagent.tutor.collect_tutor_experience import (
    collect_tutor_experience_one_chronic,
    generate_tutor_experience,
)


class TestTutorGeneral:
    """
    Test suite for General Tutor
    """

    def test_overall_run_one_action(self, test_paths_env, test_action_paths):
        """
        Testing, whether the tutor works
        """
        random.seed(42)
        data_path, _ = test_paths_env
        test_data_path = Path(__file__).parent.parent / "data"
        single_path, _, _ = test_action_paths

        save_path = test_data_path / "temporary_save"

        if save_path.is_dir():
            if len(os.listdir(save_path)) > 0:
                shutil.rmtree(save_path, ignore_errors=True)

        if not save_path.is_dir():
            os.mkdir(save_path)


        assert not os.listdir(save_path)
        generate_tutor_experience(
            env_name_path=data_path,
            action_paths=[single_path],
            save_path=save_path,
            num_chronics=1,
            num_sample=None,
            seed=42,
        )

        file_name = os.listdir(save_path)[0]
        assert "tutor_experience" in file_name
        result = np.load(save_path / file_name)
        # First and last choice is (6)
        assert result[1, 0] == 6.0
        assert result[-1, 0] == 6.0

        if os.path.exists(save_path / file_name):
            os.remove(str(save_path / file_name))

        assert not os.listdir(save_path)

    def test_overall_run_tripple(self, test_paths_env, test_action_paths):
        """
        Testing, whether the tutor works
        """
        random.seed(42)
        data_path, _ = test_paths_env
        test_data_path = Path(__file__).parent.parent / "data"
        single_path, _, tripple_path = test_action_paths

        save_path = test_data_path / "temporary_save"

        if save_path.is_dir():
            if len(os.listdir(save_path)) > 0:
                shutil.rmtree(save_path, ignore_errors=True)

        if not save_path.is_dir():
            os.mkdir(save_path)

        assert not os.listdir(save_path)
        print(single_path)
        generate_tutor_experience(
            env_name_path=data_path,
            action_paths=[single_path, tripple_path],
            save_path=save_path,
            num_chronics=2,
            num_sample=None,
            seed=42,
        )

        file_name = os.listdir(save_path)[0]
        assert "tutor_experience" in file_name
        result = np.load(save_path / file_name)
        # Check whether the first data set was correctly selected

        assert result[2, 0] == 0.0

        if os.path.exists(save_path / file_name):
            os.remove(str(save_path / file_name))

        assert not os.listdir(save_path)

    def test_act_with_id_tutor(self, test_env, test_action_paths):
        """
        Testing whether the actions are correctly appended
        """
        test_env.seed(42)
        single_path, _, tripple_path = test_action_paths

        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=single_path,
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )

        # List of 20 actions
        assert len(tutor.actions) == 20
        assert isinstance(tutor.actions, list)

        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=[tripple_path, single_path],
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )

        assert len(tutor.actions) == 2
        assert len(tutor.actions[0]) == 20
        assert len(tutor.actions[1]) == 20

    def test_act_with_id_tutor(self, test_env, test_action_paths):
        """
        Testing the act function of the tutor
        """
        test_env.seed(42)
        single_path, _, tripple_path = test_action_paths

        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path, tripple_path],
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )

        action_set = np.load(single_path)

        assert len(tutor.actions) == 2

        obs = test_env.get_obs()
        out, idx = tutor.act_with_id(obs)
        assert idx == 6
        assert np.array_equal(out, action_set[6, :])

    def test_collect_tutor_one_chronic(self, test_paths_env, test_action_paths):
        """
        Testing the single chronic
        """
        random.seed(42)
        test_env_p, _ = test_paths_env
        single_path, _, tripple_path = test_action_paths

        out = collect_tutor_experience_one_chronic(
            action_paths=[single_path, tripple_path], chronics_id=1, env_name_path=test_env_p, seed=42
        )
        assert out[1, 1] == 2012
        assert out.shape == (13, 1430)

    def test_multiple_actions_id_selections(self, test_env, test_action_paths):
        """
        Testing whether the idx is correctly passed to the actions.
        """
        test_env.seed(42)
        single_path, _, tripple_path = test_action_paths

        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path, tripple_path],
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )
        # General stuff
        assert isinstance(tutor.actions, list)
        assert isinstance(tutor.actions[0], dict)
        assert isinstance(tutor.actions[1], dict)
        assert len(tutor.actions) == 2

        for i in range(10):
            assert isinstance(tutor.actions[0][i], np.ndarray)
        for i in range(10, 20):
            assert isinstance(tutor.actions[1][i], np.ndarray)

    def test_test_actions_idx(self, test_env, test_action_single):
        """
        Testing whether the idx is correctly passed to the actions.
        """
        single_path1, single_path2 = test_action_single

        np.random.seed(42)  #
        random.seed(42)
        test_env.seed(42)
        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path1, single_path2],
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )
        test_env.set_id(1)
        obs = test_env.reset()
        while obs.rho.max() <= 1.0:
            obs, _, _, _ = test_env.step(test_env.action_space({}))

        array_out, idx = tutor.act_with_id(observation=obs)
        #select from second action space
        assert idx == 6

        ##############################################
        # Now switch the order and check which id we get
        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path2, single_path1],
            do_nothing_threshold=0.9,
            best_action_threshold=0.95,
            return_status=True,
        )
        test_env.set_id(1)
        np.random.seed(42)  #
        random.seed(42)
        test_env.seed(42)
        obs = test_env.reset()
        while obs.rho.max() <= 1.0:
            obs, _, _, _ = test_env.step(test_env.action_space({}))

        array_out, idx = tutor.act_with_id(observation=obs)
        # Select from first action space.
        assert idx == 1
