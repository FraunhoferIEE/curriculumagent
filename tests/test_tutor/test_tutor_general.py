import os
import random
from pathlib import Path

import numpy as np

from curriculumagent.tutor.tutors.general_tutor import generate_tutor_experience, GeneralTutor, \
    collect_tutor_experience_one_chronic


class TestTutorGeneral:
    """
    Test suite
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

        if not save_path.is_dir():
            os.mkdir(save_path)

        files = os.listdir(save_path)
        if len(files) > 0:
            for file in files:
                os.remove(str(save_path / file))

        assert not os.listdir(save_path)
        generate_tutor_experience(env_name_path=data_path,
                                  action_paths=[single_path],
                                  save_path=save_path,
                                  num_chronics=1,
                                  num_sample=None,
                                  seed=42)

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

        if not save_path.is_dir():
            os.mkdir(save_path)

        files = os.listdir(save_path)
        if len(files) > 0:
            for file in files:
                os.remove(str(save_path / file))

        assert not os.listdir(save_path)
        print(single_path)
        generate_tutor_experience(env_name_path=data_path,
                                  action_paths=[tripple_path, single_path],
                                  save_path=save_path,
                                  num_chronics=2,
                                  num_sample=None,
                                  seed=42)

        file_name = os.listdir(save_path)[0]
        assert "tutor_experience" in file_name
        result = np.load(save_path / file_name)
        # Check whether the first data set was correctly selected

        assert result[2, 0] == 0.0

        # Check whether the second data set was correctly selected
        assert result[-4, 0] == 17.0
        assert result[1, 0] == 16.0

        if os.path.exists(save_path / file_name):
            os.remove(str(save_path / file_name))

        assert not os.listdir(save_path)

    def test_act_with_id_tutor(self, test_env, test_action_paths):
        """
        Testing whether or not the actions are correctly appended
        """
        test_env.seed(42)
        single_path, _, tripple_path = test_action_paths

        tutor = GeneralTutor(action_space=test_env.action_space,
                             action_space_file=single_path,
                             do_nothing_threshold=0.9,
                             best_action_threshold=0.95,
                             return_status=True)

        assert len(tutor.actions) == 10

        tutor = GeneralTutor(action_space=test_env.action_space,
                             action_space_file=[tripple_path, single_path],
                             do_nothing_threshold=0.9,
                             best_action_threshold=0.95,
                             return_status=True)

        assert len(tutor.actions) == 20

    def test_act_with_id_tutor(self, test_env, test_action_paths):
        """
        Testing the act function of the tutor
        """
        test_env.seed(42)
        single_path, _, tripple_path = test_action_paths

        tutor = GeneralTutor(action_space=test_env.action_space,
                             action_space_file=[single_path, tripple_path],
                             do_nothing_threshold=0.9,
                             best_action_threshold=0.95,
                             return_status=True)

        action_set = np.load(single_path)

        assert len(tutor.actions) == 20

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

        out = collect_tutor_experience_one_chronic(action_paths=[single_path, tripple_path],
                                                   chronics_id=1,
                                                   env_name_path=test_env_p,
                                                   seed=42)
        assert out[1, 0] == 6.0
        assert out[-1, 0] == 2.0
