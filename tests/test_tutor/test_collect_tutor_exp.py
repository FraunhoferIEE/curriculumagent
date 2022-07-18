import os
import random
import shutil
from pathlib import Path

import numpy as np

from curriculumagent.tutor.collect_tutor_experience import prepare_dataset


class TestCreateJuniorData:
    """
    Test Suite of the create Junior data
    """

    def test_prepare_dataset(self, test_action_paths):
        """
        Testing the prepare dataset method.
        """
        single_path, tuple_path, tripple_path = test_action_paths
        test_data_path = Path(__file__).parent.parent / "data" / "tutor_data"
        save_path = test_data_path / "temporary_save"

        if not save_path.is_dir():
            os.mkdir(save_path)

        assert (test_data_path / "tutor_experience_tutor_test.npy").is_file()
        prepare_dataset(traindata_path=test_data_path,
                        target_path=save_path,
                        dataset_name="test",
                        extend=False, seed=42,
                        filtered_obs=True)

        files = os.listdir(save_path)
        for file in files:
            assert file in ["test_test.npz", "test_train.npz", "test_val.npz", "test_statistics.txt"]

        # Load Train_data:
        train_data = np.load(save_path / "test_train.npz")
        assert np.array_equal(train_data['a_train'], np.array([[0], [0], [17], [16], [16]]))

        test_data = np.load(save_path / "test_test.npz")
        val_data = np.load(save_path / "test_val.npz")
        assert len(train_data["s_train"]) + len(val_data["s_validate"]) + \
               len(test_data["s_test"]) == 12
        assert test_data["a_test"][-1] == [18]

        del train_data, test_data, val_data
        # Delete full temporary_save file and then create it again
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        assert not os.listdir(save_path)

    def test_filtered_obs(self, test_action_paths):
        """
        Testing whether the filtered_obs converts the actions correctly.
        Similar to the test for the SeniorEnvRllib
        """
        single_path, tuple_path, tripple_path = test_action_paths
        test_data_path = Path(__file__).parent.parent / "data" / "tutor_data"
        save_path = test_data_path / "temporary_save"

        if not save_path.is_dir():
            os.mkdir(save_path)

        assert (test_data_path / "tutor_experience_tutor_test.npy").is_file()

        # Now run the prepare dataset:
        prepare_dataset(traindata_path=test_data_path,
                        target_path=save_path,
                        dataset_name="test",
                        extend=False, seed=42,
                        filtered_obs=True)

        # Load file and check size of columns:
        train_data = np.load(save_path / "test_train.npz")
        assert train_data["s_train"].shape == (5, 1221)

        prepare_dataset(traindata_path=test_data_path,
                        target_path=save_path,
                        dataset_name="test",
                        extend=False, seed=42,
                        filtered_obs=False)
        train_data = np.load(save_path / "test_train.npz")
        assert train_data["s_train"].shape == (5, 1362)

        # Now run the prepare dataset:
        prepare_dataset(traindata_path=test_data_path,
                        target_path=save_path,
                        dataset_name="test",
                        extend=False, seed=42,
                        filtered_obs=list(range(1, 100))
                        )
        train_data = np.load(save_path / "test_train.npz")
        assert train_data["s_train"].shape == (5, 98)

        del train_data

        shutil.rmtree(save_path)
        os.mkdir(save_path)
        assert not os.listdir(save_path)
