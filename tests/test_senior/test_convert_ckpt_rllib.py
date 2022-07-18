import os
import json
import pytest
import shutil


from curriculumagent.senior.rllib_execution.convert_rllib_ckpt import load_config, load_and_save_model, \
    collect_ckpt_from_ray_dir

# TODO: Overwriting of json file, to ensure correct path of env

class TestCollectionCheckpoints():
    """
    Test Suite to test the collection of checkpoints
    """

    def test_json_loader(self, test_path_rllib):
        """
        Testing, whether the json loader works.

        After executing, overwrite the path of the json file.
        """
        assert os.path.isdir(test_path_rllib)
        config,_ = load_config(test_path_rllib)
        assert isinstance(config, dict)
        assert config['num_sgd_iter'] == 4
        assert config['lr'] == 1e-4

# TODO: Add new Rllib file here:
    # def test_collect_ckpt_from_ray_dir_all(self, test_path_rllib, test_paths_env):
    #     """
    #     Testing the conversion method
    #     """
    #
    #     # Preparation:
    #     # First we have to overwrite the config to ensure the correct path, depending on the
    #     # test suite:
    #     config,_ = load_config(test_path_rllib)
    #     test_env_path, test_action_path = test_paths_env
    #     test_env_path.parent / "curriculumagent"/"action_space"
    #     config["env_config"] = {"action_space_path": test_action_path.as_posix(),
    #                             "env_path": test_env_path.as_posix(),
    #                             "action_threshold": 0.9,"filtered_obs":False,
    #                             "scaler": None}
    #
    #     config_path = os.path.join(test_path_rllib, "params.json")
    #     with open(config_path, "w") as write_file:
    #         json.dump(config, write_file)
    #
    #     # Now first test, whether checkpoints are in testdata:
    #     assert os.path.isdir(os.path.join(test_path_rllib, "checkpoint_002300"))
    #     assert os.path.isdir(os.path.join(test_path_rllib, "checkpoint_002325"))
    #
    #     # Now set file, where to save the checkpoints:
    #     ckpt_1 = os.path.join(os.getcwd(), "ckpt_2300")
    #     ckpt_2 = os.path.join(os.getcwd(), "ckpt_2325")
    #
    #     # Check if it exist (it shouldn't)
    #     assert os.path.isdir(ckpt_1) is False
    #     assert os.path.isdir(ckpt_2) is False
    #
    #     # Now run the conversion:
    #     collect_ckpt_from_ray_dir(folder=test_path_rllib,
    #                               save_path=os.getcwd(),
    #                               ckpt_nr=None)
    #
    #     # Now it should exist
    #     assert os.path.isdir(ckpt_1)
    #     assert os.path.isdir(ckpt_2)
    #     # Checkpoints should be there as well
    #     assert os.path.isfile(os.path.join(ckpt_1, "saved_model.pb"))
    #     assert os.path.isfile(os.path.join(ckpt_2, "saved_model.pb"))
    #
    #     # Remove files:
    #     shutil.rmtree(ckpt_1, ignore_errors=True)
    #     shutil.rmtree(ckpt_2, ignore_errors=True)
    #
    #     # Check if it exist (it shouldn't)
    #     assert os.path.isdir(ckpt_1) is False
    #     assert os.path.isdir(ckpt_2) is False
    #
