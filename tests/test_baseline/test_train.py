import os
import shutil

import grid2op
import pytest
import ray
from lightsim2grid import LightSimBackend


class TestTrainBaseline:
    """
    Run Training
    """
    @pytest.mark.ultra_slow
    @pytest.mark.slow
    def test_run_training(self, test_temp_save):
        """
        Testing, whether the model can be saved and loaded completely.

        Not this does not work on Windows due to ray!
        """
        from curriculumagent.baseline import train
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        assert test_temp_save.is_dir() and len(os.listdir(test_temp_save)) == 0

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        train(env=env,
              name="Are you Serious? This will take Bonkers amount of time!",
              iterations=1,
              save_path=test_temp_save,
              max_actionspace_size = 10
              )

        # Check if files are within the directories
        for name in ["teacher", "tutor", "junior", "senior", "model", "actions"]:
            assert len(os.listdir(test_temp_save / name)) > 0

        # Assert specific files (e.g. the tutor file and the model of the agent)
        assert (test_temp_save / "teacher" / "general_teacher_experience.csv").is_file()
        assert (test_temp_save / "tutor" / "tutor_experience.npy").is_file()
        assert (test_temp_save / "actions" / "actions.npy").is_file()
        assert (test_temp_save / "junior" / "saved_model.pb").is_file()
        assert (test_temp_save / "model" / "saved_model.pb").is_file()

        # Now lastly let's delete the model:
        shutil.rmtree(test_temp_save, ignore_errors=True)
        test_temp_save.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(test_temp_save)) == 0

        ray.shutdown()
