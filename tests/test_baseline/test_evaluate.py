import os
import shutil

import grid2op
import numpy as np
import pytest
import tensorflow as tf
from lightsim2grid import LightSimBackend


class TestEvaluateBaseline:
    """ Testing the evaluation"""

    @pytest.mark.slow
    def test_evaluate(self, test_baseline_models, test_temp_save):
        """
        Testing, whether the evaluate methode works
        """
        from curriculumagent.baseline import evaluate
        senior_path, _ = test_baseline_models
        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()

        # Clear as usual:
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        assert test_temp_save.is_dir() and len(os.listdir(test_temp_save)) == 0

        log_paths = test_temp_save / "logs"

        np.random.seed(42)
        tf.random.set_seed(42)
        out = evaluate(
            env,
            load_path=senior_path,
            logs_path=log_paths,
            nb_episode=2,
            nb_process=1,
            max_steps=100,
            verbose=0,
            save_gif=False,
        )

        assert log_paths.is_dir()
        assert (log_paths / "dict_action_space.json").is_file()

        # Now lastly let's delete the model:
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
