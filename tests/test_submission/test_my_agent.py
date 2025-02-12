import os
import shutil
import types

import grid2op
import numpy as np
import pathlib
import pytest
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from curriculumagent.junior import Junior
from curriculumagent.submission.my_agent import MyAgent

import platform

# Workaround for cross-platform torch/lightning models usage
system = platform.system()
if system == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
elif system == "Linux":
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

class TestAdvancedAgent:
    """
    Test suite of the advanced submission agent
    """

    def test_init_ray24(self, test_env, test_submission_models, test_submission_model_torch,  test_paths_env):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env

        act_list = [
            act_path.parent / "submission" / "actionspace_tuples.npy",
            act_path.parent / "submission" / "actionspace_nminus1.npy",
        ]

        env = test_env
        _,_,ray_v24_tf = test_submission_models
        agent = MyAgent(
            env.action_space,
            model_path=ray_v24_tf,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=True,
            scaler=None,
        )

        # Note that model size and action space do not fit, because this is a model
        # for the IEEE118 grid. So we test differently
        assert isinstance(agent.model, tf.keras.models.Model)

        # We test with an example input :
        np.random.seed(42)
        tf.random.set_seed(42)
        a1 = np.random.random((1429,))
        out = agent.model.predict(a1.reshape(1,-1))
        action_prob = tf.nn.softmax(out).numpy().reshape(-1)
        assert action_prob.argmax()== 2


        # Now we test torch model
        ray_v24_torch= test_submission_model_torch


        agent = MyAgent(
            env.action_space,
            model_path=ray_v24_torch,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=True,
            scaler=None,
            run_with_tf=False
        )

        assert isinstance(agent.model, nn.Module)

        # We test with an example input :
        np.random.seed(42)
        a1 = np.random.random((1429,)).astype('f')
        out, _ = agent.model.forward({"obs": a1.reshape(1, -1)})
        action_prob = F.softmax(out,dim=1)
        assert action_prob.argmax().item() == 0



    def test_init_junior(self, test_env, test_paths_env,senior_values_tf, senior_values_torch, test_temp_save,
                         test_junior_input_path):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """

        _,act_list,junior_model,_,_,_ = senior_values_tf

        env = test_env
        agent = MyAgent(
            env.action_space,
            model_path=junior_model,
            this_directory_path=act_list,
            action_space_path=act_list,
            subset=True,
            scaler=None,
        )

        # This one here should be a sequential model
        assert isinstance(agent.model, tf.keras.models.Sequential)

        # We test with an example input :
        np.random.seed(42)
        tf.random.set_seed(42)


        # The junior has a different model size here
        a1 = np.random.random((1429,))
        out = agent.model.predict(a1.reshape(1,-1))
        action_prob = tf.nn.softmax(out).numpy().reshape(-1)
        assert action_prob.argmax()==2

        # Now torch model
        torch.manual_seed(42)
        _, act_list, _, _, _, _ = senior_values_torch
        # to run properly create junior model one more time on the system where tests are running
        ckpt_path = test_temp_save

        if not ckpt_path.is_dir():
            os.mkdir(ckpt_path)
        else:
            if not os.listdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                os.mkdir(ckpt_path)

        data_path, name = test_junior_input_path

        junior = Junior(action_space_file=act_list, seed=42, run_with_tf=False)
        junior.train(
            run_name="junior",
            dataset_path=data_path,
            target_model_path=ckpt_path,
            dataset_name=name,
            epochs=30,
        )
        assert (ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt").is_file()

        agent = MyAgent(
            env.action_space,
            model_path=ckpt_path / "ckpt-junior" / "epoch=29-step=30.ckpt",
            this_directory_path=act_list,
            action_space_path=act_list,
            subset=True,
            scaler=None,
            run_with_tf=False
        )

        # This one here should be a sequential model
        assert isinstance(agent.model, nn.Module)

        # We test with an example input :
        np.random.seed(42)
        tf.random.set_seed(42)

        # The junior has a different model size here
        a1 = np.random.random((1429,)).astype('f')
        out = agent.model.forward(a1.reshape(1, -1))
        action_prob = F.softmax(out,dim=1)
        assert action_prob.argmax().item() == 0
        shutil.rmtree(ckpt_path)
        os.mkdir(ckpt_path)
        assert not os.listdir(ckpt_path)




