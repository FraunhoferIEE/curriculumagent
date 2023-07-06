import types

import grid2op
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.training.tracking.tracking import AutoTrackable

from curriculumagent.submission.my_agent import MyAgent


class TestAdvancedAgent:
    """
    Test suite of the advanced submission agent
    """

    def test_init_ray24(self, test_env, test_submission_models, test_paths_env):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env

        act_list = [
            act_path.parent / "submission" / "actionspace_tuples.npy",
            act_path.parent / "submission" / "actionspace_nminus1.npy",
        ]

        env = test_env
        _,_,ray_v24 = test_submission_models
        agent = MyAgent(
            env.action_space,
            model_path=ray_v24,
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
        out,_ = agent.model.predict(a1.reshape(1,-1))
        action_prob = tf.nn.softmax(out).numpy().reshape(-1)
        assert action_prob.argmax()== 681

    def test_init_junior(self, test_env, test_paths_env,senior_values):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """

        _,act_list,junior_model,_,_,_ = senior_values

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
        assert action_prob.argmax()==0


    def test_init_check_overload(self, test_env, test_submission_models, test_paths_env,senior_values):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env
        _, act_list, junior_model, _, _, _ = senior_values

        env = test_env
        tf.random.set_seed(42)# Setting seed for runs
        np.random.seed(42)
        env.seed(42)
        env.reset()
        agent = MyAgent(
            env.action_space,
            model_path=junior_model,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=False,
            scaler=None,
        )

        agent_overload = MyAgent(
            env.action_space,
            model_path=junior_model,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=False,
            scaler=None,
            check_overload=True
        )

        done = False
        obs = env.reset()
        collect_actions = []
        while not done:
            act1 = agent.act(obs,0,done)
            act2 = agent_overload.act(obs,0,done)
            collect_actions.append(act1==act2)
            obs,rew,done,info = env.step(act2)

        # Check if they differ
        assert not all(collect_actions)


    def test_init_check_overload(self, test_env, test_submission_models, test_paths_env,senior_values):
        """Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env
        _, act_list, junior_model, _, _, _ = senior_values

        env = test_env
        tf.random.set_seed(42)# Setting seed for runs
        np.random.seed(42)
        env.seed(42)
        env.reset()
        agent = MyAgent(
            env.action_space,
            model_path=junior_model,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=False,
            scaler=None,
        )

        agent_overload = MyAgent(
            env.action_space,
            model_path=junior_model,
            this_directory_path=act_path,
            action_space_path=act_list,
            subset=False,
            scaler=None,
            check_overload=True
        )

        done = False
        obs = env.reset()
        collect_actions = []
        while not done:
            act1 = agent.act(obs,0,done)
            act2 = agent_overload.act(obs,0,done)
            collect_actions.append(act1==act2)
            obs,rew,done,info = env.step(act2)

        # Check if they differ
        assert not all(collect_actions)



