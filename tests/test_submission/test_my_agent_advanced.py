import logging
import types

import grid2op
import pytest
import tensorflow as tf
from curriculumagent.submission.my_agent_advanced import MyAgent
from tensorflow.python.training.tracking.tracking import AutoTrackable


class TestAdvancedAgent():
    """
    Test suite of the advanced submission agent
    """

    def test_init_old(self, test_env, test_submission_models, test_paths_env):
        """ Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env
        env = test_env
        old_m, _ = test_submission_models
        agent = MyAgent(env.action_space, model_path=old_m,
                        this_directory_path=act_path,
                        action_space_path=None,
                        filtered_obs=True,
                        scaler=None)

        assert isinstance(agent.model, tf.keras.models.Model)
        assert len(agent.actions) == 208

    def test_init_new(self, test_env, test_submission_models, test_paths_env, test_scaler):
        """ Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        _, act_path = test_paths_env
        act_list = [act_path.parent / "submission" / "action_sets" / "actionspace_tuples.npy",
                    act_path.parent / "submission" / "action_sets" / "actionspace_nminus1.npy"]

        env = test_env
        _, new_m = test_submission_models
        agent = MyAgent(env.action_space,
                        model_path=new_m,
                        this_directory_path=act_path.parent / "submission",
                        action_space_path=act_list,
                        filtered_obs=True,
                        scaler=test_scaler)

        assert isinstance(agent.model, AutoTrackable)
        assert len(agent.actions) == 806

    def test_action_new(self, test_env, test_submission_models, test_paths_env, test_scaler):
        """ Testing, whether the init works, especially the models

        First we start with the old model and old action space
        """
        tf.random.set_seed(42)
        _, act_path = test_paths_env
        act_list = [act_path.parent / "submission" / "action_sets" / "actionspace_tuples.npy",
                    act_path.parent / "submission" / "action_sets" / "actionspace_nminus1.npy"]
        env = test_env
        _, new_m = test_submission_models
        agent = MyAgent(env.action_space, model_path=new_m,
                        this_directory_path=act_path.parent / "submission",
                        action_space_path=act_list,
                        filtered_obs=True,
                        scaler=test_scaler)
        assert agent.next_actions is None
        env.set_id(1)
        env.reset()
        obs = env.get_obs()
        do_nothing = True
        rew = 0
        done = False

        obs.simulate(env.action_space({}))

        while do_nothing:
            action = agent.act(observation=obs, reward=rew, done=done)
            action_dict = action.as_dict()
            if 'set_bus_vect' in action_dict.keys() or 'change_bus_vect' in action_dict.keys():
                do_nothing = False
            else:
                obs, rew, done, info = env.step(action)

        assert isinstance(action, grid2op.Action.BaseAction)

        _, rew, done, info = env.step(action)
        assert done is False

        # Check if tuple works:
        assert isinstance(agent.next_actions, types.GeneratorType)

        with pytest.raises(StopIteration):
            next(agent.next_actions)

    # def test_action_old(self, test_env, test_submission_models,test_paths_env):
    #     """ Testing, whether the init works, especially the models
    #
    #     First we start with the old model and old action space
    #     """
    #     _, act_path = test_paths_env
    #     env = test_env
    #     old_m, new_m = test_submission_models
    #     agent = MyAgent(env.action_space, model_path=old_m,
    #                     this_directory_path=act_path,
    #                     action_space_path=None,
    #                     filtered_obs=True,
    #                     scaler=None)
    #     env.set_id(1)
    #     env.reset()
    #     obs = env.get_obs()
    #
    #     out = agent.act(observation=obs,reward=0,done=False)
    #
    #
    #     assert isinstance(out,grid2op.Action.BaseAction)
    #     obs,rew,done,info = env.step(out)
    #
    #
