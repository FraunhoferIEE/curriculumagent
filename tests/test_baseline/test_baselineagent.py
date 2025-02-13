import os
import shutil

import grid2op
import pytest
import ray
import tensorflow as tf
from grid2op.Agent import BaseAgent
from keras.engine.functional import Functional
from lightsim2grid import LightSimBackend
from tensorflow.keras.models import Sequential

from curriculumagent.baseline import CurriculumAgent
from curriculumagent.submission.my_agent import MyAgent


class TestBaselineAgent:
    """
    Test suite of the baseline agent
    """

    def test_init(self):
        """
        Testing, whether the model is correctly loaded
        """

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()

        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="El Testo")

        # Test the default values
        assert isinstance(myagent, BaseAgent)
        assert isinstance(myagent.observation_space, grid2op.Observation.ObservationSpace)
        assert isinstance(myagent.action_space, grid2op.Action.ActionSpace)
        assert myagent.name == "El Testo"
        assert myagent.senior is None
        assert myagent.agent is None
        assert myagent.do_nothing.as_dict() == {}

    def test_runnable_do_nothing(self):
        """
        Testing, whether the model is runnable
        """

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()

        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="El Testo Secundo")
        env.seed(42)
        obs = env.reset()
        assert obs.to_vect().shape == (467,)
        done = False
        while not done:
            with pytest.warns():
                act = myagent.act(observation=obs, reward=0, done=False)
            assert act.as_dict() == {}
            obs, rew, done, info = env.step(act)
        assert done

    def test_load_errors(self, test_baseline_models):
        """
        Testing, whether the errors are correctly raised!
        """

        senior_path, junior_path = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="El Junior agento")

        # In the junior path we do not have the action file
        with pytest.raises(FileNotFoundError):
            myagent.load(junior_path)

        # Here only a npy file exists.
        with pytest.raises(FileNotFoundError):
            myagent.load(senior_path / "actions")

    def test_loading_junior_with_separate_action_file(self, test_baseline_models):
        """
        Testing, whether loading the junior model works.
        """
        senior_path, junior_path = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="El Junior agento")

        assert myagent.agent is None

        # Now load junior model, with separate action file
        myagent.load(path=junior_path,
                     actions_path=senior_path / "actions")
        assert isinstance(myagent.agent, MyAgent)
        assert isinstance(myagent.agent.model, Functional)

        # Test, wether model works:
        obs = env.reset()
        act = myagent.act(observation=obs, reward=0, done=False)
        assert isinstance(act, grid2op.Action.BaseAction)

    def test_loading_senior_with_subset(self, test_baseline_models):
        """
        Testing, whether loading the junior model works.
        """
        senior_path, _ = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="El grande senior")

        assert myagent.agent is None

        # Now load junior model, with separate action file
        myagent.load(path=senior_path, subset=False)
        assert isinstance(myagent.agent, MyAgent)
        assert isinstance(myagent.agent.model, tf.keras.models.Model)

        # Test, wether model works:
        obs = env.reset()
        act = myagent.act(observation=obs, reward=0, done=False)

        assert isinstance(act, grid2op.Action.BaseAction)

    def test_actions_senior(self,  test_baseline_models):
        """
        Testing multiple action steps
        """
        senior_path, _ = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="Run Forrest Run")
        myagent.load(path=senior_path)

        obs = env.reset()
        done = False
        non_zero = 0
        while not done:
            act = myagent.act(observation=obs, reward=0, done=done)
            if act.as_dict() != {}:
                non_zero += 1
            obs, rew, done, info = env.step(act)
        assert done is True
        assert non_zero > 0

    def test_actions_junior(self, test_baseline_models):
        """
        Testing multiple action steps
        """
        senior_path, junior_path = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="Run little junior, run!")

        assert myagent.agent is None

        # Now load junior model, with separate action file
        myagent.load(path=junior_path,
                     actions_path=senior_path / "actions")

        obs = env.reset()
        done = False
        non_zero = 0
        while not done:
            act = myagent.act(observation=obs, reward=0, done=done)
            if act.as_dict() != {}:
                non_zero += 1
            obs, rew, done, info = env.step(act)
        assert done is True
        assert non_zero > 0

    def test_save_baseline(self, test_baseline_models, test_temp_save):
        """
        Checking, whether the saving works
        """
        senior_path, _ = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="Run Forrest Run")
        myagent.load(path=senior_path)

        #
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        assert test_temp_save.is_dir() and len(os.listdir(test_temp_save)) == 0

        myagent.save(test_temp_save)

        assert (test_temp_save / "model" / "saved_model.pb")
        assert (test_temp_save / "actions" / "actions.npy")

        # Test whether loading works effortlessly:
        myagent.load(test_temp_save)

        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)

    def test_create_submission_dir(self,test_baseline_models,test_temp_save):
        """
        Testing whether create_submission works
        """
        senior_path, _ = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="Run Forrest Run")
        myagent.load(path=senior_path)

        #
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        assert test_temp_save.is_dir() and len(os.listdir(test_temp_save)) == 0

        assert myagent.agent is not None

        myagent.create_submission(test_temp_save)

        assert (test_temp_save /"common" / "__init__.py").is_file()
        assert (test_temp_save / "common" / "obs_converter.py").is_file()
        assert (test_temp_save / "common" / "utilities.py").is_file()
        assert (test_temp_save / "my_agent.py").is_file()
        assert (test_temp_save / "__init__.py").is_file()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)

    @pytest.mark.ultra_slow
    @pytest.mark.slow
    def test_training_senior(self, test_baseline_models, test_temp_save):
        """
        Testing, whether the simple training of the senior works
        """
        senior_path, _ = test_baseline_models

        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        myagent = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space,
                                  name="Run Forrest Run")
        myagent.load(path=senior_path)
        # Delete everything.
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
        assert test_temp_save.is_dir() and len(os.listdir(test_temp_save)) == 0

        ray.init()

        assert ray.is_initialized
        assert myagent.senior is None
        myagent.train(env=env,
                      iterations=1,
                      save_path=test_temp_save)

        assert myagent.senior.ppo.iteration == 1
        assert (test_temp_save / "model" / "saved_model.pb")
        assert (test_temp_save / "actions" / "actions.npy")
        # Check if the model was saved correctly:

        # Back to normal
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)
