import os
import shutil
from pathlib import Path

import grid2op
import numpy as np
import ray
from lightsim2grid import LightSimBackend

import pytest
from grid2op.Agent import BaseAgent
import tensorflow as tf
from curriculumagent.baseline import CurriculumAgent, evaluate
from curriculumagent.submission.my_agent_advanced import MyAgent


class TestBaselineAgent():
    """
    Test suite of the baseline agent
    """

    @pytest.mark.slow
    def test_init(self, test_paths_ieee_model):
        """
        Testing, whether the model is correctly loaded
        """
        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()
        path_of_model = test_paths_ieee_model

        myagent = CurriculumAgent(
            action_space=env.action_space,
            model_path=path_of_model,
            name="Test")

        assert isinstance(myagent, BaseAgent)

    @pytest.mark.slow
    def test_runnable(self, test_paths_ieee_model):
        """
        Testing, whether the model is runnable
        """
        assert grid2op.__version__ == "1.7.1"
        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()
        path_of_model = test_paths_ieee_model

        myagent = CurriculumAgent(
            action_space=env.action_space,
            model_path=path_of_model,
            name="Test")

        obs = env.reset()
        assert obs.to_vect().shape == (467,)
        done = False
        while not done:
            act = myagent.act(observation=obs, reward=0, done=False)
            obs, rew, done, info = env.step(act)
        assert done

    @pytest.mark.slow
    def test_save_and_load(self, test_paths_ieee_model):
        """
        Testing, whether the model can be saved and loaded completely.
        """
        assert grid2op.__version__ == "1.7.1"
        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()
        path_of_model = test_paths_ieee_model

        myagent = CurriculumAgent(
            action_space=env.action_space,
            model_path=path_of_model,
            name="Test")

        data_path = Path(__file__).parent.parent / "data" / "temporary_save"
        if not data_path.is_dir():
            data_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(data_path)) == 0

        myagent.save(data_path)

        # Check if model is saved
        agent_path = data_path / "agent"
        assert agent_path.is_dir()
        assert (agent_path / "actions" / "CA_actions.npy").is_file()
        assert (agent_path / "model" / "saved_model.pb").is_file()

        # Now load model: #
        #  For this we overwrite the model:
        assert myagent.model_path != agent_path / "model"
        myagent.agent = []
        assert myagent.agent == []
        myagent.load(data_path)
        assert myagent.model_path == agent_path / "model"
        assert isinstance(myagent.agent, MyAgent)

        # Now lastly let's delete the model:
        shutil.rmtree(agent_path, ignore_errors=True)

    @pytest.mark.slow
    def test_evaluate(self, test_paths_ieee_model):
        """
        Testing, whether the evaluate methode works
        """
        assert grid2op.__version__ == "1.7.1"
        env = grid2op.make("l2rpn_case14_sandbox")
        env.reset()
        path_of_model = test_paths_ieee_model
        env = grid2op.make("l2rpn_case14_sandbox")
        data_path = Path(__file__).parent.parent / "data" / "temporary_save"
        log_paths = data_path/ "logs"

        if not data_path.is_dir():
            data_path.mkdir(exist_ok=True, parents=True)

        np.random.seed(42)
        tf.random.set_seed(42)
        out = evaluate(env,
                       load_path= path_of_model,
                       logs_path= log_paths,
                       nb_episode=2,
                       nb_process=1,
                       max_steps=100,
                       verbose=0,
                       save_gif=False)


        assert (log_paths).is_dir()
        assert (log_paths/"0000").is_dir()
        assert (log_paths/"0001").is_dir()
        assert (log_paths / "dict_action_space.json").is_file()

        # Now lastly let's delete the model:
        shutil.rmtree(log_paths, ignore_errors=True)

    @pytest.mark.ultra_slow
    def test_run_training(self, test_paths_ieee_model):
        """
        Testing, whether the model can be saved and loaded completely.
        """
        assert grid2op.__version__ == "1.7.1"
        env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        env.reset()
        path_of_model = test_paths_ieee_model
        data_path = Path(__file__).parent.parent / "data" / "temporary_save"
        if not data_path.is_dir():
            data_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(data_path)) == 0

        myagent = CurriculumAgent(
            action_space=env.action_space,
            model_path=path_of_model,
            name="Test")

        myagent.train(env=env,
                      name="Test",
                      iterations=1,
                      save_path=data_path,
                      )

        # Check if files are within the directories
        for name in ["teacher", "tutor", "junior", "senior"]:
            assert len(os.listdir(data_path / name)) > 0

        # Assert specific files (e.g. the tutor file and the model of the agent)
        assert (data_path / "teacher" / "general_teacher_experience.csv").is_file()
        assert (data_path / "tutor" / "tutor_experience.npy").is_file()
        assert (data_path / "agent" / "actions" / "CA_actions.npy").is_file()
        assert (data_path / "agent" / "model" / "saved_model.pb").is_file()

        # Now lastly let's delete the model:
        shutil.rmtree(data_path, ignore_errors=True)
        data_path.mkdir(exist_ok=True, parents=True)
        assert len(os.listdir(data_path)) == 0
        ray.shutdown()
