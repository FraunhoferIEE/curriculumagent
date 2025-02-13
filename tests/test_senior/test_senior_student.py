import os
import pickle
import shutil

import pytest
import ray
import tensorflow as tf
from ray._raylet import ObjectRef
from ray.rllib.algorithms import Algorithm
from sklearn.base import BaseEstimator

from curriculumagent.senior.rllib_execution.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.senior_student import Senior
from curriculumagent.submission.my_agent import MyAgent

@pytest.mark.gitlabonly
class TestSenior:
    """
    Testing the Senior class and all its underlying methods
    """

    def test_init_base_model(self, senior_values_tf):
        """
        First, we test whether the agent is able to initialize. Note that this also means that we
        double check the model.
        """
        env_path, actions_path, path_to_junior, test_temp_save, _, scaler = senior_values_tf

        # Check if dir is empty
        if test_temp_save.is_dir():
            if len(os.listdir(test_temp_save)) > 0:
                shutil.rmtree(test_temp_save, ignore_errors=True)

        if not test_temp_save.is_dir():
            os.mkdir(test_temp_save)

        assert actions_path[0].is_file()
        assert actions_path[1].is_file()

        # Now we initialize the Senior.
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1,
                        subset=False)

        # Check for values:
        assert isinstance(senior, Senior)
        assert isinstance(senior.rllib_env, SeniorEnvRllib)
        assert isinstance(senior.ppo, Algorithm)

        # Because we did not pass a model config, this should be false:
        assert senior._Senior__advanced_model is False

        # Close everything and return to normal
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)


    def test_init_advanced_model(self, senior_values_tf):
        """
        Now let's check, whether the custom config is added !
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf

        # Check if dir is empty
        if test_temp_save.is_dir():
            if len(os.listdir(test_temp_save)) > 0:
                shutil.rmtree(test_temp_save, ignore_errors=True)

        if not test_temp_save.is_dir():
            os.mkdir(test_temp_save)

        # Now we initialize the Senior.
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        custom_junior_config=custom_config,  # Set specific config
                        scaler=scaler, num_workers=1,
                        subset=False)

        # Check for values:
        assert isinstance(senior, Senior)
        assert isinstance(senior.rllib_env, SeniorEnvRllib)
        assert isinstance(senior.ppo, Algorithm)

        # Custom config, this should be true
        assert senior._Senior__advanced_model is True

        # Close everything and return to normal
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)


    def test_scaler_imports_and_subset(self, senior_values_tf):
        """
        Testing, the different ways to import the scaler
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)

        with open(scaler, "rb") as fp:  # Pickling
            loaded_scaler = pickle.load(fp)
        if ray.is_initialized():
            ray_scal = ray.put(loaded_scaler)
        else:
            raise FileNotFoundError

        # This should raise a value error:
        with pytest.raises(ValueError):
            senior = Senior(env_path=env_path,
                            action_space_path=actions_path,
                            model_path=path_to_junior,
                            ckpt_save_path=test_temp_save,
                            scaler=scaler,
                            num_workers=1,
                            subset=True)

        # Naturally, the import via path:
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1,
                        subset=False)
        assert isinstance(senior.rllib_env.scaler, BaseEstimator)

        # Now imported scaler:

        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=loaded_scaler,
                        num_workers=1, subset=False)
        assert isinstance(senior.rllib_env.scaler, BaseEstimator)

        # Now with ray

        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=ray_scal, num_workers=1, subset=False)
        assert isinstance(senior.rllib_env.scaler, ObjectRef)

        ray.shutdown()

    @pytest.mark.ulra_slow
    @pytest.mark.slow
    def test_train_runs_without_errors(self, senior_values_tf):
        """
        Testing of training
        """

        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1, subset=False)
        assert senior.ppo.iteration == 0
        out = senior.train(1)
        assert senior.ppo.iteration == 1
        assert isinstance(out, dict)
        paths = os.listdir(test_temp_save)
        assert (test_temp_save / paths[0] / "rllib_checkpoint.json").is_file()

        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)

    @pytest.mark.ulra_slow
    @pytest.mark.slow
    def test_train_default(self, senior_values_tf):
        """
        Testing of training
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior, subset=False)
        assert senior.ppo.iteration == 0
        out = senior.train(1)
        assert senior.ppo.iteration == 1
        assert isinstance(out, dict)

        # cleanup
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)


    def test_restore(self, senior_values_tf, rllib_ckpt):
        """
        Testing whether the Policy can be loaded via restore
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1,
                        subset=False)

        assert senior.ppo.iteration == 0
        senior.restore(rllib_ckpt)
        assert senior.ppo.iteration == 1
        ray.shutdown()


    def test_save_model(self, senior_values_tf, rllib_ckpt):
        """
        Testing whether the previously loaded policy can be saved again
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1,
                        subset=False)

        senior.restore(rllib_ckpt)
        senior.save_to_model(test_temp_save)
        model = tf.keras.models.load_model(test_temp_save)
        model.compile()
        assert isinstance(model, tf.keras.models.Model)
        assert len(os.listdir(test_temp_save)) > 0

        # cleanup
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)


    def test_my_agent_return(self, senior_values_tf, rllib_ckpt, test_temp_save):
        """
        Testing, whether the my_agent is returned
        """
        if test_temp_save.is_dir():
            if len(os.listdir(test_temp_save)) > 0:
                shutil.rmtree(test_temp_save, ignore_errors=True)

        if not test_temp_save.is_dir():
            os.mkdir(test_temp_save)

        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf
        ray.init(ignore_reinit_error=True)
        senior = Senior(env_path=env_path,
                        action_space_path=actions_path,
                        model_path=path_to_junior,
                        ckpt_save_path=test_temp_save,
                        scaler=scaler, num_workers=1,
                        subset=False)

        senior.restore(rllib_ckpt)
        agent = senior.get_my_agent(test_temp_save)
        assert isinstance(agent, MyAgent)
        assert len(os.listdir(test_temp_save)) > 0

        # cleanup
        ray.shutdown()
        shutil.rmtree(test_temp_save, ignore_errors=True)
        os.mkdir(test_temp_save)


    def test_init_errors(self, senior_values_tf):
        """
        Testing for errors
        """
        env_path, actions_path, path_to_junior, test_temp_save, custom_config, scaler = senior_values_tf

        # First raise error due to missing ray init()
        with pytest.raises(AssertionError):
            Senior(env_path=env_path,
                   action_space_path=actions_path,
                   model_path=path_to_junior,
                   ckpt_save_path=test_temp_save,
                   custom_junior_config=custom_config,  # Set specific config
                   scaler=scaler, num_workers=1,
                   subset=False)

        ray.init()

        # Now pass a wrong action set so that the dimensions of the model do not work:
        with pytest.raises(ValueError):
            Senior(env_path=env_path,
                   action_space_path=[actions_path[0]],  # Error here
                   model_path=path_to_junior,
                   ckpt_save_path=test_temp_save,
                   custom_junior_config=custom_config,  # Set specific config
                   scaler=scaler, num_workers=1,
                   subset=False)

        # Now the model import should work, BUT not the environment
        with pytest.raises(ValueError):
            Senior(env_path=env_path,
                   action_space_path=scaler,
                   model_path=path_to_junior,
                   ckpt_save_path=test_temp_save,
                   custom_junior_config=custom_config,
                   scaler=scaler, num_workers=1,
                   subset=False)

        # Wrong scaler input

        senior = Senior(env_path=env_path,
               action_space_path=actions_path,
               model_path=path_to_junior,
               ckpt_save_path=test_temp_save,
               custom_junior_config={"weird": "keys"},
               scaler=actions_path[0], num_workers=1,
               subset=False)
        assert senior.env_config["scaler"] is None

        # Testing custom config
        with pytest.warns():
            Senior(env_path=env_path,
                   action_space_path=actions_path,
                   model_path=path_to_junior,
                   ckpt_save_path=test_temp_save,
                   custom_junior_config=actions_path[0],
                   scaler=scaler, num_workers=1,
                   subset=False)

        # Error from scaler, because a subset:
        with pytest.raises(ValueError):
            Senior(env_path=env_path,
                   action_space_path=actions_path,
                   model_path=path_to_junior,
                   ckpt_save_path=test_temp_save,
                   scaler=scaler, num_workers=1,
                   subset=True)
