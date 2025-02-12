import numpy as np
import pytest
import tensorflow as tf
from gymnasium.spaces import Box, Discrete

from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModelTF


class TestAdvancedCustomModelTF:

    def test_load_model(self, custom_config_tf):
        """
        First test, whether the model import works at all
        """
        model = tf.keras.models.load_model(custom_config_tf["model_path"])
        assert isinstance(model, tf.keras.Model)

    def test_create_model_one(self, obs_space, action_space, custom_config_tf):
        """
        Create Model and compare
        """
        model = Grid2OpCustomModelTF(obs_space=obs_space, action_space=action_space,
                                    num_outputs=action_space.n, model_config={},
                                    name="test_model", **custom_config_tf)

        # Test the creation of the model
        assert model.obs_space == obs_space
        assert model.action_space == action_space
        assert model.num_outputs == action_space.n
        assert model.name == "test_model"
        assert isinstance(model.base_model, tf.keras.Model)
        assert len(model.base_model.layers) == 8

    def test_wrong_custom_config(self, obs_space, action_space, custom_config_tf):
        """
        Create Model and compare
        """

        with pytest.warns():
            cc = custom_config_tf.copy()
            cc["custom_config"] = {"what the f***":"am I doing here?"}
            Grid2OpCustomModelTF(obs_space=obs_space, action_space=action_space,
                                    num_outputs=action_space.n, model_config={},
                                    name="test_model", **cc)

    def test_wrong_layer_shape(self, obs_space, action_space, custom_config_tf):
        """
        Testing wrong layer input
        """
        wrong_config = custom_config_tf.copy()
        wrong_config['custom_config']["layer1"] = 100
        wrong_config['custom_config']["layer2"] = 8

        with pytest.raises(ValueError):
            Grid2OpCustomModelTF(obs_space=obs_space, action_space=action_space,
                                num_outputs=action_space.n, model_config=wrong_config,
                                name="test_model", **custom_config_tf)

    def test_wrong_shape_of_action_and_obs(self, obs_space, action_space, custom_config_tf):
        """
        Testing wrong input of action and observation
        """
        wrong_config = custom_config_tf.copy()

        wrong_obs = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        wrong_action = Discrete(2)

        # Obs Space
        with pytest.raises(ValueError):
            Grid2OpCustomModelTF(wrong_obs, action_space, action_space.n, wrong_config, "test_model",
                                **custom_config_tf)

        with pytest.raises(ValueError):
            Grid2OpCustomModelTF(obs_space, wrong_action, action_space.n, wrong_config, "test_model",
                                **custom_config_tf)

    def test_extract_config(self, obs_space, action_space, custom_config_tf):
        """
        Create Model and test the extract_config method
        """


        # Example config
        example_config = {'activation': 'elu',
                  'batchsize': 256,
                  'dropout1': 0.018195067193059022,
                  'dropout2': 0.32907678342440344,
                  'initializer': 'RN',
                  'layer1': 1032.228390773144,
                  'layer2': 239.69398910901413,
                  'layer3': 1236.3175666745672,
                  'layer4': 1163.4560269775084,
                  'learning_rate': 0.00016864815673883727,
                  'TRIAL_BUDGET': 100}

        model = Grid2OpCustomModelTF(obs_space, action_space, action_space.n, {}, "test_model",
                                     **custom_config_tf)

        assert example_config["initializer"] == "RN"

        layer_size, initializer, activation = model._extract_config(example_config)
        assert layer_size == [1032, 240, 1236, 1163]
        assert isinstance(initializer[0], tf.keras.initializers.RandomNormal)
        assert isinstance(activation, tf.keras.layers.ELU)


    def test_params_copy(self, obs_space, action_space, custom_config_tf):
        """
        Test the _params_copy method
        """
        model = Grid2OpCustomModelTF(obs_space, action_space, action_space.n, {}, "test_model",
                                    **custom_config_tf)

        # We overwrite for the 4 layers à 1000 with the weights of the first layer
        model.base_model.layers[3].set_weights(model.base_model.layers[2].get_weights())


        # This should work
        assert np.allclose(model.base_model.layers[2].get_weights()[0], model.base_model.layers[2].get_weights()[0])
        assert np.allclose(model.base_model.layers[3].get_weights()[0], model.base_model.layers[2].get_weights()[0])

        # Now we reload the model with the original wheights:
        model._params_copy(custom_config_tf["model_path"])

        assert not np.allclose(model.base_model.layers[3].get_weights()[0], model.base_model.layers[
                2].get_weights()[0])

    #
    def test_forward(self, obs_space, action_space, custom_config_tf):
        """
        Test the forward method
        """
        model = Grid2OpCustomModelTF(obs_space, action_space, action_space.n, {}, "test_model",
                                    **custom_config_tf)


        # This method should now rais an error, because there is no value method:
        with pytest.raises(AttributeError):
            model.value_function()


        obs = tf.constant(np.ones((1, 1429)), dtype=tf.float32)
        model_out, state = model.forward({"obs": obs}, "Don't mind me, just passing through", None)
        assert model_out.shape == (1, 806)
        assert state is "Don't mind me, just passing through"

        # Now we can check,whether there is a value
        value = model.value_function()
        assert isinstance(value,tf.Tensor)
        assert "_value_out" in model.__dict__

    def test_create_model_two(self, obs_space, action_space, custom_config_tf, test_submission_models):
        """
        Testing the import of a junior model vs. a Senior model.
        """
        model = Grid2OpCustomModelTF(obs_space=obs_space, action_space=action_space,
                                    num_outputs=action_space.n, model_config={},
                                    name="test_model", **custom_config_tf)

        # Test the creation of the model
        weights = model.base_model.get_weights()
        assert weights[0][0].shape == (1000,)
        assert weights[0][0][0] == pytest.approx(0.0004194975)

        # Now we try to load a Senior model
        # Note that we do not need the custom_config, because it is overwritten !
        _,_, model_path = test_submission_models
        cc = {"model_path": model_path,
              # By default adding some bullshit here that should not effect the model
              "custom_config": {'activation': 'relu',
                              'initializer': "Z",
                              'layer1': 42,
                              'layer2': 1022,
                              'layer3': 1022,
                              'layer4': 9}}
        # model = Grid2OpCustomModelTF(obs_space=obs_space, action_space=action_space,
        #                             num_outputs=action_space.n, model_config={},
        #                             name="test_model", **cc)
        #
        # weights = model.base_model.get_weights()
        # assert weights[0][0].shape == (1000,) # And not 42!
        # assert weights[0][0][0] == pytest.approx(0.0004194975)