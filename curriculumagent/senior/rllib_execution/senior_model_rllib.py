"""
This file constist of two custom models, which transfer the junior weights into the Rllib experiment.
"""

import logging
from pathlib import Path
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym.spaces import Discrete, Box


class Grid2OpCustomModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space: Box, action_space: Discrete, num_outputs: int, model_config: dict,
                 name: str, **customized_model_kwargs):
        """ __init__ of the custom model

        Args:
            obs_space: observation space passed by rllib
            action_space: action space passed by rllib
            num_outputs: number of output passed by rllib, shape of action space
            model_config: configurations of the model
            name: name of the model, if wanted
        """

        super(Grid2OpCustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        n_cell = 1000
        initializer = tf.keras.initializers.Orthogonal()

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        layer1 = tf.keras.layers.Dense(n_cell, name="layer_1",
                                       activation=tf.nn.relu,
                                       kernel_initializer=initializer)(self.inputs)
        layer2 = tf.keras.layers.Dense(n_cell, name="layer_2", activation='relu',
                                       kernel_initializer=initializer)(layer1)
        layer3 = tf.keras.layers.Dense(n_cell, name="layer_3", activation='relu',
                                       kernel_initializer=initializer)(layer2)
        layer4 = tf.keras.layers.Dense(n_cell, name="layer_4", activation='relu',
                                       kernel_initializer=initializer)(layer3)
        act_layer = tf.keras.layers.Dense(num_outputs, name="action_out", activation=None,
                                          kernel_initializer=initializer)(layer4)
        val_hidden_layer = tf.keras.layers.Dense(action_space.n, name="layer_val_hidden", activation='relu',
                                                 kernel_initializer=initializer)(layer4)
        val_layer = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=initializer)(
            val_hidden_layer)

        self.base_model = tf.keras.Model(self.inputs, [act_layer, val_layer])

        path_to_junior = customized_model_kwargs["path_to_junior"]

        self._params_copy(path=path_to_junior)

    def _params_copy(self, path: Path):
        """ Private Method from PPO code. Overwriting the weights of the rllib model by the junior model

        Used to copy the weights of the Junior model onto the model of the Rllib custom model.

        Args:
            path: Path of the Junior model checkpoints.

        Returns: None

        """
        # ToDo: This must be checked with newer approaches by the junior!
        model = tf.keras.models.load_model(path)
        self.base_model.layers[1].set_weights(model.layers[0].get_weights())
        self.base_model.layers[2].set_weights(model.layers[1].get_weights())
        self.base_model.layers[3].set_weights(model.layers[2].get_weights())
        self.base_model.layers[4].set_weights(model.layers[4].get_weights())
        self.base_model.layers[5].set_weights((*map(lambda x: x / 5, model.layers[6].get_weights()),))

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class AdvancedCustomModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space: Box, action_space: Discrete, num_outputs: int, model_config: dict,
                 name: str, **customized_model_kwargs):
        """ __init__ of the custom model

        Args:
            obs_space: observation space passed by rllib
            action_space: action space passed by rllib
            num_outputs: number of output passed by rllib, shape of action space
            model_config: configurations of the model
            name: name of the model, if wanted
            path_to_junior: Path to Junior model
            custom_config: Custom Model config from junior Hyper-parameter selections


        """
        super(AdvancedCustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        if "custom_config" in customized_model_kwargs.keys():
            cconfig = customized_model_kwargs["custom_config"]
            if isinstance(cconfig["activation"], str):
                if cconfig["activation"] == "leaky_relu":
                    activation = tf.keras.layers.LeakyReLU()
                elif cconfig["activation"] == "relu":
                    activation = tf.keras.layers.ReLU()
                elif cconfig["activation"] == "elu":
                    activation = tf.keras.layers.ELU()
                else:
                    activation = tf.keras.layers.ReLU()

            else:
                logging.warning("Wrong input type of activation. Take relu instead")
                activation = tf.keras.layers.ReLU()

            initializer = cconfig["initializer"]

            layer_size = [cconfig["layer1"], cconfig["layer2"], cconfig["layer3"], cconfig["layer4"]]
        else:
            layer_size = [1000, 1000, 1000, 1000]
            initializer = tf.keras.initializers.Orthogonal()
            activation = tf.keras.layers.ReLU()

        # Now Init model:
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        layer1 = tf.keras.layers.Dense(layer_size[0], name="layer_1",
                                       activation=activation,
                                       kernel_initializer=initializer)(self.inputs)
        layer2 = tf.keras.layers.Dense(layer_size[1], name="layer_2", activation=activation,
                                       kernel_initializer=initializer)(layer1)
        layer3 = tf.keras.layers.Dense(layer_size[2], name="layer_3", activation=activation,
                                       kernel_initializer=initializer)(layer2)
        layer4 = tf.keras.layers.Dense(layer_size[3], name="layer_4", activation=activation,
                                       kernel_initializer=initializer)(layer3)
        act_layer = tf.keras.layers.Dense(num_outputs, name="action_out", activation=None,
                                          kernel_initializer=initializer)(layer4)
        val_hidden_layer = tf.keras.layers.Dense(action_space.n, name="layer_val_hidden", activation=activation,
                                                 kernel_initializer=initializer)(layer4)
        val_layer = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=initializer)(
            val_hidden_layer)

        self.base_model = tf.keras.Model(self.inputs, [act_layer, val_layer])

        path_to_junior = customized_model_kwargs["path_to_junior"]

        self._params_copy(path=path_to_junior)

    def _params_copy(self, path: Path):
        """ Private Method from PPO code. Overwriting the weights of the rllib model by the junior model

        Used to copy the weights of the Junior model onto the model of the Rllib custom model.

        Args:
            path: Path of the Junior model checkpoints.

        Returns: None

        """
        # ToDo: This must be checked with newer approaches by the junior!
        model = tf.keras.models.load_model(path)
        self.base_model.layers[1].set_weights(model.layers[0].get_weights())
        self.base_model.layers[2].set_weights(model.layers[1].get_weights())
        self.base_model.layers[3].set_weights(model.layers[2].get_weights())
        self.base_model.layers[4].set_weights(model.layers[4].get_weights())
        self.base_model.layers[5].set_weights((*map(lambda x: x / 5, model.layers[6].get_weights()),))

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}
