"""This file constist of two custom models, which transfer the junior weights into the Rllib experiment.

"""
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from gymnasium.spaces import Discrete, Box
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

class Grid2OpCustomModel(TFModelV2):
    """Custom model for policy gradient algorithms.

    """

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs: int,
        model_config: dict,
        name: str,
        **customized_model_kwargs
    ):
        """Constructor of the custom model.

        This is preferably a junior model, however,  a pretrained Senior model should work as well.
        NOte that when using a Senior model, the second option becomes obsolete.

        Args:
            obs_space: Observation space passed by rllib.
            action_space: Action space passed by rllib.
            num_outputs: Number of output passed by rllib, shape of action space.
            model_config: Configurations of the model for RLlib to init the model.
            name: Name of the model, if wanted.
            customized_model_kwargs: Custom Model config from junior Hyper-parameter selections. Here you plug in
            any layer information, activation method or initializer.

        Returns:
            None.

        """
        super(Grid2OpCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        if "custom_config" in customized_model_kwargs.keys():
            cconfig = customized_model_kwargs["custom_config"]
        else:
            cconfig = {}
        layer_size, initializer, activation = self._extract_config(cconfig)

        # Now Init model:
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        layer1 = tf.keras.layers.Dense(
            layer_size[0], name="layer_1", activation=activation, kernel_initializer=initializer)(
            self.inputs
        )
        layer2 = tf.keras.layers.Dense(layer_size[1], name="layer_2", activation=activation, kernel_initializer=initializer)(
            layer1
        )
        layer3 = tf.keras.layers.Dense(
            layer_size[2], name="layer_3", activation=activation, kernel_initializer=initializer
        )(layer2)
        layer4 = tf.keras.layers.Dense(
            layer_size[3], name="layer_4", activation=activation, kernel_initializer=initializer
        )(layer3)
        act_layer = tf.keras.layers.Dense(
            num_outputs, name="action_out", activation=None, kernel_initializer=initializer
        )(layer4)
        val_hidden_layer = tf.keras.layers.Dense(
            action_space.n, name="layer_val_hidden", activation=activation, kernel_initializer=initializer
        )(layer4)
        val_layer = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=initializer)(
            val_hidden_layer
        )

        self.base_model = tf.keras.Model(self.inputs, [act_layer, val_layer])

        path_to_junior = customized_model_kwargs["model_path"]

        self._params_copy(path=path_to_junior)

    def _extract_config(self, config) -> Tuple[List, tf.keras.initializers.Initializer, tf.keras.layers.Layer]:
        """
        Method to extract the hyperparameters of the config

        Args:
            config:

        Returns: layers, initializer and activation methode.

        """
        # Default values:
        layer_size = [1000, 1000, 1000, 1000]
        initializer = tf.keras.initializers.Orthogonal()
        activation = tf.keras.layers.ReLU()

        if not any([k in config.keys() for k in ["activation", "initializer", "layer1",
                                                 "layer2", "layer3", "layer4"]]):
            import warnings
            warnings.warn("The custom dictionary had not the correct keys. Using default model.")
            return layer_size, initializer, activation

        # Activation options:
        activation_option = {"leaky_relu": tf.keras.layers.LeakyReLU(),
                             "relu": tf.keras.layers.ReLU(),
                             "elu": tf.keras.layers.ELU(),
                             }
        # Initializer Options
        initializer_option = {"O": tf.keras.initializers.Orthogonal(),
                              "RN": tf.keras.initializers.RandomNormal(),
                              "RU": tf.keras.initializers.RandomUniform(),
                              "Z": tf.keras.initializers.Zeros()}

        if "activation" in config.keys():
            activation = activation_option[config["activation"]]

        if "initializer" in config.keys():
            initializer = initializer_option[config["initializer"]]

        if all([l in config.keys() for l in ["layer1", "layer2", "layer3", "layer4"]]):
            layer_size = [int(np.round(config[l])) for l in ["layer1", "layer2", "layer3", "layer4"]]

        return layer_size, initializer, activation

    def _params_copy(self, path: Path):
        """Private Method from PPO code. Overwriting the weights of the rllib model by the Junior model.

        This method does one of two things:
        1. Used to copy the weights of the Junior model onto the model of the Rllib custom model.
        2. If the model is already a PPO model, it just copies it.

        Note that for the second option, the custom config does not have an effect.

        Args:
            path: Path of the Junior model checkpoints.

        Returns:
            None.

        """
        model = tf.keras.models.load_model(path)

        if isinstance(model,tf.keras.models.Sequential):
            self.base_model.layers[1].set_weights(model.layers[0].get_weights())
            self.base_model.layers[2].set_weights(model.layers[1].get_weights())
            self.base_model.layers[3].set_weights(model.layers[2].get_weights())
            self.base_model.layers[4].set_weights(model.layers[4].get_weights())
            self.base_model.layers[5].set_weights((*map(lambda x: x / 5, model.layers[6].get_weights()),))
        else:
            model.compile()
            self.base_model = model

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}
