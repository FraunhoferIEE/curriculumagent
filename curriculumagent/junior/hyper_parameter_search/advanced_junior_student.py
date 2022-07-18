"""
Similar to the junior model, this advanced junior student is built for the optimization of the
junior agent. However, the difference is that a larger number of hyper-parameters can be set and used
for the hyper-parameter search via tune.
"""

import logging
from typing import Union, Optional, TypedDict

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.initializers import Initializer

from curriculumagent.junior.junior_student import Junior
import nni


class SendMetrics(tf.keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''

    def on_epoch_end(self, epoch, logs={}):
        """ Keras callback to send the intermediate result to NNI

        Args:
            epoch: Epoch of training
            logs: Log input

        Returns: None, reports the intermediate result to NNI

        """
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


class JuniorParam(TypedDict):
    """
    TypeDict Class with following inputs:
    action_space_path: Either path to the actions or a list containing mutliple actions.
    data_path: path of the Grid2Op environment
    action_threshold: between 0 and 1
    filtered_obs: Should the obs.to_vect be filtered similar to the original Agent. If True,
            the obs.to_vect is filted based on predefined values. If False, all values are considered.
            Alternatively, one can submit the once own values.
            testing: Indicator, whether the underlying Grid2op Env should be started in testing mode or not
    """

    activation: str
    learning_rate: Optional[Union[float, LearningRateSchedule]]
    layer1: int
    layer2: int
    layer3: int
    layer4: int
    batchsize: int
    dropout1: float
    dropout2: float
    epochs: int
    initializer: Initializer


#  Optional[str] = "relu",
# learning_rate: Optional[Union[float, LearningRateSchedule]] = 5e-4,

class AdvancedJunior(Junior):
    """
    Advanced Junior model with more flexible parameters. This Junior can either be used for the
    hyper-parameter search with tune ore with NNI
    """

    def __init__(self, config: JuniorParam,
                 trainset_size: int,
                 num_actions: int = 208,
                 seed: Optional[int] = None,
                 run_nni: bool = False
                 ):
        """ __init__ method to initialize the hyper-parameters for the hyper-parameter search

        Args:
            JuniorParam: Dictionary containing the correct input for the hyper-parameters
            trainset_size: size of training set
            num_actions: size of action set
            seed: Optional seed
            run_nni: whether nni is used. If True, then a specific callback is added.
        """

        if isinstance(config["activation"], str):
            if config["activation"] == "leaky_relu":
                activation = tf.keras.layers.LeakyReLU()
            elif config["activation"] == "relu":
                activation = tf.keras.layers.ReLU()
            elif config["activation"] == "elu":
                activation = tf.keras.layers.ELU()
            else:
                activation = tf.keras.layers.ReLU()

        else:
            logging.warning("Wrong input type of activation. Take relu instead")
            activation = tf.keras.layers.ReLU()

        # Seed:
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

        self.actions = num_actions
        self.lr = config["learning_rate"]
        self.activation = activation
        self.lf = tf.keras.losses.SparseCategoricalCrossentropy()
        self.initializer = config["initializer"]
        self.batch_size = config["batchsize"]
        self.trainset_size = trainset_size
        self.epochs = config["epochs"]
        self.model = self._build_model(config)
        if run_nni:
            self.callback = [SendMetrics()]
        else:
            self.callback = []

    def _build_model(self, config: JuniorParam) -> tf.keras.models.Sequential:
        """

        Args:
            config: Remaining parameter of the junior config

        Returns:

        """
        # Build model based on hyper-parameter:

        # Layer1:
        model_strucuture = [tf.keras.layers.Dense(config["layer1"], activation=self.activation,
                                                  kernel_initializer=self.initializer)]
        # Layer2:
        if config["layer2"] != 0:
            model_strucuture += [tf.keras.layers.Dense(config["layer2"], activation=self.activation,
                                                       kernel_initializer=self.initializer)]
        if config["layer3"] != 0:
            model_strucuture += [tf.keras.layers.Dense(config["layer3"], activation=self.activation,
                                                       kernel_initializer=self.initializer)]
        if config["dropout1"] != 0.0:
            model_strucuture += [tf.keras.layers.Dropout(config["dropout1"])]

        if config["layer4"] != 0:
            model_strucuture += [tf.keras.layers.Dense(config["layer4"], activation=self.activation,
                                                       kernel_initializer=self.initializer)]

        if config["dropout2"] != 0.0:
            model_strucuture += [tf.keras.layers.Dropout(config["dropout2"])]

        model_strucuture += [tf.keras.layers.Dense(self.actions, activation='softmax')]

        model = tf.keras.models.Sequential(model_strucuture)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=self.lf,
                      metrics=['accuracy'])
        logging.info(model.summary)
        return model
