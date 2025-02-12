import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List

import nni
import numpy as np

from collections import ChainMap

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.initializers import Initializer

from typing import Union, Optional, TypedDict, Tuple, List

from curriculumagent.common.utilities import map_actions


class JuniorParamTF(TypedDict):
    """TypeDict Class with following Attributes:
    action_space_path: Either path to the actions or a list containing mutliple actions.
    data_path: Path of the Grid2Op environment
    action_threshold: Between 0 and 1
    subset: Should the obs.to_vect be filtered similar to the original Agent. If True,
            the obs.to_vect is filtered based on predefined values. If False, all values are considered.
            Alternatively, one can submit the once own values.
            testing: Indicator, whether the underlying Grid2Op Environment should be started in testing mode or not
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


class JuniorTF:
    def __init__(
            self,
            action_space_file: Union[Path, List[Path]],
            config: JuniorParamTF =None,
            seed: Optional[int] = None,
            run_nni: bool = False
    ):
        """Constructor of the Junior simple model or model with more flexible parameters. This Junior can also be used
        for the hyperparameter search with tune ore with NNI.

        Except setting all variables, the init additionally requires the size of the train set and optionally
        the number of epochs.

        Note:
            Pass epochs, learning_rate, batchsize, layer_size in the config.

        Args:
            action_space_file: Action Space file that was used for the Tutor training. This is needed to extract the
            correct shape of the Junior model.
            config: Dictionary containing the correct input for the hyperparameters.
            seed: Optional Seed to reproduce results.
            run_nni: Whether NNI is used. If True, then a specific callback is added.

        Returns:
            None.
        """

        # Set self actions to a list to iterate for later
        if config is None:
            config = {}

        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            self.actions = np.load(str(Path(action_space_file)))

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            self.actions = np.concatenate([np.load(str(act_path)) for act_path in action_space_file], axis=0)

        else:
            self.actions = np.load(action_space_file)

        self.num_actions = len(self.actions)

        self.config = config

        layer_size, initializer, activation = self._extract_config(config)

        self.lr = config.get("learning_rate", 5e-4)
        self.batch_size = config.get("batchsize", 256)
        self.epochs = config.get("epochs", 1000)

        self.activation = activation
        self.layer_size = layer_size
        self.lf = tf.keras.losses.SparseCategoricalCrossentropy()
        self.initializer = tf.keras.initializers.Orthogonal(seed=seed)

        # Seed:
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)



        # Init Model: (either simple or advanced)
        self.model = self._build_model()
        if run_nni:
            self.callback = [SendMetrics()]
        else:
            self.callback = []

    def _build_model(self) -> tf.keras.models.Sequential:
        """Build and return the junior network as a keras model.

        Args:
            None.

        Returns:
            Compiled Keras model.

        """
        if not self.config:
            # build standart juniour model

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Dense(units=1000, activation=self.activation, kernel_initializer=self.initializer),
                    tf.keras.layers.Dense(units=1000, activation=self.activation, kernel_initializer=self.initializer),
                    tf.keras.layers.Dense(units=1000, activation=self.activation, kernel_initializer=self.initializer),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(units=1000, activation=self.activation, kernel_initializer=self.initializer),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(self.num_actions, activation="softmax"),
                ]
            )

        else:
            # Build model based on hyperparameter:

            # Layer1:
            model_structure = [
                tf.keras.layers.Dense(self.layer_size[0], activation=self.activation, kernel_initializer=self.initializer)
            ]
            # Layer2:
            if self.config["layer2"] != 0:
                model_structure += [
                    tf.keras.layers.Dense(self.layer_size[1], activation=self.activation,
                                          kernel_initializer=self.initializer)
                ]
            # Layer3:
            if self.config["layer3"] != 0:
                model_structure += [
                    tf.keras.layers.Dense(self.layer_size[2], activation=self.activation,
                                          kernel_initializer=self.initializer)
                ]
            # Dropout 1:
            if self.config["dropout1"] != 0.0:
                model_structure += [tf.keras.layers.Dropout(self.config["dropout1"])]

            # Layer 4:
            if self.config["layer4"] != 0:
                model_structure += [
                    tf.keras.layers.Dense(self.layer_size[3], activation=self.activation,
                                          kernel_initializer=self.initializer)
                ]

            # Dropout 2:
            if self.config["dropout2"] != 0.0:
                model_structure += [tf.keras.layers.Dropout(self.config["dropout2"])]

            model_structure += [tf.keras.layers.Dense(self.num_actions, activation="softmax")]
            model = tf.keras.models.Sequential(model_structure)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=self.lf,
                      metrics=["accuracy"])
        logging.info(model.summary)
        return model

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
            warnings.warn("The custom dictionary had not the correct keys. Revert to default model.")
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

        activation = activation_option[config.get("activation", "relu")]
        initializer = initializer_option[config.get("initializer", "O")]

        if all([l in config.keys() for l in ["layer1", "layer2", "layer3", "layer4"]]):
            layer_size = [int(np.round(config[l])) for l in ["layer1", "layer2", "layer3", "layer4"]]

        return layer_size, initializer, activation

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_validate: np.ndarray,
            y_validate: np.ndarray,
            log_dir: Optional[Union[str, Path]] = None,
            ckpt_dir: Optional[Union[str, Path]] = None,
            patience: Optional[int] = None,
            epochs: Optional[int] = None
    ) -> tf.keras.callbacks.History:
        """Train the junior model for given number of epochs.

        This method builds callbacks for the training and then executes the keras .fit() method
        to train the Junior model on the x_train and y_train data. Validation is recorded as well.

        Args:
            log_dir: Directory for tensorboard callback.
            ckpt_dir: Directory for checkpoint callback.
            x_train: Training data containing the grid observations.
            y_train: Training actions of the tutor.
            x_validate: Validation data containing the grid observations.
            y_validate: Validation actions of the tutor.
            epochs: Number of epochs for the training.
            patience: Optional early stopping criterion.

        Returns:
            Returns training history.

        """
        callbacks = self.callback

        if log_dir is not None:
            tensorboard_callback = TensorBoard(log_dir=log_dir, write_graph=False)
            callbacks += [tensorboard_callback]
        if isinstance(ckpt_dir, (Path, str)):
            if isinstance(ckpt_dir, str):
                ckpt_path = ckpt_dir + "/" + "ckpt_{epoch}"
            else:
                ckpt_path = ckpt_dir / "ckpt_{epoch}"

            cp_callback = ModelCheckpoint(filepath=str(ckpt_path),
                                          save_weights_only=False,
                                          save_freq=10, verbose=0)
            callbacks += [cp_callback]



        if patience is not None:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                verbose=1,
                mode="auto",
                restore_best_weights=True,
            )
            callbacks += [early_stopping]

        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs or self.epochs,
            validation_data=(x_validate, y_validate),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def test(self, x: np.ndarray, y: np.ndarray, save_path: Optional[Path] = None) -> dict:
        """Test the Junior model with input dataset x and targets/actions y.

        The method predicts based on the input x and then computes a ranking, regarding the
        accuracy on the actions.

        The ranking collects, if the action of the tutor was within the 1-20 actions.

        Args:
            x: Input with Tutor observation for the prediction.
            y: Action of the tutor to validate with the prediction.
            save_path: Optional path where the weights of the model are saved.
                If needed, the weights are loaded by model.load_weights(save_path).

        Returns:
            The dictionary that contains the top values.

        """
        if isinstance(save_path, Path):
            self.model = tf.keras.models.load_model(save_path)
            logging.info(f"Imported model from {save_path}")

        a_pred = self.model.predict(x, verbose=1)
        top_n = []
        for i in range(a_pred.shape[0]):
            top_n.append(a_pred[i, :].argsort()[-20:])

        # Added accuracy to record the prediction performance
        accuracy = {}

        for n in range(1, 21):
            correct = 0
            for i in range(a_pred.shape[0]):
                if y[i, 0] in top_n[i][-n:]:
                    correct += 1

            acc = correct / a_pred.shape[0] * 100
            logging.info(f"accuracy of top-{n} is {acc}")

            accuracy["accuracy of top-%d" % n] = correct / a_pred.shape[0] * 100
        return accuracy





class SendMetrics(tf.keras.callbacks.Callback):
    """ Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs=None):
        """Keras callback to send the intermediate result to NNI

        Args:
            epoch: Epoch of training
            logs: Log input

        Returns: None, reports the intermediate result to NNI

        """
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if logs is None:
            logs = {}
        if "val_acc" in logs:
            nni.report_intermediate_result(logs["val_acc"])
        else:
            nni.report_intermediate_result(logs["val_accuracy"])