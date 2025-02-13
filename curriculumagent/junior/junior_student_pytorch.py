"""In this file, a neural network is developed to fit the dataset generated by Tutor.
Depending on the observation space and action space, the tutor model can/has to be
adjusted.

The Junior model returns a one-hot encoded output, based on the number of actions.

Credit: The junior is a more general approach of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
import logging
import warnings
from collections import ChainMap, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple, List, Union, Optional, TypedDict

import grid2op.Environment
import lightning as L
import nni
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator

from curriculumagent.common.obs_converter import obs_to_vect
from curriculumagent.common.utilities import map_actions
from curriculumagent.junior.torch_datasets import Grid2OpDataset

import nni

logging.basicConfig(level=logging.INFO)


class JuniorParamTorch(TypedDict):
    """
    A dictionary that defines the hyperparameters and paths required for the Junior Torch model.

    Attributes:
        action_space_path (Union[Path, List[Path]]): Either path to the actions or a list containing multiple actions.
        data_path (Path): Path of the Grid2Op environment.
        action_threshold (float): A threshold value between 0 and 1 that defines the action cutoff.
        subset (Union[bool, List[int]]): If True, filters the obs.to_vect based on predefined values. If False, all values
            are considered. Alternatively, one can provide their own subset of values.
        testing (bool): Indicator of whether the Grid2Op environment should be started in testing mode.
        activation (str): The activation function to be used in the model.
        learning_rate (Optional[float]): The learning rate for the model training. Defaults to None.
        layer1 (int): The size of the first layer.
        layer2 (int): The size of the second layer.
        layer3 (int): The size of the third layer.
        layer4 (int): The size of the fourth layer.
        batchsize (int): The batch size for training.
        dropout1 (float): Dropout probability for the first dropout layer.
        dropout2 (float): Dropout probability for the second dropout layer.
        epochs (int): The number of epochs for training.
        initializer (str): The initializer used for weight initialization.
    """

    activation: str
    learning_rate: Optional[float]
    layer1: int
    layer2: int
    layer3: int
    layer4: int
    batchsize: int
    dropout1: float
    dropout2: float
    epochs: int
    initializer: str


class JuniorTorch:
    """
    Defines the Junior Torch model. This class is responsible for configuring and managing the training and evaluation process of the Junior model
    based on PyTorch and PyTorch Lightning frameworks.
    """

    def __init__(
            self,
            action_space_file: Union[Path, List[Path]],
            config: JuniorParamTorch = None,
            seed: Optional[int] = None,
            run_nni: bool = False
    ):
        """ Initializes the Junior Torch model with the specified action space and configuration.

            Except setting all variables, the init additionally requires the size of the train set and optionally
            the number of epochs.

            Note:
                Pass epochs, learning_rate, batchsize, layer_size in the config.

            Args:
                action_space_file (Union[Path, List[Path]]): The action space file(s) used during training. This is required
                to define the input size of the Junior model.
                config (JuniorParamTorch, optional): Dictionary containing hyperparameters for the model. Defaults to None.
                seed (Optional[int]): Seed for reproducibility. Defaults to None.
                run_nni (bool): Flag indicating if NNI is used. If True, a callback is added. Defaults to False.

            Returns:
                None.
        """
        # Set the seed
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
            torch.random.manual_seed(self.seed)

        # Callback:
        if run_nni:
            # ToDo Build NNi Callback.
            self.callback = [SendMetrics()] #ToDo: Need to be checked
        else:
            self.callback = []

        self.config = config
        self.batch_size = config.get("batchsize", 256)
        self.epochs = config.get("epochs", 1000)

        self.action_space_file = action_space_file
        self.lightning_model = Junior_PtL(action_space_file=action_space_file,
                                          model_config=self.config)

    def train(self,
              dataset_path: Path,
              target_model_path: Path,
              dataset_name: str = "junior_dataset",
              epochs: int = 1000,
              scaler: Optional[Union[str, Path, BaseEstimator]] = None,
              **kwargs
              ):
        """
        Trains the Junior Torch model.

        Args:
            dataset_path (Path): Path to the dataset files.
            target_model_path (Path): Path where the trained model will be saved.
            dataset_name (str): The name of the dataset file (e.g., `{dataset_name}_train.npz`). Defaults to "junior_dataset".
            epochs (int): The number of epochs to train. Defaults to 1000.
            scaler (Optional[Union[str, Path, BaseEstimator]]): Optional scaler for data normalization. Defaults to None.

        Returns:
            History: A record of the training process, including loss and accuracy.
        """

        # Create Datasets
        dat_tr = Grid2OpDataset(
            root=dataset_path,
            dataset_name=dataset_name,
            split="train",
            scaler=scaler)

        dat_val = Grid2OpDataset(
            root=dataset_path,
            dataset_name=dataset_name,
            split="val",
            scaler=scaler)

        dl_tr = torch.utils.data.DataLoader(dat_tr, batch_size=self.batch_size)
        dl_val = torch.utils.data.DataLoader(dat_val, batch_size=self.batch_size)
        if not epochs:
            epochs = self.epochs

        patience = kwargs.get("patience", False)
        history = History()
        callbacks = [history]
        if patience:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks += [early_stopping]

        model_checkpoint = ModelCheckpoint(dirpath=target_model_path)
        callbacks += [model_checkpoint]

        # run model for dummy batch to initialize LazyLinear first layer and then initialize weights
        x, _ = next(iter(dl_tr))
        self.lightning_model.forward(x)
        self.lightning_model.apply_initializer(self.lightning_model.initializer)

        trainer = L.Trainer(max_epochs=epochs, callbacks=callbacks)
        trainer.fit(model=self.lightning_model, train_dataloaders=dl_tr, val_dataloaders=dl_val)

        return history

    def test(self,
             dataset_path: Path,
             dataset_name: str = "junior_dataset",
             checkpoint_path: Optional[Path] = None,
             scaler: Optional[Union[str, Path, BaseEstimator]] = None, ):
        """
        Tests the Junior Torch model on the specified dataset.

        Args:
            dataset_path (Path): Path to the dataset files.
            dataset_name (str): The name of the dataset file (e.g., `{dataset_name}_test.npz`). Defaults to "junior_dataset".
            checkpoint_path (Optional[Path]): Path to the model checkpoint for loading the trained model. Defaults to None.
            scaler (Optional[Union[str, Path, BaseEstimator]]): Optional scaler for data normalization. Defaults to None.

        Returns:
            dict: A dictionary containing the accuracy metrics for top-N predictions.
        """

        dat_test = Grid2OpDataset(
            root=dataset_path,
            dataset_name=dataset_name,
            #env=self.env,
            split="test",
            scaler=scaler)

        trainer = L.Trainer()

        if isinstance(checkpoint_path, Path):
            self.lightning_model = Junior_PtL.load_from_checkpoint(checkpoint_path=checkpoint_path)
            logging.info(f"Imported model from {checkpoint_path}")


        dl_test = torch.utils.data.DataLoader(dat_test.s, batch_size=1)

        a_pred = trainer.predict(self.lightning_model, dataloaders=dl_test)

        top_n = []

        for i in range(len(a_pred)):
            pred_numpy = a_pred[i].numpy()
            top_n.append(pred_numpy.argsort()[-20:])

        # Added accuracy to record the prediction performance
        accuracy = {}

        for n in range(1, 21):
            correct = 0
            for i in range(len(a_pred)):
                if dat_test.a[i, 0] in top_n[i][-n:]:
                    correct += 1

            acc = correct / len(a_pred) * 100
            logging.info(f"accuracy of top-{n} is {acc}")

            accuracy["accuracy of top-%d" % n] = correct / len(a_pred) * 100

        return accuracy


class Junior_PtL(L.LightningModule):
    """
        A PyTorch Lightning module representing the Junior model architecture
    """
    def __init__(self,
                 action_space_file: Union[Path, List[Path]] = None,
                 model_config: dict = {}
                 ):
        """
          Initializes the Junior_PtL model with the given action space and configuration.

          Args:
              action_space_file (Union[Path, List[Path]]): Path or list of paths to the action space files.
              model_config (dict): Configuration dictionary containing hyperparameters such as layer sizes, activation
                  functions, and initializers.

          Returns:
              None.
        """
        super().__init__()
        self.save_hyperparameters()
        list_of_actions = []
        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            list_of_actions = [np.load(Path(action_space_file))]

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            list_of_actions = [np.load(act_path) for act_path in action_space_file]
        self.num_actions = len(dict(ChainMap(*map_actions(list_of_actions))))

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                             num_classes=self.num_actions)
        self.valaccuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_actions)
        self.valaccuracy20 = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_actions,
                                                                            top_k=20)

        self.config = model_config
        self.lr = self.config.get("learning_rate", 5e-4)
        self.batch_size = self.config.get("batchsize", 256)
        self.epochs = self.config.get("epochs", 1000)

        self.layer_size, self.initializer, self.activation = self._extract_config(self.config)

        self.model = self._build_model()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def training_step(self, batch, batch_idx):
        """
            Defines the training loop step. Performs a forward pass, computes the loss, and logs the accuracy.

            Args:
                batch (Tuple[Tensor, Tensor]): A batch of training data (inputs and targets).
                batch_idx (int): The index of the batch.

            Returns:
                Tensor: The computed loss value for the current batch.
        """
        # training_step defines the train loop.
        x, y = batch
        # forward to model
        pred = self.forward(x)
        loss = nn.NLLLoss()
        loss_res = loss(pred, y.reshape(-1))
        self.accuracy(pred, y.reshape(-1))
        self.log('loss', loss_res, on_step=False, on_epoch=True, prog_bar=True)
        self.log('accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss_res

    def configure_optimizers(self):
        """
        Configures the optimizer used for training. Adam optimizer is used with the specified learning rate.

        Returns:
            torch.optim.Optimizer: The optimizer for the model parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation loop step. Performs a forward pass, computes the validation loss, and logs accuracy.

        Args:
            batch (Tuple[Tensor, Tensor]): A batch of validation data (inputs and targets).
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed validation loss value for the current batch.
        """
        # training_step defines the train loop.
        x, y = batch

        # forward to model
        pred = self.forward(x)
        loss = nn.NLLLoss()
        loss_res = loss(pred, y.reshape(-1))
        self.valaccuracy(pred, y.reshape(-1))
        self.valaccuracy20(pred, y.reshape(-1))

        self.log('val_loss', loss_res, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.valaccuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy_top20', self.valaccuracy20, on_step=False, on_epoch=True, prog_bar=True)
        return loss_res



    def _build_model(self) -> nn.Module:
        """
        Builds and returns the neural network model based on the provided configuration or default settings.

        Returns:
            nn.Module: The constructed PyTorch model as a sequential neural network.
        """

        layers = []

        if not self.config:
            layers.extend([
                ("layer1", nn.LazyLinear(out_features=1000)),  # Matches Senior's layer1
                ("activation1", self.activation),
                ("layer2", nn.Linear(in_features=1000, out_features=1000)),  # Matches Senior's layer2
                ("activation2", self.activation),
                ("layer3", nn.Linear(in_features=1000, out_features=1000)),  # Matches Senior's layer3
                ("activation3", self.activation),
                ("dropout1", nn.Dropout(0.25)),  # Dropout layers can be skipped in Senior
                ("layer4", nn.Linear(in_features=1000, out_features=1000)),  # Matches Senior's layer4
                ("activation4", self.activation),
                ("dropout2", nn.Dropout(0.25)),  # Dropout layers can be skipped in Senior
                ("act_layer", nn.Linear(in_features=1000, out_features=self.num_actions)),  # Matches Senior's act_layer
            ])
        else:
            # Build model based on hyperparameters:
            layers = [("layer1", nn.LazyLinear(out_features=self.layer_size[0])),
                      ("activation1", self.activation)]

            if self.config["layer2"] != 0:
                layers += [
                    ("layer2", nn.Linear(in_features=self.layer_size[1], out_features=self.layer_size[2])),
                    ("activation2", self.activation)
                ]

            if self.config["layer3"] != 0:
                layers += [
                    ("layer3", nn.Linear(in_features=self.layer_size[2], out_features=self.layer_size[3])),
                    ("activation3", self.activation)
                ]

            if self.config["dropout1"] != 0.0:
                layers += [("dropout1", nn.Dropout(self.config["dropout1"]))]

            if self.config["layer4"] != 0:
                layers += [
                    ("layer4", nn.Linear(in_features=self.layer_size[3], out_features=self.num_actions)),
                    ("activation4", self.activation)
                ]

            if self.config["dropout2"] != 0.0:
                layers += [("dropout2", nn.Dropout(self.config["dropout2"]))]

            layers.append(("act_layer", nn.Linear(in_features=self.layer_size[-1], out_features=self.num_actions)))

        model = nn.Sequential(OrderedDict(layers))  # Using OrderedDict to maintain the named layers
        return model

    def forward(self, x):
        """
        Forward pass of the Junior model.

        Args:
            x (Tensor): Input torch tensor for the forward pass.

        Returns:
            Tensor: The model's output tensor (represents action).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        output = self.model(x)
        if output.dim() == 1:  # Handle 1D case
            action = self.log_softmax(output.unsqueeze(0)).squeeze(0)
        else:  # Handle 2D case (common case)
            action = self.log_softmax(output)
        return action

    def _extract_config(self, config) -> Tuple[List, any, any]:
        """
            Extracts model configuration such as layer sizes, activation function, and initializer from the provided config.

            Args:
                config (dict): Configuration dictionary with model parameters.

            Returns:
                Tuple[List[int], Callable, Callable]: A tuple containing the layer sizes, activation function, and initializer.
        """
        layer_size = [1000, 1000, 1000, 1000]
        initializer = nn.init.orthogonal_
        activation = nn.ReLU()

        if not any([k in config.keys() for k in
                    ["activation", "initializer", "layer1", "layer2", "layer3", "layer4"]]):
            warnings.warn("The custom dictionary did not have the correct keys. Using default model.")
            return layer_size, initializer, activation

        activation_option = {"leaky_relu": nn.LeakyReLU(inplace=True),
                             "relu": nn.ReLU(inplace=True),
                             "elu": nn.ELU(inplace=True)}
        initializer_option = {"O": nn.init.orthogonal_,
                              "RN": nn.init.normal_,
                              "RU": nn.init.uniform_,
                              "Z": nn.init.zeros_}

        if "activation" in config:
            activation = activation_option[config["activation"]]

        if "initializer" in config:
            initializer = initializer_option[config["initializer"]]

        if all([l in config for l in ["layer1", "layer2", "layer3", "layer4"]]):
            layer_size = [int(round(config[l])) for l in ["layer1", "layer2", "layer3", "layer4"]]

        return layer_size, initializer, activation

    def apply_initializer(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight)


class History(Callback):
    """
    A PyTorch Lightning callback to store the training and validation metrics at the end of each epoch.

    This callback collects metrics such as loss and accuracy during training and validation, and stores them in
    a dictionary for later use or analysis.

    """

    def __init__(self):
        """
        Initializes the History callback by creating an empty dictionary to store training and validation metrics.

        Returns:
            None.
        """
        self.history = {"loss": [],
                        "accuracy": [],
                        "val_loss": [],
                        "val_accuracy": [],
                        "val_accuracy_top20": []}

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of each training epoch to copy the metrics from the trainer and store them in the history.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module (model) being trained.

        Returns:
            None.
        """
        results = trainer.callback_metrics.copy()

        for k in self.history.keys():
            self.history[k].append(float(results[k]))


class SendMetrics(Callback):
    """
    A PyTorch Lightning callback to send training and validation metrics to the NNI framework.

    This callback is used to report metrics such as accuracy and loss at the end of each training epoch and after
    validation, enabling integration with the NNI (Neural Network Intelligence) framework for automated machine learning.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch to send the training accuracy and loss to the NNI framework.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module (model) being trained.

        Returns:
            None.
        """
        # Get the training accuracy and loss from the logged metrics
        train_accuracy = trainer.callback_metrics.get("accuracy")
        train_loss = trainer.callback_metrics.get("loss")

        # Report metrics to NNI
        metrics = {}
        if train_accuracy is not None:
            metrics['train_accuracy'] = train_accuracy.item()

        if train_loss is not None:
            metrics['train_loss'] = train_loss.item()

        if metrics:
            nni.report_intermediate_result(metrics)

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of the validation loop to send the validation accuracy and loss to the NNI framework.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning module (model) being trained.

        Returns:
            None.
        """
        # Get the validation accuracy and loss from the logged metrics
        val_accuracy = trainer.callback_metrics.get("val_accuracy")
        val_loss = trainer.callback_metrics.get("val_loss")

        # Report metrics to NNI
        metrics = {}
        if val_accuracy is not None:
            metrics['val_accuracy'] = val_accuracy.item()

        if val_loss is not None:
            metrics['val_loss'] = val_loss.item()

        if metrics:
            nni.report_intermediate_result(metrics)

