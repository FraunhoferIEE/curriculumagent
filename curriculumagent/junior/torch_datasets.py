import logging
import pickle
from typing import Union, Optional, Tuple

import torch
import grid2op
import numpy as np
from pathlib import Path

from lightsim2grid import LightSimBackend
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset


class Grid2OpDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            split="train",
            scaler: Union[str, Path, BaseEstimator] = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super(Grid2OpDataset).__init__()

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend())
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split

        self.scaler = None
        if isinstance(scaler, BaseEstimator):
            self.scaler = scaler
        elif isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided could not be loaded by pickle")

        self.s, self.a = self.load_dataset()


    def load_dataset(self):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / f"{self.dataset_name}_{self.split}.npz"
        data = np.load(path)

        if "val" in self.split:
            s_dat = data["s_validate"]
            a_dat = data["a_validate"]
        else:
            s_dat = data[f"s_{self.split}"]
            a_dat = data[f"a_{self.split}"]

        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def __len__(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def __getitem__(self,index:Optional[int]=None):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        if index is None:
            index = np.random.choice(len(self.a))

        state = torch.Tensor(self.s[index]).type(torch.float)
        action = torch.Tensor(self.a[index]).type(torch.long)
        return state, action
