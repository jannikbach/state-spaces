"""Network Traffic Data

Dataset: https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html

All the preprocessing done in the .ipynb could be shifted inside these classes to automatically download the data if
it is not present yet. Moreover, hyperparams could be used to configure which exact dataset should be loaded.

"""
import math
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import pickle

import warnings

warnings.filterwarnings("ignore")

from src.dataloaders.base import SequenceDataset, default_data_path


class CustomTrafficDataset(Dataset):

    def __init__(
            self,
            data_path,
            prediction_length,
            pickle_file_name="tensor_data.pkl",
            flag="train",
            context_length=75,
            meta_batch_size=3500,
    ):
        # init
        assert flag in ["train", "test"]
        type_map = {"train": 0, "test": 1}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.pickle_file_name = pickle_file_name
        if prediction_length is None:
            self.prediction_length = context_length
        self.context_length = context_length
        self.meta_batch_size = meta_batch_size
        self.__read_data__()

    def __read_data__(self):
        # put data from file into a tensor once the class is created
        # depending on the flag create the train, validation or test set

        # Train takes the first 80 flows and test takes the last 20

        # Load the tensor from the file using pickle
        complete_path = os.path.join(self.data_path, self.pickle_file_name)

        with open(complete_path, 'rb') as f:
            self.train_obs = pickle.load(f)

        # 80 percent of the batches are used for training, 20 for testing
        num_train_batches = math.floor(self.train_obs.shape[0] * 0.8)
        if self.set_type == 0:  # train
            self.train_obs = self.train_obs[:num_train_batches]
        else:  # test
            self.train_obs = self.train_obs[num_train_batches:]

        num_paths, len_path = self.train_obs.shape[:2]
        idx_path = np.random.randint(0, num_paths,
                                     size=self.meta_batch_size)  # task index, which gets mixed along the  ## would
        # select flow numbers,e.g 3500 times from 1st flow to last tarin flow

        # process
        idx_batch = np.random.randint(self.context_length, len_path - self.prediction_length,
                                      size=self.meta_batch_size)  # would select some middle points from time-steps [
        # 0:10000], 3500 times

        self.obs_batch = np.array([self.train_obs[ip,
                                   ib - self.context_length:ib + self.prediction_length, :].numpy()
                                   for ip, ib in zip(idx_path, idx_batch)], dtype=object)

        # drop timeseries only consisting of zeros
        # seems like the most of them are actually full of zeros
        # also this code does not take overlapping into concern

    def __getitem__(self, idx):
        return self.obs_batch[idx, :self.context_length, :], self.obs_batch[idx, self.context_length:, :]

    def __len__(self):
        return self.obs_batch.shape[0]

    @property
    def d_input(self):
        raise NotImplementedError
        # todo: find out what this stands for and how it is used. Maybe i can just leave it out for now.
        # return self.data_x.shape[-1]

    @property
    def d_output(self):
        raise NotImplementedError
        # if self.features in ["M", "S"]:
        #     return self.data_x.shape[-1]
        # elif self.features == "MS":
        #     return 1
        # else:
        #     raise NotImplementedError


class CustomTrafficSequenceDataset(SequenceDataset):
    _name_ = "traffic"

    @property
    def d_input(self):
        return self.dataset_train.d_input

    @property
    def d_output(self):
        return self.dataset_train.d_output

    @property
    def l_output(self):
        return self.dataset_train.pred_len

    def setup(self):
        self.dataset_train = CustomTrafficDataset(
            flag="train",
            data_path=self.data_dir,
            pickle_file_name="tensor_data.pkl",
        )
        self.split_train_val(0.9)

        self.dataset_test = CustomTrafficDataset(
            flag="test",
            data_path=self.data_dir,
            pickle_file_name="tensor_data.pkl",
            meta_batch_size=700,
        )
