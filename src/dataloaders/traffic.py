"""Network Traffic Data

Dataset: https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html

All the preprocessing done in the .ipynb could be shifted inside these classes to automatically download the data if
it is not present yet. Moreover, hyperparams could be used to configure which exact dataset should be loaded.

"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

from src.dataloaders.base import SequenceDataset, default_data_path


#erstmal ein data set mit train und test. train wird dann in train und val aufgeteilt.

class CustomTrafficDataset(Dataset):
    def __init__(
            self,
            root_path,
            pickle_file_path,
            pickle_file_name="univ2",
            flag="train",
            size=None,
            data_path="univ2",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.forecast_horizon = self.pred_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # put data from file into a tensor once the class is created
        # depending on the flag create the train, validation or test set
        pass

    def __getitem__(self, idx):
        return torch.tensor(seq_x), torch.tensor(seq_y), torch.tensor(mark), torch.tensor(mask)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

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




class TrafficSequenceDataset(SequenceDataset):
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

        self.dataset_train = TrafficDataset(
            root_path=self.data_dir,
            flag="train",
            size=self.size,
            data_path="univ2",
        )

        self.dataset_val = TrafficDataset(
            root_path=self.data_dir,
            flag="val",
            size=self.size,
            data_path="univ2",
        )

        self.dataset_test = TrafficDataset(
            root_path=self.data_dir,
            flag="test",
            size=self.size,
            data_path="univ2",
        )