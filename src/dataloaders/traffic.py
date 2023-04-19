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

