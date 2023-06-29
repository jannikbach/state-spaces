"""ET Dataset from Informer Paper.

Dataset: https://github.com/zhouhaoyi/ETDataset
Dataloader: https://github.com/zhouhaoyi/Informer2020
"""
import math
import pickle
from typing import List
import os

import numpy
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

from src.dataloaders.base import SequenceDataset, default_data_path


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc=1, freq="h"):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:
        dates["month"] = dates.date.apply(lambda row: row.month, 1)
        dates["day"] = dates.date.apply(lambda row: row.day, 1)
        dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
        dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
        dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
        dates["minute"] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            "y": [],
            "m": ["month"],
            "w": ["month"],
            "d": ["month", "day", "weekday"],
            "b": ["month", "day", "weekday"],
            "h": ["month", "day", "weekday", "hour"],
            "t": ["month", "day", "weekday", "hour", "minute"],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack(
            [feat(dates) for feat in time_features_from_frequency_str(freq)]
        ).transpose(1, 0)


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


class TrafficScaler:

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.new_zero = 0.0

    def fit(self, data_tensor):
        self.mean = torch.mean(data_tensor, dim=1, keepdim=True)
        self.std = torch.std(data_tensor, dim=1, keepdim=True)
        self.new_zero = -torch.div(self.mean, self.std)

    def transform(self, data_tensor):
        self.mean = torch.mean(data_tensor, dim=1, keepdim=True)
        self.std = torch.std(data_tensor, dim=1, keepdim=True)
        return torch.div(torch.sub(data_tensor, self.mean), self.std)


class InformerDataset(Dataset):
    def __init__(
            self,
            root_path,
            flag="train",
            size=None,
            features="S",
            data_path="ETTh1.csv",
            target="OT",
            scale=True,
            inverse=False,
            timeenc=0,
            freq="h",
            cols=None,
            eval_stamp=False,
            eval_mask=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
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

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.eval_stamp = eval_stamp
        self.eval_mask = eval_mask
        self.forecast_horizon = self.pred_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        return border1s, border2s

    def _process_columns(self, df_raw):
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        return df_raw[["date"] + cols + [self.target]]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        df_raw = self._process_columns(df_raw)

        border1s, border2s = self._borders(df_raw)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]: border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

        self.data_stamp = data_stamp  # for etth: [month day weekday hour]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x = np.concatenate(
            [seq_x, np.zeros((self.pred_len, self.data_x.shape[-1]))], axis=0
        )

        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin: r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len: r_end],
                ],
                0,
            )
            raise NotImplementedError
        else:
            # seq_y = self.data_y[r_begin:r_end] # OLD in Informer codebase
            seq_y = self.data_y[s_end:r_end]

        # OLD in Informer codebase
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.eval_stamp:
            mark = self.data_stamp[s_begin:r_end]
        else:
            mark = self.data_stamp[s_begin:s_end]
            mark = np.concatenate([mark, np.zeros((self.pred_len, mark.shape[-1]))], axis=0)

        if self.eval_mask:
            mask = np.concatenate([np.zeros(self.seq_len), np.ones(self.pred_len)], axis=0)
        else:
            mask = np.concatenate([np.zeros(self.seq_len), np.zeros(self.pred_len)], axis=0)
        mask = mask[:, None]

        # Add the mask to the timestamps: # 480, 5
        # mark = np.concatenate([mark, mask[:, np.newaxis]], axis=1)

        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        if self.timeenc == 0:
            mark = mark.astype(np.int64)
        else:
            mark = mark.astype(np.float32)
        mask = mask.astype(np.int64)

        # here we use that we only take the last outputs of the model automatically, by using the respective decoder
        # to apply this to the traffic data set I have to concatenate the context with the respective amount of zeros
        # using np.concatenate.
        # I still do not know what mark and mask are used for.
        # mask and mark seem to be relevant for the transformers model. but s4 seems not to need it. also in the
        # time encoder mask and mark are checked because it encodes the time for the datapoints in the mask.
        # no idea why we need a timestamp in the mordel or if it even is fitted
        return torch.tensor(seq_x), torch.tensor(seq_y), torch.tensor(mark), torch.tensor(mask)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    @property
    def d_input(self):
        return self.data_x.shape[-1]

    @property
    def d_output(self):
        if self.features in ["M", "S"]:
            return self.data_x.shape[-1]
        elif self.features == "MS":
            return 1
        else:
            raise NotImplementedError

    @property
    def n_tokens_time(self):
        if self.freq == 'h':
            return [13, 32, 7, 24]
        elif self.freq == 't':
            return [13, 32, 7, 24, 4]
        else:
            raise NotImplementedError


class _Dataset_ETT_hour(InformerDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _borders(self, df_raw):
        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        return border1s, border2s

    def _process_columns(self, df_raw):
        return df_raw

    @property
    def n_tokens_time(self):
        assert self.freq == "h"
        return [13, 32, 7, 24]


class _Dataset_ETT_minute(_Dataset_ETT_hour):
    def __init__(self, data_path="ETTm1.csv", freq="t", **kwargs):
        super().__init__(data_path=data_path, freq=freq, **kwargs)

    def _borders(self, df_raw):
        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        return border1s, border2s

    @property
    def n_tokens_time(self):
        assert self.freq == "t"
        return [13, 32, 7, 24, 4]


class _Dataset_Weather(InformerDataset):
    def __init__(self, data_path="WTH.csv", target="WetBulbCelsius", **kwargs):
        super().__init__(data_path=data_path, target=target, **kwargs)


class _Dataset_ECL(InformerDataset):
    def __init__(self, data_path="ECL.csv", target="MT_320", **kwargs):
        super().__init__(data_path=data_path, target=target, **kwargs)


class InformerSequenceDataset(SequenceDataset):

    @property
    def n_tokens_time(self):
        # Shape of the dates: depends on `timeenc` and `freq`
        return self.dataset_train.n_tokens_time  # data_stamp.shape[-1]

    @property
    def d_input(self):
        return self.dataset_train.d_input

    @property
    def d_output(self):
        return self.dataset_train.d_output

    @property
    def l_output(self):
        return self.dataset_train.pred_len

    def _get_data_filename(self, variant):
        return self.variants[variant]

    _collate_arg_names = ["mark", "mask"]  # Names of the two extra tensors that the InformerDataset returns

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / 'informer' / self._name_

        self.dataset_train = self._dataset_cls(
            root_path=self.data_dir,
            flag="train",
            size=self.size,
            features=self.features,
            data_path=self._get_data_filename(self.variant),
            target=self.target,
            scale=self.scale,
            inverse=self.inverse,
            timeenc=self.timeenc,
            freq=self.freq,
            cols=self.cols,
            eval_stamp=self.eval_stamp,
            eval_mask=self.eval_mask,
        )

        self.dataset_val = self._dataset_cls(
            root_path=self.data_dir,
            flag="val",
            size=self.size,
            features=self.features,
            data_path=self._get_data_filename(self.variant),
            target=self.target,
            scale=self.scale,
            inverse=self.inverse,
            timeenc=self.timeenc,
            freq=self.freq,
            cols=self.cols,
            eval_stamp=self.eval_stamp,
            eval_mask=self.eval_mask,
        )

        self.dataset_test = self._dataset_cls(
            root_path=self.data_dir,
            flag="test",
            size=self.size,
            features=self.features,
            data_path=self._get_data_filename(self.variant),
            target=self.target,
            scale=self.scale,
            inverse=self.inverse,
            timeenc=self.timeenc,
            freq=self.freq,
            cols=self.cols,
            eval_stamp=self.eval_stamp,
            eval_mask=self.eval_mask,
        )


class ETTHour(InformerSequenceDataset):
    _name_ = "etth"

    _dataset_cls = _Dataset_ETT_hour

    init_defaults = {
        "size": None,
        "features": "S",
        "target": "OT",
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        0: "ETTh1.csv",
        1: "ETTh2.csv",
    }


class ETTMinute(InformerSequenceDataset):
    _name_ = "ettm"

    _dataset_cls = _Dataset_ETT_minute

    init_defaults = {
        "size": None,
        "features": "S",
        "target": "OT",
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "t",
        "cols": None,
    }

    variants = {
        0: "ETTm1.csv",
        1: "ETTm2.csv",
    }


class Weather(InformerSequenceDataset):
    _name_ = "weather"

    _dataset_cls = _Dataset_Weather

    init_defaults = {
        "size": None,
        "features": "S",
        "target": "WetBulbCelsius",
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        0: "WTH.csv",
    }


class ECL(InformerSequenceDataset):
    _name_ = "ecl"

    _dataset_cls = _Dataset_ECL

    init_defaults = {
        "size": None,
        "features": "S",
        "target": "MT_320",
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        0: "ECL.csv",
    }


class CustomTrafficDataset(Dataset):
    def __init__(
            self,
            data_path=None,
            pickle_file_name="tensor_data.pkl",
            flag="train",
            context_length=75,
            prediction_length=None,
            meta_batch_size=3500,
            eval_mask=True,
            scale=True,
            **kwargs,
    ):

        super().__init__(**kwargs)
        # init
        assert flag in ["train", "test"]
        type_map = {"train": 0, "test": 1}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.pickle_file_name = pickle_file_name
        if prediction_length is None:
            self.prediction_length = context_length
        if self.data_path is None:
            self.data_path = default_data_path
        self.context_length = context_length
        self.meta_batch_size = meta_batch_size
        self.eval_mask = eval_mask
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        # put data from file into a tensor once the class is created
        # depending on the flag create the train, validation or test set

        # Train takes the first 80 flows and test takes the last 20

        # Load the tensor from the file using pickle
        complete_path = os.path.join(self.data_path, self.pickle_file_name)

        self.scaler = TrafficScaler()

        with open(complete_path, 'rb') as f:
            self.train_obs = pickle.load(f)

        # 80 percent of the batches are used for training, 20 for testing
        num_train_batches = math.floor(self.train_obs.shape[0] * 0.8)
        if self.set_type == 0:  # train
            self.train_obs = self.train_obs[:num_train_batches]
        else:  # test
            self.train_obs = self.train_obs[num_train_batches:]

        self.scaler.fit(self.train_obs)
        if self.scale:
            self.train_obs_scaled = self.scaler.transform(self.train_obs)
        else:
            self.train_obs_scaled = self.train_obs

        num_paths, len_path, features = self.train_obs_scaled.shape

        # Iterator that fills the np.array till it is full with non zero values
        # make sure random numbers only appear once

        n = 0
        used_points = []
        self.obs_batch = np.zeros([self.meta_batch_size, self.context_length + self.prediction_length, features],
                                  dtype=np.float32)

        while n < self.meta_batch_size:
            batch = np.random.randint(0, num_paths)
            middle_point = np.random.randint(self.context_length, len_path - self.prediction_length)

            if (batch, middle_point) not in used_points:
                candidate = self.train_obs_scaled[batch,
                            middle_point - self.context_length:middle_point + self.prediction_length, :].numpy()
                # assure context and prediction have at least one non zero value
                no_information_level = self.scaler.new_zero[batch].item() if self.scale else 0
                if (not np.all(candidate[:self.context_length] == no_information_level)) & (
                        not np.all(candidate[self.context_length:] == no_information_level)):
                    used_points.append((batch, middle_point))
                    # self.obs_batch = np.insert(self.obs_batch, n, candidate, axis=0)
                    # same as above but in place
                    self.obs_batch[n:n + 1, :, :] = candidate[np.newaxis, :]
                    n += 1

        # we could do scaling here as well

        self.x = np.concatenate(
            [self.obs_batch[:, :self.context_length, :],
             np.zeros((self.obs_batch.shape[0], self.prediction_length, self.obs_batch.shape[-1]), dtype=np.float32)],
            axis=1
        )

        # x = [meta_batch_size (3500), time (150), features (1)]

        self.y = self.obs_batch[:, -self.prediction_length:, :]

        # if self.eval_stamp:
        #     mark = self.data_stamp[s_begin:r_end]
        # else:
        #     mark = self.data_stamp[s_begin:s_end]
        #     mark = np.concatenate([mark, np.zeros((self.pred_len, mark.shape[-1]))], axis=0)

        if self.eval_mask:
            mask = np.concatenate([np.zeros(self.context_length), np.ones(self.prediction_length)], axis=0)
        else:
            mask = np.concatenate([np.zeros(self.context_length), np.zeros(self.prediction_length)], axis=0)
        self.mask = mask[:, None]


        print(self.x.shape)
        print(self.y.shape)




    def __getitem__(self, idx):
        return self.x[idx], self.y[
            idx], self.mask  # , no mark given (could be timestamps used in embeddings, but not worth it because i only have 1 hour of data)

    def __len__(self):
        return self.x.shape[0]

    @property
    def d_input(self):
        return self.x.shape[-1]  # i think this maybe has to be the number of input verctors (150, self.x.shape[1]) ???

    # already forgot how the encoding and feeding to the model works...

    @property
    def d_output(self):
        return self.x.shape[-1]

    @property
    def l_output(self):
        return self.prediction_length

    @property
    def forecast_horizon(self):
        return self.prediction_length


class CustomTrafficSequenceDataset(SequenceDataset):
    _name_ = "traffic"

    _collate_arg_names = ["mask"]  # ["mark", "mask"]  # Names of the two extra tensors that the InformerDataset returns

    @property
    def d_input(self):
        return self.dataset_test.d_input
        # I assume this is the dimension of the data that goes into the encoder, which in this case is
        # feature dimension (1)

    # d_model = 128, d_output = 128

    @property
    def d_output(self):
        return self.dataset_test.d_output
        # prediction feature dimension (same as input dimension)

    @property
    def l_output(self):
        return self.dataset_test.l_output
        # this is the dimension the decoder should transfer to
        # is will be [batch-size, l_output, d_output]

    @property
    def forecast_horizon(self):
        return self.dataset_test.forecast_horizon

    def setup(self):
        self.dataset_train = CustomTrafficDataset(
            flag="train",
            pickle_file_name="tensor_data.pkl",
            meta_batch_size=3500,
        )
        # todo: change split to make sure the properties are accessible and correct
        self.split_train_val(0.1)  # be careful, after the splitting the attributes are no longer accessible

        self.dataset_test = CustomTrafficDataset(
            flag="test",
            pickle_file_name="tensor_data.pkl",
            meta_batch_size=700,
        )


class CustomRobotDataset(Dataset):
    def __init__(
            self,
            data_path=None,
            pickle_file_name="SinData.pkl",
            flag="train",
            eval_mask=True,
            **kwargs,
    ):

        super().__init__(**kwargs)
        # init
        assert flag in ["train", "test"]
        type_map = {"train": 0, "test": 1}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.pickle_file_name = pickle_file_name
        if self.data_path is None:
            self.data_path = default_data_path
        self.eval_mask = eval_mask
        self.__read_data__()

    def __read_data__(self):
        # depending on the flag create the train, validation or test set

        # Load the tensor from the file using pickle
        complete_path = os.path.join(self.data_path, self.pickle_file_name)

        with open(complete_path, 'rb') as f:
            self.data_dict = pickle.load(f)
            print("Train Obs Shape", self.data_dict['train_obs'].shape)
            # print("Train Act Shape", self.data_dict['train_act'].shape)
            # print("Train Targets Shape", self.data_dict['train_targets'].shape)
            # print("Test Obs Shape", self.data_dict['test_obs'].shape)
            # print("Test Act Shape", self.data_dict['test_act'].shape)
            # print("Test Targets Shape", self.data_dict['test_targets'].shape)
            print("Normalizer", self.data_dict['normalizer'])

        if self.set_type == 0:  # train
            self.x = torch.cat((self.data_dict['train_obs'], self.data_dict['train_act']), dim=2)

            self.y = self.data_dict['train_targets']
        else:  # test
            self.x = torch.cat((self.data_dict['test_obs'], self.data_dict['test_act']), dim=2)
            self.y = self.data_dict['test_targets']

        print('context batches before cat: ', self.x.shape)
        self.x = torch.cat((self.x, torch.zeros(self.x.shape[0], self.y.shape[1], self.x.shape[-1], dtype=torch.float32)),
                           dim=1)
        print('context batches after cat: ', self.x.shape)


        if self.eval_mask:
            mask = torch.cat((torch.zeros(self.x.shape[1]), torch.ones(self.y.shape[1])), axis=0)
        else:
            mask = torch.cat((torch.zeros(self.x.shape[1]), torch.zeros(self.y.shape[1])), axis=0)
        self.mask = mask[:, None]

        print(self.x.shape)
        print(self.y.shape)
        print(self.mask.shape)

    def __getitem__(self, idx):
        a = self.x[idx]
        b = self.y[idx]
        c = self.mask
        return self.x[idx], self.y[idx], self.mask

    def __len__(self):
        return self.x.shape[0]

    @property
    def d_input(self):
        return self.x.shape[-1]  # i think this maybe has to be the number of input verctors (150, self.x.shape[1]) ???

    # already forgot how the encoding and feeding to the model works...

    @property
    def d_output(self):
        return self.y.shape[-1]

    @property
    def l_output(self):
        return self.y.shape[1] # sollte 900 sein

    # @property
    # def forecast_horizon(self):
    #     return self.y.shape[1] # sollte 900 sein damit decoder auch auf 900 schrumpft


class CustomRobotSequenceDataset(SequenceDataset):
    _name_ = "robot"

    _collate_arg_names = [
        "mask"]  # ["mark", "mask"]  # Names of the two extra tensors that the InformerDataset returns

    @property
    def d_input(self):
        return self.dataset_test.d_input
        # I assume this is the dimension of the data that goes into the encoder, which in this case is
        # feature dimension (1)

    @property
    def d_output(self):
        return self.dataset_test.d_output
        # prediction feature dimension (same as input dimension)

    @property
    def l_output(self):
        return self.dataset_test.l_output
        # this is the dimension the decoder should transfer to
        # is will be [batch-size, l_output, d_output]

    @property
    def forecast_horizon(self):
        return self.dataset_test.forecast_horizon

    def setup(self):
        self.dataset_train = CustomRobotDataset(
            flag="train",
        )
        # todo: change split to make sure the properties are accessible and correct
        self.split_train_val(0.2)  # be careful, after the splitting the attributes are no longer accessible

        self.dataset_test = CustomRobotDataset(
            flag="test",
        )
