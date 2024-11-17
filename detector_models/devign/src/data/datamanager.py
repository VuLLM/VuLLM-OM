import glob

import pandas as pd
import numpy as np
import os
import src.utils.functions.parse as parse

from os import listdir
from os.path import isfile, join
from src.utils.objects.input_dataset import InputDataset
# from sklearn.model_selection import train_test_split


def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
    # dataset.info()
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)


def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    return data_frame.rename(columns={old: new})


def tokenize(data_frame: pd.DataFrame):
    data_frame.code = data_frame.code.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, 'code', 'tokens')
    # Keep just the tokens
    return data_frame[["tokens"]]


def to_files(data_frame: pd.DataFrame, out_path):
    os.makedirs(out_path, exist_ok=True)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.code)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    train = data_frame[data_frame.test.apply(lambda x: x == [0] if isinstance(x, list) else x == 0)]
    train = train.sample(frac=1).reset_index(drop=True)
    test = data_frame[data_frame.test == 1]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test)


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f)) and f.endswith(".pkl")])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])
    i = 0
    for ds_file in data_sets_files:
        print(i)
        dataset = dataset.append(load(data_sets_dir, ds_file))
        i += 1

    return dataset


def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="code", keep='first')


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)