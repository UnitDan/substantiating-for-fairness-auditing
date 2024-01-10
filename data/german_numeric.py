import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator

# This dataset classifies people described by a set of attributes as good or bad credit risks.
# We choose to use the data format that contains categorical/symbolic attributes, so that we can get the name of the attributes.

def _read_data_(fpath):
    columns = [str(i+1) for i in range(24)]
    columns.append('y')
    data = pd.read_csv(fpath, sep=' ', skipinitialspace=True, names=columns, index_col=None)
    data['y'] = data['y'].replace({1: 1, 2: 0})
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/german"

    filename = os.path.join(rootdir, 'german.data-numeric')
    data = _read_data_(filename)

    df_X, df_Y = _get_input_output_df_(data)
    return df_X, df_Y

def _get_input_output_df_(data):

    cols = sorted(data.columns)
    output_col = 'y'
    input_cols = [col for col in cols if col not in output_col]

    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y

def load_data(sensitive_vars=None):
    if sensitive_vars == None:
        sensitive_vars = ['1']

    df_X, df_Y = _data_preprocess_()
    sensitive_idx = [df_X.columns.get_loc(x) for x in sensitive_vars]
    # for i, c in enumerate(df_X.columns):
    #     print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, sensitive_idx

class Generator(DataGenerator):
    def __init__(self, include_sensitive_feature, sensitive_vars, device) -> None:
        self.X, self.y, self.sensitive_idx = load_data(sensitive_vars)

        self.continuous_columns = [i for i in range(24)]
        self.onehot_ranges = []
        self.feature_name = [str(i) for i in range(24)]

        super().__init__(include_sensitive_feature, device)

__all__ = [
    'load_data',
    'Generator'
]