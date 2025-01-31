import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator

# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
# The classification goal is to predict if the client will subscribe a term deposit (variable y).

def _read_data_(fpath):

    data = pd.read_table(fpath, sep=';', skipinitialspace=True)
    data['y'] = data['y'].replace({'no': 0, 'yes': 1})
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/bank"

    filename = os.path.join(rootdir, 'bank-full.csv')
    data = _read_data_(filename)
    

    data['month'] = data['month'].replace({
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

    binary_vars = ['default', 'housing', 'loan']
    data[binary_vars] = data[binary_vars].replace({'yes': 1, 'no': 0})

    categorical_vars = [
        'job', 'marital', 'education', 'contact', 'poutcome'
    ]
    data = pd.get_dummies(data, columns=categorical_vars)

    cols_to_drop = [
        'duration'
    ]
    data.drop(cols_to_drop, axis=1, inplace=True)

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
        sensitive_vars = ['age']

    df_X, df_Y = _data_preprocess_()
    sensitive_idx = [df_X.columns.get_loc(x) for x in sensitive_vars]
    # for i, c in enumerate(df_X.columns):
    #     print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, sensitive_idx

class Generator(DataGenerator):
    def __init__(self, include_sensitive_feature, sensitive_vars, device) -> None:
        self.X, self.y, self.sensitive_idx = load_data(sensitive_vars)

        self.continuous_columns = [0, 1, 2, 6, 7, 12, 25, 29, 30, 35]
        self.onehot_ranges = [
            [3, 6],
            [8, 12],
            [13, 25],
            [26, 29],
            [31, 35]
        ]
        self.feature_name = [
            'age', 'balance', 'campaign', 'day', 'default', 'housing', 'loan', 'month', 'pdays', 'previous', 
            'contact', 'education', 'job', 'marital', 'poutcome'
        ]

        super().__init__(include_sensitive_feature, device)

__all__ = [
    'load_data',
    'Generator'
]