import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator

# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
# The classification goal is to predict if the client will subscribe a term deposit (variable y).

def _read_data_(fpath):

    data = pd.read_excel(fpath, header=0, skiprows=[0])
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/loans_default"

    filename = os.path.join(rootdir, 'default_of_credit_card_clients.xls')
    data = _read_data_(filename)
    
    data['SEX'] = data['SEX'].replace({1:0, 2:1})

    cols_to_drop = [
        'ID'
    ]
    data.drop(cols_to_drop, axis=1, inplace=True)

    df_X, df_Y = _get_input_output_df_(data)
    return df_X, df_Y

def _get_input_output_df_(data):

    cols = sorted(data.columns)
    output_col = 'default payment next month'
    input_cols = [col for col in cols if col not in output_col]

    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y

def load_data(sensitive_vars=None):
    if sensitive_vars == None:
        sensitive_vars = ['SEX']

    df_X, df_Y = _data_preprocess_()
    sensitive_idx = [df_X.columns.get_loc(x) for x in sensitive_vars]
    # for i, c in enumerate(df_X.columns):
    #     print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, sensitive_idx

class Generator(DataGenerator):
    def __init__(self, include_sensitive_feature, sensitive_vars, device) -> None:
        self.X, self.y, self.sensitive_idx = load_data(sensitive_vars)

        self.continuous_columns = [i for i in range(self.X.shape[1])]
        self.onehot_ranges = []
        self.feature_name = [
            'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'EDUCATION', 'LIMIT_BAL', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX'
        ]

        super().__init__(include_sensitive_feature, device)

__all__ = [
    'load_data',
    'Generator'
]