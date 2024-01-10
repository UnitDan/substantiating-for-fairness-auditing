import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator

# This dataset classifies people described by a set of attributes as good or bad credit risks.
# We choose to use the data format that contains categorical/symbolic attributes, so that we can get the name of the attributes.

def _read_data_(fpath):
    columns = [
        'acount_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'saving_acount', 'employment_since', 
        'installment_rate', 'marital', 'other_debtors', 'residence_since', 'property', 'age', 'installment_plan', 'housing',
        'existing_credits', 'job', 'maintenance', 'telephone', 'foreign_worker', 'y'
    ]
    data = pd.read_table(fpath, sep=' ', skipinitialspace=True, names=columns)
    data['y'] = data['y'].replace({1: 1, 2: 0})
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/german"

    filename = os.path.join(rootdir, 'german.data')
    data = _read_data_(filename)

    data['acount_status'] = data['acount_status'].replace({
        'A11': 1, 'A12': 2, 'A13': 3, 'A14': 4
    })
    
    data['credit_history'] = data['credit_history'].replace({
        'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4
    })

    data['saving_acount'] = data['saving_acount'].replace({
        'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5 
    })

    data['employment_since'] = data['employment_since'].replace({
        'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5
    })

    data['telephone'] = data['telephone'].replace({
        'A191': 0, 'A192': 1
    })

    data['foreign_worker'] = data['foreign_worker'].replace({
        'A201': 1, 'A202': 0
    })

    def assign_sex(row):
        if row['marital'] in ["A91", "A93", "A94"]:
            return 1
        elif row['marital'] in ["A92", "A95"]:
            return 0
        else:
            return None 

    data['sex'] = data.apply(assign_sex, axis=1)

    categorical_vars = [
        'purpose', 'marital', 'other_debtors', 'property', 'installment_plan', 'housing', 'job'
    ]
    data = pd.get_dummies(data, columns=categorical_vars)

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
        sensitive_vars = ['sex']

    df_X, df_Y = _data_preprocess_()
    sensitive_idx = [df_X.columns.get_loc(x) for x in sensitive_vars]
    # for i, c in enumerate(df_X.columns):
    #     print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, sensitive_idx

class Generator(DataGenerator):
    def __init__(self, include_sensitive_feature, sensitive_vars, device) -> None:
        self.X, self.y, self.sensitive_idx = load_data(sensitive_vars)

        self.continuous_columns = [0, 1, 2, 3, 4, 5, 6, 7, 14, 19, 41, 42, 43, 44]
        self.onehot_ranges = [
            [8, 11],
            [11, 14],
            [15, 19],
            [20, 24],
            [24, 27],
            [27, 31],
            [31, 41]
        ]
        self.feature_name = [
            'acount_status', 'age', 'credit_amount', 'credit_history', 'duration', 'employment_sinc', 'existing_credit',
            'foreign_worker', 'installment_rate', 'maintenance', 'residence_since', 'saving_acount', 'sex', 'telephone', 
            'housing', 'installment_plan', 'job', 'marital', 'other_debtors', 'property', 'purpose'
        ]

        super().__init__(include_sensitive_feature, device)

__all__ = [
    'load_data',
    'Generator'
]