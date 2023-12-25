import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator
import random
from sklearn.preprocessing import StandardScaler

# Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

def _read_data_(fpath, train_or_test):

    names = [
        'age', 'workclass', 'fnlwgt', 'education', 
        'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 
        'capital-loss', 'hours-per-week', 'native-country',
        'annual-income'
    ]

    if train_or_test == 'train':
        data = pd.read_csv(
            fpath, sep=',', header=None, names=names,
            na_values=['?'], skipinitialspace=True
        )
    elif train_or_test == 'test':
        data = pd.read_csv(
            fpath, sep=',', header=None, names=names,
            na_values=['?'], skiprows=1, skipinitialspace=True
        )
        data['annual-income'] = data['annual-income'].str.rstrip('.')

    data['annual-income'] = data['annual-income'].replace({'<=50K': 0, '>50K': 1})
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/adult"

    filename = lambda x: os.path.join(rootdir, f'{x}.csv')
    
    train_data = _read_data_(filename('train'), 'train')
    test_data = _read_data_(filename('test'), 'test')

    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # remove rows with NaNs
    data.dropna(inplace=True)

    categorical_vars = [
        'workclass', 'marital-status', 'occupation', 
        'relationship', 'race', 'sex', 'native-country'
    ]

    data = pd.get_dummies(data, columns=categorical_vars)

    cols_to_drop = [
        'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
        'race_Other', 'sex_Female', 'native-country_Cambodia', 'native-country_Canada', 
        'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 
        'native-country_Dominican-Republic', 'native-country_Ecuador', 
        'native-country_El-Salvador', 'native-country_England', 'native-country_France', 
        'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 
        'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 
        'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 
        'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 
        'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 
        'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 
        'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 
        'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 
        'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia',
        'fnlwgt', 'education'
    ]

    data.drop(cols_to_drop, axis=1, inplace=True)

    # # Standardize continuous columns
    # continuous_vars = [
    #     'age', 'education-num', 'capital-gain', 
    #     'capital-loss', 'hours-per-week'
    # ]
    # scaler = StandardScaler().fit(data[continuous_vars])
    # data[continuous_vars] = scaler.transform(data[continuous_vars])

    df_X, df_Y = _get_input_output_df_(data)
    return df_X, df_Y

def _get_input_output_df_(data):

    cols = sorted(data.columns)
    output_col = 'annual-income'
    input_cols = [col for col in cols if col not in output_col]

    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y

def load_data(protected_vars=None):
    if protected_vars == None:
        protected_vars = ['race_White', 'sex_Male']

    df_X, df_Y = _data_preprocess_()
    protected_idx = [df_X.columns.get_loc(x) for x in protected_vars]

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, protected_idx


class Generator(DataGenerator):
    def __init__(self, sensitive_columns, include_protected_feature) -> None:
        self.X, self.y, self.protected_idx = load_data()

        self.continuous_columns = [0, 1, 2, 3, 4, 26, 33]
        self.onehot_ranges = [
            [5, 12],
            [12, 26],
            [27, 33],
            [34, 41]
        ]
        self.include_protected_feature = include_protected_feature
        self.feature_name = ['age', 'capital-gain', 'capital-loss', 'education-num',
       'hours-per-week', 'race_White', 'sex_Male', 'marital-status', 'occupation', 'relationship', 'workclass']

        super()._initialize(sensitive_columns=sensitive_columns)

    # def _data2feature(self, data):
    #     if len(data.shape) == 1:
    #         data = data.unsqueeze(dim=0)

    #     feature1 = data[:, self.continuous_columns]
    #     feature2 = data[:, self.onehot_ranges[0][0]:self.onehot_ranges[0][1]]
    #     feature3 = data[:, self.onehot_ranges[1][0]:self.onehot_ranges[1][1]]
    #     feature4 = data[:, self.onehot_ranges[2][0]:self.onehot_ranges[2][1]]
    #     feature5 = data[:, self.onehot_ranges[3][0]:self.onehot_ranges[3][1]]

    #     features = [feature1]
    #     for x in [feature2, feature3, feature4, feature5]:
    #         x = onehot_to_idx(x)
    #         features.append(x)
    #     features = torch.concat(features, dim=1)
    #     return features
    
    # def _feature2data(self, feature):
    #     if len(feature.shape) == 1:
    #         feature = feature.unsqueeze(dim=0)
        
    #     data = []
    #     data.append(feature[:, [0, 1, 2, 3, 4]])
    #     data.append(idx_to_onehot(feature[:, 7], 7))
    #     data.append(idx_to_onehot(feature[:, 8], 14))
    #     data.append(feature[:, 5].unsqueeze(0).T)
    #     data.append(idx_to_onehot(feature[:, 9], 6))
    #     data.append(feature[:, 6].unsqueeze(0).T)
    #     data.append(idx_to_onehot(feature[:, 10], 7))
    #     data = torch.concat(data, dim=1)
    #     return data
    
__all__ = [
    'load_data',
    'Generator'
]