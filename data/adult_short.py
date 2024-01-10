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

    data.drop(['relationship', 'occupation'], axis=1, inplace=True)

    categorical_vars = [
        'marital-status', 'workclass', 
        'race', 'sex', 'native-country'
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

def load_data(sensitive_vars=None):
    if sensitive_vars == None:
        sensitive_vars = ['race_White', 'sex_Male']

    df_X, df_Y = _data_preprocess_()
    sensitive_idx = [df_X.columns.get_loc(x) for x in sensitive_vars]
    # for i, c in enumerate(df_X.columns):
    #     print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, sensitive_idx


class Generator(DataGenerator):
    def __init__(self, include_sensitive_feature, sensitive_vars, device) -> None:
        self.X, self.y, self.sensitive_idx = load_data(sensitive_vars)

        self.continuous_columns = [0, 1, 2, 3, 4, 12, 13]
        self.onehot_ranges = [
            [5, 12],
            [14, 21]
        ]
        self.feature_name = ['age', 'capital-gain', 'capital-loss', 'education-num',
       'hours-per-week', 'race_White', 'sex_Male', 'marital-status', 'workclass']

        super().__init__(include_sensitive_feature, device)

    def data_format(self, data):
        df = self.feature_dataframe(data=data, dtype='float32')
        df['marital-status'] = df['marital-status'].replace({
            0: 'Divorced',
            1: 'Married-AF-spouse',
            2: 'Married-civ-spouse',
            3: 'Married-spouse-absent',
            4: 'Never-married',
            5: 'Separated',
            6: 'Widowed'
        })
        df['race'] = df['race_White'].apply(lambda x: 'Others' if x < 0.5 else 'White')
        df['sex'] = df['sex_Male'].apply(lambda x: 'Female' if x < 0.5 else 'Male')
        df['workclass'] = df['workclass'].replace({
            0 : 'Federal-gov',
            1 : 'Local-gov',
            2 : 'Private',
            3 : 'Self-emp-inc',
            4 : 'Self-emp-not-inc',
            5 : 'State-gov',
            6 : 'Without-pay'
        })
        # json = df[['age', 'capital-gain', 'capital-loss', 'education-num',
        # 'hours-per-week', 'race', 'sex', 'marital-status', 'occupation', 'relationship', 'workclass']].iloc[0].to_dict()
        json = df[['age', 'capital-gain', 'capital-loss', 'education-num',
        'hours-per-week', 'race', 'sex', 'marital-status', 'workclass']].iloc[0].to_dict()
        features = []
        for k, v in json.items():
            features.append(f'{k} is {int(v) if isinstance(v, float) else v}')
        return ', '.join(features)
    
__all__ = [
    'load_data',
    'Generator'
]