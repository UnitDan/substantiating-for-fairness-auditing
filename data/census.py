import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator
import random
from sklearn.preprocessing import StandardScaler

# This data set contains weighted census data extracted from the 1994 and 1995 current population surveys conducted by the U.S. Census Bureau.

def _read_data_(fpath):

    names = [
        'age', 'workclass', 'detailed-industry-recode', 'detailed-occupation-recode',
        'education', 'wage-per-hour', 'enroll-in-edu-inst-last-wk', 'marital', 'major-industry-code',
        'major-occupation-code', 'race', 'hispanic-origin', 'sex', 'labor-union',
        'unemployment_reason', 'employment', 'capital-gains', 'capital-losses',
        'dividends-from-stocks', 'tax-filer', 'previous-region', 'state-of-previous-residence',
        'detailed-household-and-family-stat', 'detailed-household-summary-in-household', 
        'instance-weight', 'migration-code-change-in-msa', 'migration-code-change-in-reg',
        'migration-code-move-within-reg', 'house-1-year', 'migration-prev-res-in-sunbelt',
        'persons-worked-for', 'family-members-under-18', 'country-of-birth-father',
        'country-of-birth-mother', 'country', 'citizenship', 'self-employed',
        'fill-inc-questionnaire-for-veteran-admin', 'veterans-benefits', 'weeks-worked-in-year', 'year', 'y'
    ]

    data = pd.read_csv(
        fpath, names=names, na_values=['?'], skipinitialspace=True
    )

    data['y'] = data['y'].replace({'- 50000.': 0, '50000+.': 1})
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/census"

    filename = lambda x: os.path.join(rootdir, f'census-income.{x}')
    
    train_data = _read_data_(filename('data'))
    test_data = _read_data_(filename('test'))

    data = pd.concat([train_data, test_data], ignore_index=True)

    data = data[data['family-members-under-18']=='Not in universe']

    data.drop([
        'detailed-industry-recode', 'detailed-occupation-recode', 'education', 'enroll-in-edu-inst-last-wk', 'major-industry-code',
        'major-occupation-code', 'hispanic-origin', 'state-of-previous-residence', 'detailed-household-summary-in-household',
        'detailed-household-and-family-stat', 'instance-weight', 'migration-code-change-in-msa', 'migration-code-change-in-reg',
        'migration-code-move-within-reg', 'migration-prev-res-in-sunbelt', 'family-members-under-18', 'country-of-birth-father', 'country-of-birth-mother',
        'citizenship', 'fill-inc-questionnaire-for-veteran-admin', 'year'
    ], axis=1, inplace=True)
    
    # remove rows with NaNs
    data.dropna(inplace=True)

    categorical_vars = [
        'workclass', 'marital', 'race', 'sex', 'labor-union', 'unemployment_reason',
        'employment', 'tax-filer', 'previous-region',
        'house-1-year', 'country', 'self-employed',
        'veterans-benefits'
    ]

    data = pd.get_dummies(data, columns=categorical_vars)

    cols_to_drop = [
        'race_Black', 'race_Other', 'race_Amer Indian Aleut or Eskimo', 'race_Asian or Pacific Islander',
        'sex_Female', 'country_Mexico', 'country_Puerto-Rico', 'country_Peru',
        'country_Canada', 'country_South Korea', 'country_India', 'country_Japan',
        'country_Haiti', 'country_El-Salvador', 'country_Dominican-Republic', 'country_Portugal',
        'country_Columbia', 'country_England', 'country_Thailand', 'country_Cuba',
        'country_Laos', 'country_China', 'country_Germany',
        'country_Vietnam', 'country_Italy', 'country_Honduras', 'country_Outlying-U S (Guam USVI etc)',
        'country_Hungary', 'country_Philippines', 'country_Panama', 'country_Poland', 'country_Ecuador',
        'country_Iran', 'country_Guatemala', 'country_Holand-Netherlands', 'country_Taiwan',
        'country_Nicaragua', 'country_France', 'country_Jamaica', 'country_Scotland', 
        'country_Yugoslavia', 'country_Hong Kong', 'country_Trinadad&Tobago', 'country_Greece',
        'country_Cambodia', 'country_Ireland'
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

        self.continuous_columns = [0, 1, 2, 3, 4, 26, 33, 37, 53, 54]
        self.onehot_ranges = [
            [5, 13],
            [13, 16],
            [16, 19],
            [19, 26],
            [27, 33],
            [34, 37],
            [38, 44],
            [44, 50],
            [50, 53],
            [55, 64]
        ]
        self.feature_name = [
            'age', 'capital-gains', 'capital-losses', 'country_US', 'dividends-from-stocks',
            'n_persons-worked-for', 'race_White', 'sex_Male', 'wage-per-hour', 'weeks-worked-in-year',
            'employment', 'house-1-year', 'labor-union', 'marital', 'previous-region', 'self_employed',
            'tax-filer', 'unemployment-reason', 'veterans-benefit', 'workclass'
        ]

        super().__init__(include_sensitive_feature, device)
    
__all__ = [
    'load_data',
    'Generator'
]