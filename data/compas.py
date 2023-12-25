import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, DataGenerator

def _read_data_(fpath):

    data = pd.read_csv(fpath, skipinitialspace=True)
    
    return data

def _data_preprocess_(rootdir=None):
    if rootdir is None:
        rootdir = "dataset/compas"

    filename = os.path.join(rootdir, 'compas-scores-two-years.csv')
    data = _read_data_(filename)

    data = data.loc[
        (data['days_b_screening_arrest'] <= 30) &
        (data['days_b_screening_arrest'] >=-30) &
        (data['is_recid'] != -1) &
        (data['c_charge_degree'] != 'O') &
        (data['score_text'] != 'N/A')
    ]

    data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])
    data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])
    data['length_of_stay'] = (data['c_jail_out'] - data['c_jail_in']).dt.days

    data = data[[
        'c_charge_degree', 'race', 'age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'score_text', 'sex', 'priors_count', 'two_year_recid'
    ]]

    data['score_text'] = data['score_text'].map(lambda x: 0 if x=='Low' else 1)

    categorical_vars = [
        'sex', 'race', 'c_charge_degree', 'age_cat'
    ]
    data = pd.get_dummies(data, columns=categorical_vars)

    cols_to_drop = [
        'sex_Female', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Asian', 'race_Other'
    ]

    # cols_to_drop = [
    #     'id', 'name', 'first', 'last', 'compas_screening_date', 'sex_Female',
    #     'dob', 'age_cat', 'race_Other', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Asian',
    #     'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date',
    #     'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',
    #     'r_charge_desc', 'r_jail_in', 'r_jail_out', 
    # ]
    data.drop(cols_to_drop, axis=1, inplace=True)
    # print(data.columns)

    data.dropna(inplace=True)

    df_X, df_Y = _get_input_output_df_(data)
    return df_X, df_Y

def _get_input_output_df_(data):

    cols = sorted(data.columns)
    output_col = 'two_year_recid'
    input_cols = [col for col in cols if col not in output_col]
    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y

def load_data(protected_vars=None):
    if protected_vars == None:
        protected_vars = ['race_African-American', 'sex_Male']

    df_X, df_Y = _data_preprocess_()
    protected_idx = [df_X.columns.get_loc(x) for x in protected_vars]
    for i, c in enumerate(df_X.columns):
        print(i, c)

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, protected_idx

class Generator(DataGenerator):
    def __init__(self, sensitive_columns, include_protected_feature) -> None:
        self.X, self.y, self.protected_idx = load_data()

        self.continuous_columns = [5, 6, 7, 8, 9, 10, 11]
        self.onehot_ranges = [
            [0, 3],
            [3, 5],
        ]
        self.include_protected_feature = include_protected_feature
        self.feature_name = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'race_Black', 'score_text', 'sex_Male', 'age_catagory', 'crime_charge_degree']

        super()._initialize(sensitive_columns=sensitive_columns)

__all__ = [
    'load_data',
    'Generator'
]