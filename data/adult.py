import os
import pandas as pd
import torch
from data.data_utils import convert_df_to_tensor, onehot_to_idx, idx_to_onehot, Data_gen
import random
from sklearn.preprocessing import StandardScaler


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

def data_preprocess(rootdir=None):
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

def load_data():
    df_X, df_Y = data_preprocess()
    protected_vars = ['race_White', 'sex_Male']
    protected_idx = [df_X.columns.get_loc(x) for x in protected_vars]

    X, y = convert_df_to_tensor(df_X, df_Y)

    return X, y, protected_idx


def get_original_feature(x):
    if len(x.shape) == 1:
        x = x.unsqueeze(dim=0)

    x1 = x[:, [0, 1, 2, 3, 4, 26, 33]]
    x2 = x[:, 5:12]
    x3 = x[:, 12:26]
    x4 = x[:, 27:33]
    x5 = x[:, 34:]

    new_x = [x1]
    for xx in [x2, x3, x4, x5]:
        xx = onehot_to_idx(xx)
        new_x.append(xx)
    new_x = torch.concat(new_x, dim=1).detach().numpy()
    columns = ['age', 'capital-gain', 'capital-loss', 'education-num',
       'hours-per-week', 'race_White', 'sex_Male', 'marital-status', 'occupation', 'relationship', 'workclass']
    features = pd.DataFrame(new_x, columns=columns)
    return features


def generate_from_origin(age, capital_gain, capital_loss, education_num, hours_per_week, race_white, sex_male, marital_status, occupation, relationship, workclass):
    def _x(x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0).T
        return x
    def _onehot(x, n_choice):
        return idx_to_onehot(x, n_choice)
    
    x = []
    x.append(_x(age))
    x.append(_x(capital_gain))
    x.append(_x(capital_loss))
    x.append(_x(education_num))
    x.append(_x(hours_per_week))
    x.append(_onehot(marital_status, 7))
    x.append(_onehot(occupation, 14))
    x.append(_x(race_white))
    x.append(_onehot(relationship, 6))
    x.append(_x(sex_male))
    x.append(_onehot(workclass, 7))
    x = torch.concat(x, dim=1)
    return x


def _get_input_output_df_(data):

    cols = sorted(data.columns)
    output_col = 'annual-income'
    input_cols = [col for col in cols if col not in output_col]

    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y


class Adult_gen(Data_gen):
    # static variables
    continous_columns = [0, 1, 2, 3, 4, 26, 33]
    sensitive_columns = [26, 33]
    onehot_ranges = [
        [5, 12],
        [12, 26],
        [27, 33],
        [34, 41]
    ]

    def __init__(self, include_protected_feature) -> None:
        self.X, self.y, self.protected_idx = load_data()
        self.include_protected_feature = include_protected_feature
        self.columns_to_keep = [i for i in range(self.X.shape[1]) if i not in self.sensitive_columns]
        self.data_range = torch.quantile(self.X, torch.Tensor([0, 1]), dim=0)
        
        self.all_features = self._data2feature(self.X)
        self.range = torch.quantile(self.all_features, torch.Tensor([0, 1]), dim=0)

    @staticmethod
    def _data2feature(x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        x1 = x[:, Adult_gen.continous_columns]
        x2 = x[:, Adult_gen.onehot_ranges[0][0]:Adult_gen.onehot_ranges[0][1]]
        x3 = x[:, Adult_gen.onehot_ranges[1][0]:Adult_gen.onehot_ranges[1][1]]
        x4 = x[:, Adult_gen.onehot_ranges[2][0]:Adult_gen.onehot_ranges[2][1]]
        x5 = x[:, Adult_gen.onehot_ranges[3][0]:Adult_gen.onehot_ranges[3][1]]

        new_x = [x1]
        for xx in [x2, x3, x4, x5]:
            xx = onehot_to_idx(xx)
            new_x.append(xx)
        new_x = torch.concat(new_x, dim=1)
        return new_x
    
    @staticmethod
    def _feature2data(x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        
        new_x = []
        new_x.append(x[:, [0, 1, 2, 3, 4]])
        new_x.append(idx_to_onehot(x[:, 7], 7))
        new_x.append(idx_to_onehot(x[:, 8], 14))
        new_x.append(x[:, 5].unsqueeze(0).T)
        new_x.append(idx_to_onehot(x[:, 9], 6))
        new_x.append(x[:, 6].unsqueeze(0).T)
        new_x.append(idx_to_onehot(x[:, 10], 7))
        new_x = torch.concat(new_x, dim=1)
        return new_x
    
    def gen_by_range(self, n=1):
        l, u = self.range[0], self.range[1]
        xs = torch.rand((n, self.all_features.shape[1]))
        xs = (l + xs*(u - l)).to(torch.int)
        xs = self._feature2data(xs)

        if not self.include_protected_feature:
            xs = xs[:, self.columns_to_keep]

        return xs
    
    def gen_by_distribution(self, n=1):
        idxs = torch.randint(self.all_features.shape[0], (n, self.all_features.shape[1]))
        xs = []
        for i in range(n):
            x = self.all_features[idxs[i], torch.arange(self.all_features.shape[1])]
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        xs = self._feature2data(xs)

        if not self.include_protected_feature:
            xs = xs[:, self.columns_to_keep]

        return xs
    
    def clip(self, x, with_protected_feature=None):
        if with_protected_feature is None:
            with_protected_feature = self.include_protected_feature
        def _onehot(x):
            o = torch.zeros_like(x)
            o[torch.arange(x.shape[0]), torch.argmax(x, dim=1)] = 1
            return o
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if not with_protected_feature:
            xx = torch.zeros((x.shape[0], x.shape[1] + len(Adult_gen.sensitive_columns)))
            xx[:, self.columns_to_keep] = x
            x = xx
        
        for r in Adult_gen.onehot_ranges:
            x[:, r[0]: r[1]] = _onehot(x[:, r[0]: r[1]])
        contious_low, contious_high = self.data_range[0][Adult_gen.continous_columns], self.data_range[1][Adult_gen.continous_columns]
        x[:, Adult_gen.continous_columns] = x[:, Adult_gen.continous_columns].clip(contious_low, contious_high)
        x = torch.round(x)

        if not with_protected_feature:
            x = x[:, self.columns_to_keep]

        return x

    def perturb_within_epsilon(self, x, dx=None, epsilon=None):
        def _gen_new(x):
            l, u = self.range[0], self.range[1]
            feature_x = self._data2feature(x)

            new_ = []
            for i in range(feature_x.shape[1]):
                if feature_x[0][i] < u[i]:
                    p = torch.zeros_like(feature_x[0])
                    p[i] = 1
                    new_.append((feature_x + p))
                    # new_.append((feature_x + p).int())
                if feature_x[0][i] > l[i]:
                    p = torch.zeros_like(feature_x[0])
                    p[i] = -1
                    new_.append((feature_x + p))
                    # new_.append((feature_x + p).int())
            # print('-----------------------')
            # print(feature_x.int())
            # print()
            # print(*[i for i in new_], sep='\n')
            # print('-----------------------')
            return self._feature2data(torch.concat(new_, dim=0))
        x_pert = _gen_new(x)
        distances = dx(x_pert, x, itemwise_dist=False)
        # print(distances)
        # print(distances < 1/epsilon)
        id_to_choose = torch.where(distances < 1/epsilon)[0]
        return x_pert[id_to_choose]
            

    def random_perturb(self, x, dx=None, epsilon=None):
        def _gen_new(x):
            l, u = self.range[0], self.range[1]
            feature_new = torch.rand((1, self.all_features.shape[1]))
            feature_new = (l + feature_new*(u - l)).to(torch.int)

            feature_x = self._data2feature(x)
            idx = random.randint(0, len(feature_x))
            x_pert = feature_x.repeat(feature_x.shape[1], 1)
            for idx in range(feature_x.shape[1]):
                if x_pert[idx][idx] < feature_new[0][idx]:
                    x_pert[idx][idx] += 1
                elif x_pert[idx][idx] > feature_new[0][idx]:
                    x_pert[idx][idx] -= 1
            x_pert = self._feature2data(x_pert)
            return x_pert

        x_pert = _gen_new(x)
        distances = dx(x_pert, x, itemwise_dist=False)
        id_to_choose = torch.where(distances < 1/epsilon)[0]
        return x_pert[id_to_choose]
    
    def data_around(self, x_sample):
        x_sample = x_sample.squeeze()

        ceil = torch.ceil(x_sample[self.continous_columns])
        floor = torch.floor(x_sample[self.continous_columns])
        cont_matrix = torch.concat([ceil, floor]).view(2, -1)
        choices = [torch.unique(cont_matrix[:, i]) for i in range(cont_matrix.shape[1])]
        for r in self.onehot_ranges:
            x_onehot = x_sample[r[0]: r[1]]
            choices.append(torch.where(x_onehot)[0].float())
        # print('-------------------------------------------------------------------------')
        # print(choices)
        features = torch.cartesian_prod(*choices)
        # print(len(features))
        # print('-------------------------------------------------------------------------')
        datas = self._feature2data(features)
        return datas
            