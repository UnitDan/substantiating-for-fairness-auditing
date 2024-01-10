from data import adult, german, loans_default, german_numeric, census
from distances.sensitive_subspace_distances import LogisticRegSensitiveSubspace
from inFairness.distances import SquaredEuclideanDistance
from models.trainer import STDTrainer, SenSeiTrainer
from models.model import MLP
from models.metrics import accuracy, consistancy, accuracy_variance
from utils import get_data
from datetime import datetime
import os
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='adult', choices=['adult', 'german', 'loans_default', 'census'])
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_all_features', action='store_true')
    parser.add_argument('--sensitive_vars', nargs='*')
    parser.add_argument('--trainer', type=str, default='std', choices=['std', 'sensei'])
    parser.add_argument('--rho', type=float, default=5)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--save_path', type=str, default='new')
    args = parser.parse_args()

    print('start')

    data_choices = {
        'adult': adult,
        'german': german,
        'loans_default': loans_default,
        'german_numeric': german_numeric,
        'census': census
    }
    data = data_choices[args.data]
    data_gen = data.Generator(args.use_all_features, args.sensitive_vars, args.device)

    for i in range(args.repeat):
        dataset, train_dl, test_dl = get_data(data, i, args.sensitive_vars)
        dataset.use_sensitive_attr = args.use_all_features
        feature_dim = dataset.dim_feature()
        print('feature_dim:', feature_dim)
        output_dim = 2

        model = MLP(input_size=feature_dim, output_size=output_dim, data_gen=data_gen, n_layers=args.n_layers)

        if args.trainer == 'std':
            trainer = STDTrainer(model, train_dl, test_dl, device=args.device, epochs=args.epoch, lr=args.lr)
        if args.trainer == 'sensei':
            
            train_distance_x = LogisticRegSensitiveSubspace()
            train_distance_y = SquaredEuclideanDistance()

            dataset.use_sensitive_attr = True
            all_X_train = []
            for x, _ in train_dl:
                all_X_train.append(x)
            all_X_train = torch.concat(all_X_train, dim=0).to(args.device)
            dataset.use_sensitive_attr = args.use_all_features

            train_distance_x.to(args.device)
            train_distance_y.to(args.device)

            if args.use_all_features:
                train_distance_x.fit(all_X_train, data_gen=data_gen, sensitive_idxs=dataset.sensitive_idxs)
            else:
                sensitive_ = all_X_train[:, dataset.sensitive_idxs]
                no_sensitive = all_X_train[:, [i for i in range(all_X_train.shape[1]) if i not in dataset.sensitive_idxs]]
                train_distance_x.fit(no_sensitive, data_gen=data_gen, data_SensitiveAttrs=sensitive_)
            train_distance_y.fit(output_dim)
            trainer = SenSeiTrainer(model, train_dl, test_dl, device=args.device, epochs=args.epoch, lr=args.lr, distance_x=train_distance_x, distance_y=train_distance_y, rho=args.rho)
        trainer.train()

        # evaluation
        train_ac = accuracy(model, train_dl, args.device)
        test_ac = accuracy(model, test_dl, args.device)

        dataset.use_sensitive_attr = True
        train_X, train_y, test_X, test_y = [], [], [], []
        for x, y in train_dl:
            train_X.append(x)
            train_y.append(y)
        for x, y in test_dl:
            test_X.append(x)
            test_y.append(y)
        train_X = torch.concat(train_X, dim=0)
        train_y = torch.concat(train_y, dim=0)
        test_X = torch.concat(test_X, dim=0)
        test_y = torch.concat(test_y, dim=0)
        train_group = train_X[:, dataset.sensitive_idxs[0]]
        test_group = test_X[:, dataset.sensitive_idxs[0]]
        if args.use_all_features:
            train_X_counter = train_X.clone()
            test_X_counter = test_X.clone()
            train_X_counter[:, dataset.sensitive_idxs[0]] = 1 - train_X_counter[:, dataset.sensitive_idxs[0]]
            test_X_counter[:, dataset.sensitive_idxs[0]] = 1 - test_X_counter[:, dataset.sensitive_idxs[0]]
            train_cons = consistancy(model, train_X, train_X_counter, args.device)
            test_cons = consistancy(model, test_X, test_X_counter, args.device)
            train_acvar = accuracy_variance(model, train_X, train_y, train_group, args.device)
            test_acvar = accuracy_variance(model, test_X, test_y, test_group, args.device)
        else:
            train_cons = -1
            test_cons = -1
            train_X = train_X[:, [i for i in range(train_X.shape[1]) if i not in dataset.sensitive_idxs]]
            test_X = test_X[:, [i for i in range(test_X.shape[1]) if i not in dataset.sensitive_idxs]]
            train_acvar = accuracy_variance(model, train_X, train_y, train_group, args.device)
            test_acvar = accuracy_variance(model, test_X, test_y, test_group, args.device)

        arguments = vars(args)
        print(' | '.join([f"{arg}: {value}" for arg, value in arguments.items()]))
        print(f'Train Accuracy: {train_ac}',
              f'Test Accuracy: {test_ac}',
              f'Train consistency: {train_cons}', 
              f'Test consistency: {test_cons}',
              f'Train accuracy varience: {train_acvar}',
              f'Test accuracy varience: {test_acvar}',
              sep='\n')
        
        if args.save_path != 'no':

            save_dir = os.path.join('trained_models', args.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            with open(os.path.join(save_dir, 'train_history.csv'), 'a+') as f:
                f.seek(0)
                if not f.read(100):
                    f.write('train accuracy, test accuracy, train consistency, test consistency, train acva, test acva, time, data, trainer, sensitive_attr, use_sensitive_attr, seed, note\n')
                f.seek(0, os.SEEK_END)

                current_date_time = datetime.now()
                date = current_date_time.strftime("%Y-%m-%d")
                note = args.note+f'_rho={args.rho}' if args.trainer == 'sensei' else args.note
                log = f'{train_ac}, {test_ac}, {train_cons}, {test_cons}, {train_acvar}, {test_acvar}, {date}, {args.data}, {args.trainer}, {args.sensitive_vars}, {args.use_all_features}, {i}, {note}\n'
                f.write(log)

            file_name = f'MLP_{args.data}_{args.trainer}_{"all-features" if args.use_all_features else "without-"+"-".join(args.sensitive_vars)}_{i}{note}'
            model.save(os.path.join(save_dir, file_name))
