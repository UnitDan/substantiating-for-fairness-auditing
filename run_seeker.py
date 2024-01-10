import torch
import random
import argparse
from seeker.random import RandomSelectSeeker, RangeGenSeeker, DistributionGenSeeker
from seeker.gradiant_based import WhiteboxSeeker, BlackboxSeeker
from utils import UnfairMetric, load_model
from data import adult, german, loans_default, census
from train_dnn import get_data
from models.model import MLP
from distances.normalized_mahalanobis_distances import ProtectedSEDistances
from distances.sensitive_subspace_distances import LogisticRegSensitiveSubspace
from distances.binary_distances import BinaryDistance
from datetime import datetime

class Logger:
    def __init__(self, log_path) -> None:
        self.log_path = log_path
    
    def init(self, msg):
        with open(self.log_path, 'w') as file:
            file.write(msg+'\n')
    
    def log(self, *args, **kwargs):
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        msg = [formatted_time]

        for v in args:
            msg.append(str(v))
        for k, v, in kwargs.items():
            msg.append(f'{k}: {str(v)}')
        msg = '\t'.join(msg) + '\n'

        with open(self.log_path, 'a') as file:
            file.write(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeker', type=str, default='black', choices=['select', 'distribution', 'range', 'white', 'black', 'black_'])
    parser.add_argument('--data', type=str, default='adult', choices=['adult', 'german', 'loans_default', 'census'])
    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--trainer', type=str, default='std', choices=['std', 'sensei'])
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--with_sensitive_attr', action='store_true')
    parser.add_argument('--sensitive_vars', nargs='*')
    parser.add_argument('--eps', type=float)
    parser.add_argument('--random_seed', type=int, default=422)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--repeat', type=int, default=1000)
    parser.add_argument('--max_query', type=int, default=5e4)
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    data_choices = {
        'adult': adult,
        'german': german,
        'loans_default': loans_default,
        'census': census
    }
    data = data_choices[args.data]
    data_gen = data.Generator(args.with_sensitive_attr, args.sensitive_vars, 'cpu')

    dataset, train_dl, test_dl = get_data(data, args.model_id, sensitive_vars=args.sensitive_vars)
    dataset.use_sensitive_attr = args.with_sensitive_attr
    in_dim = dataset.dim_feature()
    out_dim = 2

    all_X, all_y = dataset.get_all_data(), dataset.labels

    model = MLP(in_dim, out_dim, data_gen=data_gen, n_layers=4)
    load_model(model, args.data, args.trainer, use_sensitive_attr=args.with_sensitive_attr, \
            sensitive_vars=args.sensitive_vars, id=args.model_id, note=args.note)

    distance_x_Causal = ProtectedSEDistances()
    distance_x_LR = LogisticRegSensitiveSubspace()
    distance_y = BinaryDistance()

    if args.with_sensitive_attr:
        distance_x_Causal.fit(num_dims=dataset.dim_feature(), data_gen=data_gen, sensitive_idx=dataset.sensitive_idxs)
        chosen_dx = distance_x_Causal
    else:
        sensitive_ = dataset.data[:, dataset.sensitive_idxs]
        distance_x_LR.fit(dataset.get_all_data(), data_gen=data_gen, data_SensitiveAttrs=sensitive_)
        chosen_dx = distance_x_LR

    epsilon = args.eps
    unfair_metric = UnfairMetric(dx=chosen_dx, dy=distance_y, epsilon=epsilon)

    log_path = f'log/{args.log_path}.log'
    logger = Logger(log_path)
    arguments = vars(args)
    logger.init(' | '.join([f"{arg}: {value}" for arg, value in arguments.items()]))
    print(' | '.join([f"{arg}: {value}" for arg, value in arguments.items()]))


    if args.seeker == 'select':
        seeker = RandomSelectSeeker(model=model, unfair_metric=unfair_metric, data=dataset.get_all_data(), data_gen=data_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'distribution':
        seeker = DistributionGenSeeker(model=model, unfair_metric=unfair_metric, data_gen=data_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'range':
        seeker = RangeGenSeeker(model=model, unfair_metric=unfair_metric, data_gen=data_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'white':
        seeker = WhiteboxSeeker(model=model, unfair_metric=unfair_metric, data_gen=data_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(lamb=1, origin_lr=0.1, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'black':
        seeker = BlackboxSeeker(model=model, unfair_metric=unfair_metric, data_gen=data_gen, easy=False)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(lamb=1, origin_lr=0.1, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'black_':
        seeker = BlackboxSeeker(model=model, unfair_metric=unfair_metric, data_gen=data_gen, easy=True)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(lamb=1, origin_lr=0.1, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)