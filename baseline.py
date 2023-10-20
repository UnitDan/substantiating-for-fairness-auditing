import torch
import random
from seeker import Random_select_seeker, Random_gen_seeker, White_seeker, Black_seeker
from utils import Unfair_metric, load_model
from data import adult
from train_dnn import get_data
from dnn_models.model import MLP
from distances import BinaryDistance, NormalizedSquaredEuclideanDistance
import argparse
import os
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
    parser.add_argument('--seeker', type=str, choices=['dx_select', 'select', 'distribution', 'range', 'white', 'black'])
    parser.add_argument('--dnn_model_id', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--use_protected_attr', action='store_true')
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--max_query', type=int, default=100000)
    parser.add_argument('--repeat', type=int, default=1000)
    args = parser.parse_args()

    rand_seed = args.dnn_model_id
    use_protected_attr = args.use_protected_attr

    print('preparing data and model...')
    dataset, train_dl, test_dl = get_data(adult, rand_seed)
    dataset.use_protected_attr = use_protected_attr
    in_dim = dataset.dim_feature()
    out_dim = 2

    model = MLP(in_dim, out_dim)
    load_model(model, 'MLP', 'adult', 'STDTrainer', use_protected_attr=use_protected_attr, id=rand_seed)

    # prepare data
    all_X, all_y = dataset.data, dataset.labels
    all_pred = model.get_prediction(all_X)

    adult_gen = adult.Adult_gen(include_protected_feature=use_protected_attr)

    print('preparing distances...')
    # prepare distances
    distance_x_NSE = NormalizedSquaredEuclideanDistance()
    distance_y = BinaryDistance()

    distance_x_NSE.fit(num_dims=dataset.dim_feature(), data_gen=adult_gen)

    epsilon = 1
    unfair_metric = Unfair_metric(dx=distance_x_NSE, dy=distance_y, epsilon=epsilon)

    log_path = f'log/{args.log_path}.log'
    logger = Logger(log_path)
    logger.init(f'seeker: {args.seeker}, model_id: {args.dnn_model_id}, use_protected_attr: {args.use_protected_attr}, max_query: {args.max_query}, repeat: {args.repeat}')

    print('start')
    random.seed(422)
    torch.manual_seed(422)
    if args.seeker == 'dx_select':
        seeker = Random_select_seeker(model=model, unfair_metric=unfair_metric, data=all_X)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(dx_constraint=True, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'select':
        seeker = Random_select_seeker(model=model, unfair_metric=unfair_metric, data=all_X)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(dx_constraint=False, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'distribution':
        seeker = Random_gen_seeker(model=model, unfair_metric=unfair_metric, data_gen=adult_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(by_range=False, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'range':
        seeker = Random_gen_seeker(model=model, unfair_metric=unfair_metric, data_gen=adult_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(by_range=True, max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'white':
        seeker = White_seeker(model=model, unfair_metric=unfair_metric, data_gen=adult_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)
    elif args.seeker == 'black':
        seeker = Black_seeker(model=model, unfair_metric=unfair_metric, data_gen=adult_gen)
        for _ in range(args.repeat):
            pair, n_query = seeker.seek(max_query=args.max_query)
            if pair != None:
                logger.log(1, n_query, pair.int().tolist())
            else:
                logger.log(0, n_query, None)