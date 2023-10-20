from data.data_utils import ProtectedDataset
from data import adult
import torch
import argparse
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from dnn_models.trainer import STDTrainer
from dnn_models.model import MLP
import os


def get_data(data, rand_seed):
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)

    X, y, protected_idxs = data.load_data()

    # randomly split into train/test splits
    total_samples = len(X)
    train_size = int(total_samples * 0.8)

    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataset = ProtectedDataset(X, y, protected_idxs)
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1000, sampler=test_sampler)

    return dataset, train_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='adult', choices=['adult', 'german', 'bank'])
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP'])
    parser.add_argument('--use_protected_attr', action='store_true')
    parser.add_argument('--trainer', type=str, default='STDTrainer', choices=['STDTrainer'])
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()

    for i in range(args.repeat):
        data_name = globals()[args.data]
        trainer_name = globals()[args.trainer]
        model_name = globals()[args.model]

        dataset, train_dl, test_dl = get_data(data_name, i)
        dataset.use_protected_attr = args.use_protected_attr
        feature_dim = dataset.dim_feature()
        output_dim = 2

        model = model_name(input_size=feature_dim, output_size=output_dim)
        trainer = trainer_name(model, train_dl, test_dl, device='cpu', epochs=10, lr=1e-3)
        trainer.train()

        save_dir = 'models_to_test'
        file_name = f'{args.model}_{args.data}_{args.trainer}_{"protected" if not args.use_protected_attr else "no"}_{i}.pth'
        model.save(os.path.join(save_dir, file_name))
