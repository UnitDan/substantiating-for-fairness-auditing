import torch
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.auto import tqdm

def load_model(model, model_name, dataset_name, trainer_name, use_protected_attr, id):
    root_dir = 'models_to_test'
    file_name = f'{model_name}_{dataset_name}_{trainer_name}_{"protected" if not use_protected_attr else "no"}_{id}.pth'
    model.load(os.path.join(root_dir, file_name))

class Unfair_metric():
    def __init__(self, dx, dy, epsilon) -> None:
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
    
    def is_unfair(self, x1, x2, y1, y2):
        return (self.dy(y1, y2).item() > self.dx(x1, x2).item()*self.epsilon)
    
def get_L_matrix(all_X, all_pred, dx, dy):
    ds = TensorDataset(all_X, all_pred)
    dl = DataLoader(ds, batch_size=500, shuffle=False)
    L = []
    for b in tqdm(dl):
        dxs = dx(b[0], all_X, itemwise_dist=False)
        dys = dy(b[1], all_pred, itemwise_dist=False)

        L_batch = (dys/dxs).squeeze()
        L.append(L_batch)
    L = torch.concat(L, dim=0)
    return L