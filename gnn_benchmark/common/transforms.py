import torch
from torch_geometric.utils import degree
from torch_geometric import transforms as T
import numpy as np


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def create_one_hot_transform(dataset):
    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())

    return T.OneHotDegree(max_degree)


def create_categorical_degree_transform(dataset):
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]

    deg = torch.cat(degs, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()
    return NormalizedDegree(mean, std)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


class CreatePlanetoidSplit:
    def __init__(self, n_classes, seed, randomize=True, mask_name=None):
        self.randomize = randomize
        self.seed = seed
        self.mask_name = mask_name
        self.n_classes = n_classes

    def __call__(self, data):
        # From https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/train_eval.py
        # Set new random planetoid splits:
        # * 20 * num_classes labels for training
        # * 500 labels for validation
        # * 1000 labels for testing
        if self.randomize:
            random_state = np.random.RandomState(seed=self.seed)
            indices = []
            for i in range(self.n_classes):
                index = (data.y == i).nonzero().view(-1)
                perm = random_state.permutation(index.size(0))
                index = index[perm]
                indices.append(index)

            train_index = torch.cat([i[:20] for i in indices], dim=0)

            rest_index = torch.cat([i[20:] for i in indices], dim=0)
            rest_perm = random_state.permutation(rest_index.size(0))
            rest_index = rest_index[rest_perm]

            data.train_mask = index_to_mask(train_index, size=data.num_nodes)
            data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
            data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
        if self.mask_name is not None:
            data.mask = data[self.mask_name]
        return data
