import pickle
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import torch

def load_binary(path):
    with open(f'{path}', 'rb') as rb:
        data = pickle.load(rb)
    return data

name_list = ['kato']

data = HeteroData()

A = load_binary('../data/kato/author/author_1.Matrix')
V = load_binary('../data/kato/conference/conference_1.Matrix')
S = load_binary('../data/kato/term/term_1.Matrix')
Y = load_binary('../data/kato/year/year_1.Matrix')

print(A.shape)
print(V.shape)

# data['key'] = torch.tensor()

