import pickle
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import torch
import numpy as np

def load_binary(path):
    with open(f'{path}', 'rb') as rb:
        data = pickle.load(rb)
    return data

def createMetaPathBasedAdj(m1, m2):
    adj = np.matmul(m1, m2)
    adj[adj > 1] = 1
    return adj

name_list = ['kato']

# author行列
A = load_binary('../data/kato/author/author_1.Matrix').toarray()
# venue行列
V = load_binary('../data/kato/conference/conference_1.Matrix').toarray()
# keyword行列
K = load_binary('../data/kato/term/term_1.Matrix').toarray()
# year行列
Y = load_binary('../data/kato/year/year_1.Matrix').toarray()

PAP = createMetaPathBasedAdj(A, A.T)
APA = createMetaPathBasedAdj(A.T, A)

data = HeteroData()
data['author'] = torch.tensor(A)
data['venue'] = torch.tensor(V)
data['keyword'] = torch.tensor(K)
data['year'] = torch.tensor(Y)

print(data)
