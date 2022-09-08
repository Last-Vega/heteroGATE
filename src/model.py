import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import HeteroConv, GATConv, Linear, to_hetero, GAE
# from torch_geometric.data import HeteroData
# import torch_geometric.transforms as T

class GATE(nn.Module):
	def __init__(self, hidden_dim, out_dim):
		super(GATE,self).__init__()
		self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
		self.lin1 = Linear(-1, hidden_dim)
		self.conv2 = GATConv((-1, -1), out_dim, add_self_loops=False)
		self.lin2 = Linear(-1, out_dim)

	def encoder(self, x, edge_index):
		hidden = self.conv1(x, edge_index, return_attention_weights=True)
		hidden = hidden.relu()
		z = self.conv2(hidden, edge_index, return_attention_weights=True)
		self.z = z
		return z
	
	def decoder(Z):
		A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
		return A_pred

	def forward(self, x, edge_index):
		Z = self.encoder(x, edge_index)
		A_pred = self.decoder(Z)
		return A_pred