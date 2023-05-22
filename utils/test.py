import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear

path = osp.join(osp.dirname(osp.realpath('__file__')), '../data/DBLP')
# We initialize conference node features with a single one-vector as feature:
dataset = DBLP(path)
data = dataset[0]
print(data)