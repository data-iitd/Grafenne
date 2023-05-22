from torch_geometric.datasets import Reddit, Amazon,Planetoid, Actor, WikipediaNetwork, WebKB
import os
import torch
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data, InMemoryDataset

import tqdm
import torch.nn.functional as F
import numpy as np
import random
import copy
import argparse
import random
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
src_dir = os.path.dirname(__file__)
import sys
sys.path.append(src_dir)
from musae_data_loader import load_musae_data
from mimic_data_loader import load_mimic_data
sys.path.append(os.path.dirname(src_dir))
from seed import seed
import torch

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper
def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]
def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes
def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))
def keep_only_largest_connected_component(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))
    
    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data

    return dataset
def seed_everything(seed: int):

    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed)


def load_data(dataset_name,musae_dataset_path=None,train_ratio=0.4,val_ratio=0.3):
    if dataset_name == 'Cora':
        path = os.path.join(os.getcwd(), 'data','Cora')
        dataset = Planetoid(path,name='Cora')
        data = dataset[0]
    elif dataset_name == 'CiteSeer':
        path = os.path.join(os.getcwd(), 'data','CiteSeer')
        dataset = Planetoid(path,name='CiteSeer')
        data = dataset[0]
    elif dataset_name == 'Reddit':
        path = os.path.join(os.getcwd(), 'data', 'Reddit')
        dataset = Reddit(path)
        data = dataset[0]
    elif dataset_name == 'Actor':
        path = os.path.join(os.getcwd(), 'data', 'Actor')
        dataset = Actor(path)
        data = dataset[0]
        
    elif dataset_name == 'Chameleon':
        path = os.path.join(os.getcwd(), 'data', 'Chameleon')
        dataset = WikipediaNetwork(path,'Chameleon')
        data = dataset[0]
        
    elif dataset_name == 'Squirrel':
        path = os.path.join(os.getcwd(), 'data', 'Squirrel')
        dataset = WikipediaNetwork(path,'Squirrel')
        data = dataset[0]
        
    elif dataset_name == 'Cornell':
        path = os.path.join(os.getcwd(), 'data', 'Cornell')
        dataset = WebKB(path,'Cornell')
        data = dataset[0]
        
    elif dataset_name == 'Wisconsin':
        path = os.path.join(os.getcwd(), 'data', 'Wisconsin')
        dataset = WebKB(path,'Wisconsin')
        dataset = keep_only_largest_connected_component(dataset)
        data = dataset[0]

    elif dataset_name == 'Texas':
        path = os.path.join(os.getcwd(), 'data', 'Texas')
        dataset = WebKB(path,'Texas')
        data = dataset[0]
        
    elif dataset_name == 'Photo':
        path = os.path.join(os.getcwd(), 'data', 'Wisconsin')
        dataset = Amazon(path, dataset_name)
        #dataset = keep_only_largest_connected_component(dataset)
        data = dataset.data
        data = Data(
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=torch.zeros(data.y.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(data.y.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(data.y.size()[0], dtype=torch.bool)
        )
        
    elif dataset_name in ['DE','ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW','facebook','git','ZHTW']:
        data = load_musae_data(dataset_name)
        
    elif dataset_name == 'mimic3_icd':
        data = load_mimic_data('icd')
    elif dataset_name == 'mimic3_expire':
        data = load_mimic_data('expire')   
    else:
        import sys
        print(f'dataset {dataset_name} is not defined')
        sys.exit(0)

    list_all_ids = list(range(0,len(data.train_mask)))
    random.shuffle(list_all_ids)
    val_ratio = train_ratio+val_ratio
    split_train = list_all_ids[0:int(train_ratio*len(list_all_ids))]
    split_val = list_all_ids[int(train_ratio*len(list_all_ids)): int(val_ratio*len(list_all_ids))]
    split_test = list_all_ids[int(val_ratio*len(list_all_ids)): ]


    data.train_mask =torch.Tensor([False]*len(list_all_ids))
    data.val_mask =torch.Tensor([False]*len(list_all_ids))
    data.test_mask =torch.Tensor([False]*len(list_all_ids))

    data.train_mask[split_train]=True
    data.val_mask[split_val]=True
    data.test_mask[split_test]=True

    data.train_mask=data.train_mask.to(torch.bool)
    data.val_mask=data.val_mask.to(torch.bool)
    data.test_mask=data.test_mask.to(torch.bool)
    return data
