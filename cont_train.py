#
import os
import sys
import torch
from torch_geometric.loader import NeighborSampler
import tqdm
import torch_sparse
from torch_geometric.nn import GINConv,SAGEConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
import numpy as np
import random
import copy
import argparse
src_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_dir)
from utils.data_loader import load_data
from utils.utils import seed_everything,create_otf_edges,get_feature_mask
from models.fognn_scale import ScalableFOGNN as FOGNN
from feature_propagation import FeaturePropagation
import pickle
import gc 
from torch_geometric.data import Data
import collections

    
class GIN(torch.nn.Module):
    def __init__(self, drop_rate, num_node_features, num_layers, hidden_size, out_channels, dropout=0.2):
        super(GIN, self).__init__()
        self.drop_rate = drop_rate
        self.num_node_features = num_node_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.dropout = 0.2
        self.gin_convs = torch.nn.ModuleList()
        self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        self.gin_convs.append(SAGEConv(self.hidden_size, self.out_channels))
        self.projects =torch.nn.ModuleList()
        self.projects.append(Sequential(
                    Linear(in_features=self.num_node_features, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size),
                    ReLU(),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU()
                ))
        self.projects.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size),
                    ReLU(),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU()
                ))


    def forward(self, x_batch, adjs):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_batch = self.projects[i](x_batch)
            x_target = x_batch[:size[1]]
            x_batch = self.gin_convs[i]((x_batch, x_target), edge_index)
            x_batch = F.elu(x_batch)
        out = F.log_softmax(x_batch, dim=-1)
        return out

    def inference(self, x_all,graph_loader):

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in graph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = self.projects[i](x)
                x_target = x[:size[1]]
                x = self.gin_convs[i]((x, x_target), edge_index)
                x = F.elu(x)
                
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    
def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0
    return mean_neighborhood_features

def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)

seed_everything(0)


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="name of the dataset",
                    type=str)
parser.add_argument("--gpu",help="GPU no. to use, -1 in case of no gpu", type=int)
parser.add_argument("--missing_rate",help="% of features to be missed randomly", type=float)
parser.add_argument("--categorical",default=False,help="Make edges only when feature is present/categorical", type=bool)
parser.add_argument("--verbose",default=False,help="Print Model output during training", type=bool)
parser.add_argument("--num_epochs",default=200,help="Print Model output during training", type=int)
parser.add_argument("--num_layers",default=1,help="Num of layers (1,2)", type=int)
parser.add_argument("--bs_train_nbd",default=512,help="Num of nodes in training computation subgraph", type=int)
parser.add_argument("--bs_test_nbd",default=-1,help="Num of nodes in testing computation subgraph", type=int)
parser.add_argument("--drop_rate",default=0.2,help="Drop rate", type=float)
parser.add_argument("--result_file",type=str,default="")
parser.add_argument("--edge_value_thresh",default=0.01,type=float)
parser.add_argument("--imputation",default='zero',type=str)
parser.add_argument("--heads",default=4,type=int)
parser.add_argument("--weight_decay",default=0,type=float)
parser.add_argument("--otf_sample",default=0,type=int)
parser.add_argument("--fto_sample",default=0,type=int)
parser.add_argument("--num_obs_samples",default=30,type=int)
parser.add_argument("--num_feat_samples",default=30,type=int)
parser.add_argument("--use_data_x_otf",default=0,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining
parser.add_argument("--use_data_x_fto",default=0,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining
parser.add_argument("--otf_sample_testing",default=0,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining
parser.add_argument("--sampling_in_loop",default=0,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining

parser.add_argument("--memory_size",default=300,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining
parser.add_argument("--ewc",default=100000,type=int)  ### 
parser.add_argument("--hidden_size",default=256,type=int)  ### If this is on then during samples, nbrs having values as 1 will be selected first and uniform sampling from remaining
parser.add_argument("--lr",default=0.0005,type=float)  
parser.add_argument("--lro",default=0.001,type=float)  

parser.add_argument("--node_prob",default=0.04,type=float)  # node select prob
parser.add_argument("--edge_prob",default=0.0005,type=float)  # edge select prob


parser.add_argument("--del_prob",default=0.4,type=float) #del feat prob
parser.add_argument("--feat_add_delete_rate",default=0.05,type=float) #feat add prob


args = parser.parse_args()
num_epochs = args.num_epochs
gpu = int(args.gpu)
dataset_name = args.data
missing_rate = args.missing_rate
categorical = args.categorical
verbose = args.verbose
num_layers = args.num_layers
bs_train_nbd = args.bs_train_nbd
bs_test_nbd = args.bs_test_nbd
drop_rate = args.drop_rate
result_file = args.result_file
edge_value_thresh = args.edge_value_thresh
imputation_method = args.imputation
heads = args.heads
weight_decay = args.weight_decay
otf_sample = args.otf_sample
fto_sample = args.fto_sample
num_feat_samples = args.num_feat_samples
num_obs_samples = args.num_obs_samples
use_data_x_otf = args.use_data_x_otf
use_data_x_fto = args.use_data_x_fto
otf_sample_testing = args.otf_sample_testing
sampling_in_loop = args.sampling_in_loop
print(args)

device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
data = load_data(dataset_name,train_ratio=0.6,val_ratio=0.2)
print("train dataset, val dataset and test dataset ", data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())

num_communities = len(set(data.y.numpy().tolist()))
print(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}")
print(f"Node Feature Matrix Info: # Node Features: {data.x.shape[1]}")
print(f"Edge Index Shape: {data.edge_index.shape}")
print(f"Edge Weight: {data.edge_attr}")
print(f"# Labels/classes: {num_communities}")

if missing_rate >0 :
    print("missing rate,", missing_rate)
    feature_mask = get_feature_mask(missing_rate,data['x'].shape[0],data['x'].shape[1])
    data['x'][~feature_mask] = float('nan')  ### replaced values with nan
    if imputation_method=='zero':
        X_reconstructed = torch.zeros_like(data['x'])
    if imputation_method == 'nf':
        print("Neighbourhood mean")
        X_reconstructed = neighborhood_mean_filling(data.edge_index,data.x,feature_mask)
    if imputation_method == 'fp':
        print("Feature propogation")
        X_reconstructed = feature_propagation(data.edge_index,data.x,feature_mask,50)  
    #X_reconstructed = feature_propagation(data.edge_index,data.x,feature_mask,50)#
    data['x'] = torch.where(feature_mask, data.x, X_reconstructed)
else:
    feature_mask = torch.ones_like(data['x']).bool()
# print("Remaining edges ",feature_mask.sum(),data['x'].shape[0]*data['x'].shape[1])
# print("Sum of data after masking", data.x.sum())
print("train dataset, val dataset and test dataset ", data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())

num_samples = [20,15]

if bs_train_nbd == -1:
    bs_train_nbd = data.x.shape[0]

if bs_test_nbd == -1:
    bs_test_nbd = data.x.shape[0]

def create_multiple_copies_data_feature_sampling_Phy(data, num_time_steps=50, node_select_prob=0.1, feat_update_rate=0, feat_add_delete_rate = 0, edge_delete_prob =0, edge_add_prob=0, feature_mask=None):
    
    dataset = copy.deepcopy(data)
    feature_mask_copy = copy.deepcopy(feature_mask)
    edge_index_copy = data.edge_index.clone()#copy.deepcopy(data.edge_index)s

    tp = data.x.clone() 
    
    delete_initial = 50
    number_of_edges = edge_index_copy.shape[1]
    indices_to_delete_initial = random.sample(list(range(0,number_of_edges)),delete_initial)#andom.sample(list, n)
    nodes_affected_edge_deletion_initial = []
    for index_delete in indices_to_delete_initial:
        nodes_affected_edge_deletion_initial.append(edge_index_copy[0][index_delete].item())
        nodes_affected_edge_deletion_initial.append(edge_index_copy[1][index_delete].item())

    edges_deleted_beginning = edge_index_copy[:,indices_to_delete_initial]
    nodes_affected_edge_deletion_initial =list(set(nodes_affected_edge_deletion_initial))
    nodes_structured_effected = nodes_affected_edge_deletion_initial
    indices_to_keep_initial = list(set(list(range(0,number_of_edges))).difference(set(indices_to_delete_initial)))
    edge_index_copy=edge_index_copy[:, indices_to_keep_initial]
    
    datax = [(data.x.clone(),list(range(data.x.shape[0])) , edge_index_copy.clone(), [], feature_mask_copy)]
    nodes_structured_effected = []
    
    for i in range(0, num_time_steps):
       
        expected_node_change = int(data.y.shape[0]*node_select_prob) 
            
        expected_feat_change = int(data.x.shape[1]*feat_update_rate)
        
        number_of_edges = edge_index_copy.shape[1]
        exepected_edge_del = int(number_of_edges*edge_delete_prob)
        exepected_edge_add = int(number_of_edges*edge_add_prob)
        
        number_of_edges = edge_index_copy.shape[1]
        indices_to_delete = random.sample(list(range(0,number_of_edges)),exepected_edge_del)#andom.sample(list, n)
        nodes_affected_edge_deletion = []
        for index_delete in indices_to_delete:
            nodes_affected_edge_deletion.append(edge_index_copy[0][index_delete].item())
            nodes_affected_edge_deletion.append(edge_index_copy[1][index_delete].item())
            
        nodes_affected_edge_deletion =list(set(nodes_affected_edge_deletion))
        nodes_structured_effected = nodes_affected_edge_deletion
        indices_to_keep = list(set(list(range(0,number_of_edges))).difference(set(indices_to_delete)))
        
        indicest_to_add_from_begin = random.sample(list(range(0,len(indices_to_delete_initial))),exepected_edge_add)
        edges_from_beginning = edges_deleted_beginning[:,indicest_to_add_from_begin]
        edge_index_copy=edge_index_copy[:, indices_to_keep]
        
        edge_index_copy = torch.cat((edge_index_copy, edges_from_beginning), 1)

        nodes_to_update = random.sample(list(range(0,data.x.shape[0])), expected_node_change)
                
        for node_id in nodes_to_update:
            feat_to_update = random.sample(list(range(0,data.x.shape[1])), expected_feat_change)
            for feat_id in feat_to_update:
                del_or_update = random.uniform(0,1)
                tp[node_id][feat_id] = 0
                if(del_or_update < feat_add_delete_rate): #### add this feature back
                    tp[node_id][feat_id] = data.x[node_id][feat_id]
                    feature_mask[node_id][feat_id] = 1
                else:  ### Delete this feature please
                    tp[node_id][feat_id] = 0  
                    feature_mask[node_id][feat_id] = 0
     
        datax.append((tp.clone(),nodes_to_update, edge_index_copy.clone(), nodes_structured_effected, copy.deepcopy(feature_mask)))#data.edge_index.clone()))
        
    return datax

def create_multiple_copies_data_feature_sampling_cora_cite(data, num_time_steps=50, node_select_prob=0.1, feat_update_rate=0, feat_add_delete_rate = 0, edge_delete_prob =0, edge_add_prob=0):
    
    dataset = copy.deepcopy(data)
    
    edge_index_copy = data.edge_index.clone()
   
    tp = data.x.clone() 
    
    delete_initial = 50#
    number_of_edges = edge_index_copy.shape[1]
    indices_to_delete_initial = random.sample(list(range(0,number_of_edges)),delete_initial)#andom.sample(list, n)

    nodes_affected_edge_deletion_initial = []
    for index_delete in indices_to_delete_initial:
        nodes_affected_edge_deletion_initial.append(edge_index_copy[0][index_delete].item())
        nodes_affected_edge_deletion_initial.append(edge_index_copy[1][index_delete].item())

    edges_deleted_beginning = edge_index_copy[:,indices_to_delete_initial]
    nodes_affected_edge_deletion_initial =list(set(nodes_affected_edge_deletion_initial))
    nodes_structured_effected = nodes_affected_edge_deletion_initial
    indices_to_keep_initial = list(set(list(range(0,number_of_edges))).difference(set(indices_to_delete_initial)))
    edge_index_copy=edge_index_copy[:, indices_to_keep_initial]
    
    datax = [(data.x.clone(),list(range(data.x.shape[0])) , edge_index_copy.clone(), [])]

    nodes_structured_effected = []
    
    for i in range(0, num_time_steps):
        expected_node_change = int(data.y.shape[0]*node_select_prob) 
        expected_feat_change = int(data.x.shape[1]*feat_update_rate)
        
        number_of_edges = edge_index_copy.shape[1]
        exepected_edge_del = int(number_of_edges*edge_delete_prob)
        exepected_edge_add = int(number_of_edges*edge_add_prob)
        number_of_edges = edge_index_copy.shape[1]
       
        indices_to_delete = random.sample(list(range(0,number_of_edges)),exepected_edge_del)
        
        nodes_affected_edge_deletion = []
        for index_delete in indices_to_delete:
            nodes_affected_edge_deletion.append(edge_index_copy[0][index_delete].item())
            nodes_affected_edge_deletion.append(edge_index_copy[1][index_delete].item())
            
        nodes_affected_edge_deletion =list(set(nodes_affected_edge_deletion))
        nodes_structured_effected = nodes_affected_edge_deletion
        indices_to_keep = list(set(list(range(0,number_of_edges))).difference(set(indices_to_delete)))
        
        indicest_to_add_from_begin = random.sample(list(range(0,len(indices_to_delete_initial))),exepected_edge_add)
        edges_from_beginning = edges_deleted_beginning[:,indicest_to_add_from_begin]
        edge_index_copy=edge_index_copy[:, indices_to_keep]
        edge_index_copy = torch.cat((edge_index_copy, edges_from_beginning), 1)
        
        nodes_to_update = random.sample(list(range(0,data.x.shape[0])), expected_node_change)
        for node_id in nodes_to_update:
            feat_to_update = random.sample(list(range(0,data.x.shape[1])), expected_feat_change)
            for feat_id in feat_to_update:
                del_or_update = random.uniform(0,1)
                tp[node_id][feat_id] = 0
                if(del_or_update < feat_add_delete_rate): #### add this feature back
                    tp[node_id][feat_id] = data.x[node_id][feat_id]
                else:  
                    tp[node_id][feat_id] = 0  
            
        datax.append((tp.clone(),nodes_to_update, edge_index_copy.clone(), nodes_structured_effected))#data.edge_index.clone()))
        
    return datax

steps = 10

edge_delete_prob = args.edge_prob
edge_add_prob =  args.edge_prob
nodes_prob = args.node_prob
del_feat_prob = args.del_prob 
feat_add_delete_rate= args.feat_add_delete_rate

if args.data in ['Cora','CiteSeer']:
    datax = create_multiple_copies_data_feature_sampling_cora_cite(data,steps, nodes_prob,del_feat_prob, feat_add_delete_rate, edge_delete_prob, edge_add_prob) #vary and play

elif args.data in ['Physics']:
    datax = create_multiple_copies_data_feature_sampling_Phy(data,steps, nodes_prob,del_feat_prob, feat_add_delete_rate, edge_delete_prob, edge_add_prob, feature_mask) #vary and play

    
    
num_communities = len(set(data.y.tolist()))
print(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}")
print(f"Node Feature Matrix Info: # Node Features: {data.x.shape[1]}")
print(f"Edge Index Shape: {data.edge_index.shape}")
print(f"Edge Weight: {data.edge_attr}")
print(f"# Labels/classes: {num_communities}")

def train_model(model, train_neigh_sampler, feature_mask, obs_features, feat_features,data, num_layers, sampling_in_loop, optimizer, X):
    model.train()

    total_loss = total_correct = 0
    total_computed = 0

    for batch_size, n_id, adjs in train_neigh_sampler:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers,sampling_in_loop=sampling_in_loop) 
        if bs_train_nbd == X.shape[0]:  
            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
        else:
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())

        total_computed+= out.shape[0]

    loss = total_loss / len(train_neigh_sampler_oracle)
    approx_acc = total_correct / total_computed#int(data.train_mask.sum())
   
    del out,a,b
    torch.cuda.empty_cache()


def importance(model,optimizer,train_importance_sampler,obs_features,feature_mask,feat_features,X,Y,num_layers=2, ):
    
    fisher_dict = {}
    optpar_dict = {}

    numPosSamples =64
    model.train()
    total_loss =total_correct= total_computed=0
    optimizer.zero_grad()
    
    for batch_size, n_id, adjs in train_importance_sampler:
        adjs = [adj.to(device) for adj in adjs]
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers) 
        if bs_train_nbd == data.x.shape[0]:  #### whole batch is coming as out
            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
        else:
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())
        total_computed+= out.shape[0]

    loss = total_loss / len(train_importance_sampler)
    approx_acc = total_correct / total_computed
    
    for name, param in model.named_parameters():
        if(param.grad is not None):
            optpar_dict[name] = param.data.clone()
            fisher_dict[name] = param.grad.data.clone().pow(2)
            
    return optpar_dict, fisher_dict 
    

def train_model_cont(model, train_neigh_sampler, feature_mask, obs_features, feat_features,data, num_layers, sampling_in_loop, optimizer,X, fisher_dict, optpar_dict, ewc_lambda ):

    model.train()

    total_loss = total_correct = 0
    total_computed = 0
    
    optimizer.zero_grad()
    loss=0

    for batch_size, n_id, adjs in train_neigh_sampler:
        adjs = [adj.to(device) for adj in adjs]

        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers,sampling_in_loop=sampling_in_loop) 
        if bs_train_nbd == X.shape[0]:  

            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = loss+ F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
        else:
            loss = loss + F.nll_loss(out, Y[n_id[:batch_size]])

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())

        total_computed+= out.shape[0]
            
    for name, param in model.named_parameters():
        
        if name in fisher_dict:
            fisher = fisher_dict[name]
            optpar = optpar_dict[name]
            loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
    
    loss.backward()
    optimizer.step()
        
    loss = total_loss / len(train_neigh_sampler_oracle)
    approx_acc = total_correct / total_computed#int(data.train_mask.sum())
 
    del out,a,b
    torch.cuda.empty_cache()
    
    
def test(model, subgraph_loader, feature_mask, obs_features, feat_features,data, X, num_layers):
    with torch.no_grad():
        model.eval()
        outs = []
        for batch_size, n_id, adjs in subgraph_loader:
            adjs = [adj.to(device) for adj in adjs]
            out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
                feat_features=feat_features,obs_adjs = adjs,data_x = X[n_id],num_layers=num_layers)
            outs.append(out)
            del a,b

        out = torch.cat(outs, dim=0)
        del outs
        train_acc = int(out.argmax(dim=-1).eq(Y)[data.train_mask].sum())*100.0/(data.train_mask.sum().item())
        val_acc = int(out.argmax(dim=-1).eq(Y)[data.val_mask].sum())*100.0/(data.val_mask.sum().item())
        test_acc = int(out.argmax(dim=-1).eq(Y)[data.test_mask].sum())*100.0/(data.test_mask.sum().item())
        
        del out
        import gc
        gc.collect()
        return  train_acc, val_acc, test_acc
    
def add_element_in_buffer(memory,item,buffer_size,current_len):
    if item in memory:
        return memory
    if len(memory) < buffer_size:
        memory.append(item)
    else:
        index = random.randrange(current_len)
        if index < buffer_size:
            memory[index]=item
    return memory
 

def add_elem(current_stream_length_logical, model,train_nodes_ft, memory,buffer_size, data_graph_time,obs_features,feature_mask,feat_features,num_layers=2,type_select = 'random' ):

    for node in train_nodes_ft.tolist():
        current_stream_length_logical +=1
        memory = add_element_in_buffer(memory,node,buffer_size,current_stream_length_logical)
        
    counter = collections.Counter(data_graph_time['data'].y[memory].tolist())
    return memory, current_stream_length_logical


num_epochs_ft = 120
num_epochs_ct = 120

current_stream_length_logical =0 

memory = []
mem_size = args.memory_size
memory_nodes_importance = []

time_graph = {}
    
for i in range(0, len(datax) -1):
    print("------Training start at time ", i , "------------>")
    
    X = datax[i][0]
    edge_index = datax[i][2]
    nodes_structured_effected = datax[i][3]
    
    if args.data == 'Physics':
        feature_mask =datax[i][4]
    
    time_graph[i] = {}
    time_graph[i]['data']= data
    time_graph[i]['edge_index']= datax[i][2]
    time_graph[i]['train_nodes'] = datax[i][1] 
    time_graph[i]['train_mask'] =torch.Tensor([False]*(X.shape[0])).to(torch.bool)
    
    if i ==0:
        time_graph[i]['train_mask'] = data.train_mask
            
    time_graph[i]['test_mask'] =data.test_mask
    time_graph[i]['val_mask'] =data.val_mask
    
    if args.data == 'Physics':
        if args.categorical == False:
            feature_mask = datax[i][4]
        else:
            feature_mask = X > 0
    else:
        feature_mask = X > 0
    
    nodes_changed_feature = datax[i][1]
    nodes_changed_structure = nodes_structured_effected
    
    train_nodes = []
    nodes_changed_anything= set(nodes_changed_feature).union(nodes_changed_structure)
        
    subgraph_loader = NeighborSampler(
        time_graph[i]['edge_index'], node_idx=None,
        sizes=[-1,-1], batch_size=bs_test_nbd, shuffle=False, num_workers=0)    
    
    for node in nodes_changed_anything:# list(range(0, len(time_graph[i].train_mask))):#sahil FIX#nodes_changed_anything:
        if time_graph[0]['train_mask'][node]==True:# train_mask_oracle[node]==True :#time_graph[i].train_mask[node]==True:
            train_nodes.append(node)
            
    for id_ in train_nodes:
        time_graph[i]['train_mask'][id_] = True
    
    if i >0:
        oracle_nodes = list(set(train_nodes).union(time_graph[i-1]['oracle_nodes']))
    else:
        oracle_nodes = list(train_nodes)
        
    time_graph[i]['oracle_nodes'] = oracle_nodes
    
    oracle_nodes = torch.LongTensor(oracle_nodes).to(device)
    
    train_neigh_sampler_oracle = NeighborSampler(
        time_graph[i]['edge_index'], node_idx= oracle_nodes,   ### Remeber to change
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)
    
    train_nodes_ft = [item for item in train_nodes]
    train_nodes_ft = torch.LongTensor(train_nodes_ft).to(device)     
    
    train_neigh_sampler_ft = NeighborSampler(
        time_graph[i]['edge_index'], node_idx= train_nodes_ft ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

    model_oracle = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
        num_feat_node_features=data.num_node_features,
        num_layers=2, hidden_size=args.hidden_size, out_channels=num_communities,heads=heads,
        categorical=categorical,device=device,feat_val_thresh=edge_value_thresh,
        otf_sample=otf_sample,fto_sample = fto_sample,
        num_obs_samples=num_obs_samples,num_feat_samples=num_feat_samples,
        use_data_x_otf=use_data_x_otf,use_data_x_fto=use_data_x_fto,otf_sample_testing=otf_sample_testing)
    model_oracle = model_oracle.to(device)  
    optimizer_oracle = torch.optim.Adam(model_oracle.parameters(), lr=args.lro,weight_decay = weight_decay)
    
    Y = data.y.squeeze().to(device)
    obs_features = torch.ones(data.x.shape[0],data.x.shape[1],dtype=torch.float32).to(device) 
    feat_features = np.eye(data.x.shape[1])
    feat_features = torch.tensor(feat_features,dtype=torch.float32).to(device)
    feature_mask  = feature_mask.to(device)

    model_save_path = "./../models/temp/"
    from pathlib import Path
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    actual_test_acc = 0
    best_val_acc = 0
    best_epoch = 0
    numPosSamples =64
    
    best_oracle_model = None
    for epoch in range(0,num_epochs):

        train_model(model_oracle, train_neigh_sampler_oracle, feature_mask, obs_features, feat_features,data, num_layers, sampling_in_loop, optimizer_oracle, X)
            
        train_acc, val_acc, test_acc = test(model_oracle, subgraph_loader, feature_mask, obs_features, feat_features,data, X, num_layers)
        
        if val_acc > best_val_acc:
                best_val_acc = val_acc
                actual_test_acc = test_acc
                best_epoch = epoch
                best_oracle_model = copy.deepcopy(model_oracle)
        if epoch%50==0:
            if verbose:
                print(f'ORACLE OUR : epoch:{epoch} , Train: {train_acc:.4f},Val acc:{val_acc:.4f} ,Test Acc: {test_acc:.4f},actual_test_acc: {actual_test_acc:.4f}')

    print("timestamp=",i,':::Oracle_Test_acc: ',actual_test_acc)
    
    X = X.to(device)
    Y = data.y.squeeze().to(device)
    
    
    if i==0:
        model_ct =  copy.deepcopy(best_oracle_model)
        
    optimizer_ct = torch.optim.Adam(model_ct.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    
    if i >0:

        sample_initial_from_memory_node_importance = list(set(memory) - set(nodes_changed_anything ) )
        train_nodes_importance = torch.LongTensor(sample_initial_from_memory_node_importance).to(device)  
        train_mix_ct =list(set(train_nodes_importance).union(train_nodes_ft) )
        train_mix_ct = torch.LongTensor(train_mix_ct).to(device)    
        
        train_neigh_sampler_mem_imp = NeighborSampler(
        time_graph[i]['edge_index'], node_idx= train_nodes_importance ,   
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)

        optpar_dict, fisher_dict = importance(model_ct,optimizer_ct,train_neigh_sampler_mem_imp,obs_features,feature_mask,feat_features,X,Y,num_layers=2)        
        
        ewc_lambda = args.ewc
        
        for epoch in range(num_epochs_ct):
            
            train_model_cont(model_ct, train_neigh_sampler_ft, feature_mask, obs_features, feat_features,data, num_layers, sampling_in_loop, optimizer_ct,X, fisher_dict=fisher_dict, optpar_dict = optpar_dict, ewc_lambda = ewc_lambda)

            train_acc_ct, val_acc_ct, test_acc_ct = test(model_ct, subgraph_loader, feature_mask, obs_features, feat_features,data,X, num_layers)
                
        print(f'timestamp={i}:::= Continual_Test_acc_ct= {test_acc_ct}::')
    
    if i ==0:
        memory, current_stream_length_logical = add_elem(current_stream_length_logical, model_ct,train_nodes_ft, memory,mem_size,time_graph[i],obs_features,feature_mask,feat_features,num_layers=2,type_select = 'entropy' )
        print('memory ', len(memory))
        

    