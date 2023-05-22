#
import os
import sys
import torch
from torch_geometric.loader import NeighborSampler
import tqdm
import torch_sparse

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


def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
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
    if imputation_method in ['nf','fp']:
        if categorical == 0:
            print("modifying the feature mask in case of fp/nf")
            print("Remaining edges before this ",feature_mask.sum(),data['x'].shape[0]*data['x'].shape[1])
            feature_mask = torch.logical_or(data['x']>edge_value_thresh,feature_mask)
            print(feature_mask.shape)
            
else:
    feature_mask = torch.ones_like(data['x']).bool()
print("Remaining edges ",feature_mask.sum(),data['x'].shape[0]*data['x'].shape[1])
print("Sum of data after masking", data.x.sum())
num_samples = [20,15]

if bs_train_nbd == -1:
    bs_train_nbd = data.x.shape[0]

if bs_test_nbd == -1:
    bs_test_nbd = data.x.shape[0]
    
print("bs_train_nbd and test_nbd", bs_train_nbd,bs_test_nbd)
train_neigh_sampler = NeighborSampler(
        data.edge_index, node_idx= data.train_mask ,   ### Remeber to change
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0)



subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1,-1], batch_size=bs_test_nbd, shuffle=False, num_workers=0)


# import pdb
# pdb.set_trace()
num_communities = len(set(data.y.numpy().tolist()))
print(f"Node Feature Matrix Info: # Nodes: {data.x.shape[0]}")
print(f"Node Feature Matrix Info: # Node Features: {data.x.shape[1]}")
print(f"Edge Index Shape: {data.edge_index.shape}")
print(f"Edge Weight: {data.edge_attr}")
print(f"# Labels/classes: {num_communities}")

 

import gc 




## head = 4
print("number of heads,", heads)
model = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
    num_feat_node_features=data.num_node_features,
    num_layers=2, hidden_size=256, out_channels=num_communities,heads=heads,
    categorical=categorical,device=device,feat_val_thresh=edge_value_thresh,
    otf_sample=otf_sample,fto_sample = fto_sample,
    num_obs_samples=num_obs_samples,num_feat_samples=num_feat_samples,
    use_data_x_otf=use_data_x_otf,use_data_x_fto=use_data_x_fto,otf_sample_testing=otf_sample_testing)
model = model.to(device)  #0001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay = weight_decay)

Y = data.y.squeeze().to(device)
obs_features = torch.ones(data.x.shape[0],data.x.shape[1],dtype=torch.float32).to(device) 
print(obs_features.shape)
feat_features = np.eye(data.x.shape[1])
feat_features = torch.tensor(feat_features,dtype=torch.float32).to(device)
print(feat_features.shape)
feature_mask  = feature_mask.to(device)
#data.x = data.x.to(device)


model_save_path = "./../models/temp/"
from pathlib import Path
Path(model_save_path).mkdir(parents=True, exist_ok=True)
actual_test_acc = 0
best_val_acc = 0
best_epoch = 0
numPosSamples =64

# import pdb
# pdb.set_trace()

for epoch in range(0,num_epochs):

    model.train()
    
    total_loss = total_correct = 0
    total_computed = 0

    for batch_size, n_id, adjs in train_neigh_sampler:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        # print(obs_features.shape,obs_features[n_id].shape,feature_mask.shape,n_id.shape)
        # import pdb
        # pdb.set_trace()
        # break
        out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
            feat_features=feat_features,obs_adjs = adjs,data_x = data.x[n_id],num_layers=num_layers,sampling_in_loop=sampling_in_loop) 
        #print(out.shape,batch_size,Y.shape)
        if bs_train_nbd == data.x.shape[0]:  #### whole batch is coming as out

            
            p = torch.Tensor([1/batch_size*1.0]*batch_size)
            sampledPosIndex = p.multinomial(num_samples=numPosSamples, replacement=False)
            newMask = torch.Tensor([False]*batch_size)
            newMask = newMask.to(torch.bool)
            newMask[sampledPosIndex]=True
            loss = F.nll_loss(out[newMask], Y[n_id[:batch_size]][newMask])
            #loss = F.nll_loss(out[newMask], Y_temp[newMask])
        else:
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(Y[n_id[:batch_size]]).sum())
        
        total_computed+= out.shape[0]

    loss = total_loss / len(train_neigh_sampler)
    approx_acc = total_correct / total_computed#int(data.train_mask.sum())
    if verbose:
        print(f"epoch:{epoch},loss:{loss:.4f},train_acc_approx:{approx_acc}")
    del out,a,b
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        outs = []
        for batch_size, n_id, adjs in subgraph_loader:
            adjs = [adj.to(device) for adj in adjs]
            out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
                feat_features=feat_features,obs_adjs = adjs,data_x = data.x[n_id],num_layers=num_layers)
            outs.append(out)
            del a,b

        out = torch.cat(outs, dim=0)
        del outs
        train_acc = int(out.argmax(dim=-1).eq(Y)[data.train_mask].sum())*100.0/(data.train_mask.sum().item())
        val_acc = int(out.argmax(dim=-1).eq(Y)[data.val_mask].sum())*100.0/(data.val_mask.sum().item())
        test_acc = int(out.argmax(dim=-1).eq(Y)[data.test_mask].sum())*100.0/(data.test_mask.sum().item())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            actual_test_acc = test_acc
            best_epoch = epoch
            #torch.save(model, model_save_path + f'/best_val_{val_acc:.2f}.pt')
        if verbose:
            print(f'epoch:{epoch} , Train: {train_acc:.4f},Val acc:{val_acc:.4f} ,Test Acc: {test_acc:.4f},actual_test_acc: {actual_test_acc:.4f}')
                
        del out
        import gc
        gc.collect()
print("Test accuracy,",actual_test_acc )

if result_file.strip() != '':
    with open(result_file,"a") as f:
        f.write(str(actual_test_acc))
        f.write("\n")
        f.close()