import torch
import numpy as np
import os
import random
import sys
src_dir = os.path.dirname(os.path.dirname('__file__'))
sys.path.append(src_dir)
from seed import seed


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return 
seed_everything(seed)




def create_otf_edges_sample(node_features,feature_mask):
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero(as_tuple=True)
    otf_edge_attr = node_features[otf_edge_index[0],otf_edge_index[1]].reshape(otf_edge_index[0].shape[0], -1)
    otf_edge_index = torch.cat((otf_edge_index[0].unsqueeze(0),otf_edge_index[1].unsqueeze(0)), dim=0)
    return otf_edge_index,otf_edge_attr

def create_otf_edges(node_features,feature_mask):
    #print('feature_mask ', feature_mask.sum())
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero(as_tuple=True)
    
    otf_edge_attr = node_features[otf_edge_index[0],otf_edge_index[1]].reshape(otf_edge_index[0].shape[0], -1)
    otf_edge_index = torch.cat((otf_edge_index[0].unsqueeze(0),otf_edge_index[1].unsqueeze(0)), dim=0)
    # print('otf_edge_index otf_edge_attr', otf_edge_index.shape,otf_edge_attr.shape)
    return otf_edge_index,otf_edge_attr

def get_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """ Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing"""
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()
    
def drop_negative_edges_from_feature_mask(node_features,feature_mask,drop_rate):
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero()
    otf_edge_attr = node_features[otf_edge_index[:,0],otf_edge_index[:,1]] 
    otf_edge_index_pos = otf_edge_index[otf_edge_attr.bool()!=False]
    otf_edge_index_neg = otf_edge_index[otf_edge_attr.bool()==False]
    neg_indices_to_keep = otf_edge_index_neg[torch.bernoulli(torch.Tensor([1 - drop_rate]).repeat(otf_edge_index_neg.shape[0])).bool()]
    indices_to_keep = torch.cat((otf_edge_index_pos,neg_indices_to_keep),dim=0)
    new_feature_mask = torch.zeros_like(feature_mask).bool()
    new_feature_mask[indices_to_keep[:,0],indices_to_keep[:,1]] = True
    #print(otf_edge_index_pos.shape,neg_indices_to_keep.shape,new_feature_mask.sum())
    return new_feature_mask



def sample_otf_edge_index(otf_edge_index,otf_edge_attr,num_features,info_flow="otf",num_samples=30,data_x=None,prob_eps =0.0000001, use_data_x_otf= False,device=0):
    otf = torch.zeros((otf_edge_index[0].max()+1,num_features),dtype=torch.bool).to(device)
    otf[otf_edge_index[0],otf_edge_index[1]] = 1
    #print(otf.shape,otf.sum())
    otf_attr = torch.zeros((otf.shape[0],otf.shape[1]),dtype=torch.float32).to(device)
    otf_attr[otf_edge_index[0],otf_edge_index[1]] = otf_edge_attr.squeeze(-1)
    
    if use_data_x_otf:  ### This can be used in case of no missing features
        otf[otf_edge_index[0],otf_edge_index[1]] = data_x.detach().to(device).to(torch.float32)[otf_edge_index[0],otf_edge_index[1]] > 0
        
    
    otf = otf.to(torch.float32)   ### No need to transpose , since we first will sample per obs node only
    otf += prob_eps
    otf = otf/otf.sum(dim=-1).unsqueeze(-1)
    
    sampled_otf_edge_index = torch.multinomial(otf,num_samples= num_samples,replacement=False).view(1,-1) ### Sampling 30 features per node
    dummy_node_tensor = torch.arange(otf.shape[0]).unsqueeze(-1).repeat(1,num_samples).view(1,-1).to(torch.long).to(device)
    sampled_otf_edge_index = torch.cat((dummy_node_tensor,sampled_otf_edge_index),dim=0)
    sampled_otf_edge_attr = otf_attr[sampled_otf_edge_index[0],sampled_otf_edge_index[1]].unsqueeze(-1)
    if info_flow == "fto":  ### features nodes to observation nodes
        sampled_otf_edge_index = torch.stack([sampled_otf_edge_index[1], sampled_otf_edge_index[0]], dim=0)
    return sampled_otf_edge_index,sampled_otf_edge_attr

# def sample_otf_edge_index(otf_edge_index,otf_edge_attr,num_obs_samples=30,data_x=None,device=0): ### edge
#     otf = torch.zeros((otf_edge_index[0].max()+1,otf_edge_index[1].max()+1),dtype=torch.float32).to(device)
#     otf[otf_edge_index[0],otf_edge_index[1]] = 1
#     if data_x is not None:  ### This can be used in case of no missing features
#         otf[otf_edge_index[0],otf_edge_index[1]] = data_x.detach().to(device).to(torch.float32)[otf_edge_index[0],otf_edge_index[1]]
#         #print(otf[otf_edge_index[0],otf_edge_index[1]].sum(),data_x[otf_edge_index[0],otf_edge_index[1]].sum())
#     otf = otf.T.to(torch.float32)
#     otf += 0.00001
#     otf = otf/otf.sum(dim=-1).unsqueeze(-1)
#     # otf = otf*30
#     # sampled_otf_edge_index = torch.bernoulli(otf)
#     sampled_otf_edge_index = torch.multinomial(otf,num_samples= num_samples,replacement=False).view(1,-1) ### Sampling 30 nodes per feature
#     dummy_feature_tensor = torch.arange(otf.shape[0]).unsqueeze(-1).repeat(1,num_obs_samples).view(1,-1).to(torch.long).to(device)
#     sampled_otf_edge_index = torch.cat((sampled_otf_edge_index,dummy_feature_tensor),dim=0)
#     otf_attr = torch.zeros((otf_edge_index[0].max()+1,otf_edge_index[1].max()+1),dtype=torch.long)
#     otf_attr[otf_edge_index[0],otf_edge_index[1]] = torch.arange(otf_edge_attr.shape[0],dtype=torch.long)
#     sampled_otf_edge_attr = otf_edge_attr[otf_attr[sampled_otf_edge_index[0],sampled_otf_edge_index[1]]] #.reshape(sampled_otf_edge_index[0].shape[0], -1)
#     return sampled_otf_edge_index,sampled_otf_edge_attr

def sample_otf_edge_index_v0(otf_edge_index,otf_edge_attr,num_samples=30,data_x=None,prob_eps =0.00001, use_data_x_otf= False,device=0): ### edge
    otf = torch.zeros((otf_edge_index[0].max()+1,otf_edge_index[1].max()+1),dtype=torch.bool).to(device)
    otf[otf_edge_index[0],otf_edge_index[1]] = 1
    otf_attr = torch.zeros((otf_edge_index[0].max()+1,otf_edge_index[1].max()+1),dtype=torch.float32).to(device)
    otf_attr[otf_edge_index[0],otf_edge_index[1]] = otf_edge_attr.squeeze(-1)

    if use_data_x_otf:  ### This can be used in case of no missing features
        #print("using datax")
        otf[otf_edge_index[0],otf_edge_index[1]] = data_x.detach().to(device).to(torch.float32)[otf_edge_index[0],otf_edge_index[1]] > 0
    otf = otf.T.to(torch.float32)
    otf += prob_eps
    otf = otf/otf.sum(dim=-1).unsqueeze(-1)
    # otf = otf*30
    # sampled_otf_edge_index = torch.bernoulli(otf)
    #print(otf.shape,num_samples)
    sampled_otf_edge_index = torch.multinomial(otf,num_samples= num_samples,replacement=False).view(1,-1) ### Sampling 30 nodes per feature
    dummy_feature_tensor = torch.arange(otf.shape[0]).unsqueeze(-1).repeat(1,num_samples).view(1,-1).to(torch.long).to(device)
    sampled_otf_edge_index = torch.cat((sampled_otf_edge_index,dummy_feature_tensor),dim=0)
    sampled_otf_edge_attr = otf_attr[sampled_otf_edge_index[0],sampled_otf_edge_index[1]].unsqueeze(-1) #.reshape(sampled_otf_edge_index[0].shape[0], -1)
    # print(sampled_otf_edge_attr.shape,sampled_otf_edge_attr.sum(),otf_edge_attr.sum())
    # import pdb
    # pdb.set_trace()
    
    #print(otf_edge_attr.shape,otf_edge_attr.sum(),sampled_otf_edge_attr.shape,sampled_otf_edge_attr.sum())
    return sampled_otf_edge_index,sampled_otf_edge_attr
