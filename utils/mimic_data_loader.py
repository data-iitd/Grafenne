from torch_geometric.data import Data
import pandas as pd
import json
import numpy as np
from sklearn import preprocessing
import torch
import sys
import os
import pickle

import pathlib

def get_item_to_id_dict(lsts):
    dict_ = {}
    for j in lsts:
        if j != '' and j not in dict_:
            dict_[j] = len(dict_)
    return dict_
def get_items_to_id_dict(lsts):
    dict_ = {}
    for i in lsts:
        for j in i:
            if j not in dict_:
                dict_[j] = len(dict_)
    return dict_

def load_mimic_data(type_='icd'):
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/Mimic3/"
    print(data_dir)
    print("loading mimic data of ", type_)
    edge_list = pickle.load(open(data_dir+"/edge_list_{}.pkl".format(type_),"rb"))
    admissions = pd.read_pickle(data_dir+"/processed_admissions_{}.pkl".format(type_))
    print(len(edge_list),admissions.shape)
    
    node_to_id = get_item_to_id_dict(admissions['HADM_ID'])
    gender_to_id = get_item_to_id_dict(admissions['GENDER'])
    if type_ != 'icd':  ### list of procedures
        proced_to_id = get_items_to_id_dict(admissions['PROCED_ICD9_CODES'])
    else:   ### single procedure performed
        proced_to_id = get_item_to_id_dict(admissions['PROCED_ICD9_CODES'])
        
        
    drugs_to_id = get_items_to_id_dict(admissions['DRUGS'])
    lab_items_to_id = get_items_to_id_dict(admissions['LAB_ITEMIDS'])
    
    final_edge_list = []
    for edge in edge_list:
        node1 = node_to_id[edge[0]]
        node2 = node_to_id[edge[1]]
        final_edge_list.append((node1,node2))
        final_edge_list.append((node2,node1))
    print(len(final_edge_list))

    print(len(final_edge_list))
    final_edge_list = list(set(final_edge_list))
    print(len(final_edge_list))
    assert np.max(final_edge_list) + 1 == len(node_to_id)
    final_edge_list = np.transpose(final_edge_list)
    edge_index = torch.tensor(final_edge_list,dtype=torch.long)
    print(edge_index.shape)
    
    
    Y = [0]*len(node_to_id)
    node_feats = [0]*len(node_to_id)
    for node,HOSPITAL_EXPIRE_FLAG,GENDER,PROCED_ICD9_CODES,DRUGS,LAB_ITEMIDS in admissions[['HADM_ID','HOSPITAL_EXPIRE_FLAG','GENDER','PROCED_ICD9_CODES','DRUGS','LAB_ITEMIDS']].values:   
        index = node_to_id[node]
        
        if type_ == 'icd':
            y_label = proced_to_id[PROCED_ICD9_CODES]
        else:
            y_label = int(HOSPITAL_EXPIRE_FLAG)
        
        gender_vector = np.zeros(2)
        drug_vector = np.zeros(len(drugs_to_id))
        lab_item_vector = np.zeros(len(lab_items_to_id))
        if GENDER in gender_to_id:
            gender_vector[gender_to_id[GENDER]] = 1
        if type_ != 'icd':
            proced_vector = np.zeros(len(proced_to_id))
            for item in PROCED_ICD9_CODES:
                proced_vector[proced_to_id[item]]= 1
        for item in DRUGS:
            drug_vector[drugs_to_id[item]]= 1
        for item in LAB_ITEMIDS:
            lab_item_vector[lab_items_to_id[item]]= 1
            
        if type_ == 'icd':
            feat_vector = np.concatenate((gender_vector,drug_vector,lab_item_vector))
        else:
            feat_vector = np.concatenate((gender_vector,proced_vector,drug_vector,lab_item_vector))


        Y[index] = y_label
        node_feats[index] = feat_vector
        
        
        
    Y = np.array(Y)
    node_feats = np.array(node_feats)
    print(Y.shape,node_feats.shape)
    print(sum(node_feats.sum(axis=1)==0))

    node_feats = torch.tensor(node_feats,dtype=torch.float) ## boolean features present or absent
    print(node_feats.shape)
    
    

    data = Data(
            x=node_feats,
            edge_index=edge_index,
            y=torch.tensor(Y,dtype=torch.long),
            train_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            val_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            test_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            num_nodes = node_feats.shape[0],
            num_features = node_feats.shape[1]

    )
    return data