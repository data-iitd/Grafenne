from torch_geometric.data import Data
import pandas as pd
import json
import numpy as np
from sklearn import preprocessing
import torch
def create_edge_list(df):
    edge_list = df.values
    new_edge_list = []
    for edge in edge_list:
        if edge[0] != edge[1]:  ### remove self edges
            new_edge_list.append((int(edge[0]),int(edge[1])))
            new_edge_list.append((int(edge[1]),int(edge[0])))
    new_edge_list = list(set(new_edge_list)) ### remove multiple edges b/w same nodes 
    return new_edge_list
def extract_target(targets,data_type):
    node_to_target = {}
    if data_type in ['DE','ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']:
        id_ = 'new_id'
        target = 'mature'
    if data_type == 'facebook':
        id_ = 'id'
        target = 'page_type'
    if data_type == 'git':
        id_ = 'id'
        target = 'ml_target'
    ids = [int(item) for item in targets[id_]]
    targets = targets[target]

    le = preprocessing.LabelEncoder()
    targets = list(le.fit_transform(targets))
    print("Num of classes",len(le.classes_), le.classes_)
    return {id_:target for id_,target in zip(ids,targets)}

def extract_feature_matrix(features,node_targets):
    feature_list = []
    ct = 0
    for node, feats in features.items():
        feature_list.extend(feats)
        ct += len(set(feats))
    feature_list = list(set(feature_list))

    feature_matrix = np.zeros((len(node_targets),max(feature_list)+1))
    for node, feats in features.items():
        node = int(node)
        for feat in feats:
            feature_matrix[node][feat] = 1
    assert ct == feature_matrix.sum()
    return feature_matrix



##facebook: site_category classification
##github: classify node as web or machine learning developer
##lastfm: nationality of user
###WikipediaNetwork already on pyg
###lastfm asia
###deezer europse: gender
### twitch is binary classification of whether a streamer uses explicit language.

data_names= ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW','facebook','git','ZHTW']


def load_musae_data(data_type,data_folder = '/home/ctgnn/ctgnn/data/musae_datasets/'):
    
    if data_type not in data_names:
        print("Data type {} doesn't exist".format(data_type))
        return None
    edge_file= data_folder+"/edges/{}_edges.csv".format(data_type)
    feature_file =  data_folder+"/features/{}.json".format(data_type)
    target_file = data_folder+"/target/{}_target.csv".format(data_type)

    edges = pd.read_csv(edge_file)
    features = json.load(open(feature_file))
    targets = pd.read_csv(target_file)

    edge_list = np.array(create_edge_list(edges))
    node_targets = extract_target(targets,data_type)
    assert np.max(edge_list) + 1 == len(node_targets)
    Y = []
    for node_id in range(0, len(node_targets)):
        Y.append(node_targets[node_id])
    feature_matrix = extract_feature_matrix(features,node_targets)


    edge_list = np.transpose(edge_list)
    #print(edge_list.shape)
    edge_index = torch.tensor(edge_list,dtype=torch.long)
    node_feat = torch.tensor(feature_matrix,dtype=torch.bool) ## boolean features present or absent

    data = Data(
            x=node_feat,
            edge_index=edge_index,
            y=torch.tensor(Y,dtype=torch.long),
            train_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            val_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            test_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            num_nodes = node_feat.shape[0],
            num_features = node_feat.shape[1]

    )
    return data