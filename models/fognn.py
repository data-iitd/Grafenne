import torch
import sys
import os
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout,LeakyReLU
from torch_geometric.nn import GATConv,GINConv,SAGEConv, GATConv
import torch_geometric
src_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_dir)
from utils.utils import create_otf_edges,create_otf_edges_sample


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self,emb_size=64,dropout=0.2):
        super().__init__('mean')
        self.drop_rate = dropout
        self.feature_module_left = torch.nn.Sequential(
            Linear(in_features=emb_size, out_features=emb_size),
            BatchNorm1d(num_features=emb_size,track_running_stats=False,affine=False),
            Dropout(p=self.drop_rate),
            Linear(emb_size,emb_size),
            ReLU(), 
        )
        self.feature_module_edge = torch.nn.Sequential(
            Linear(in_features=emb_size, out_features=emb_size),
            BatchNorm1d(num_features=emb_size,track_running_stats=False,affine=False),
            Dropout(p=self.drop_rate),
            Linear(emb_size,emb_size),
            ReLU(), 

        )
        self.feature_module_right = torch.nn.Sequential(
            Linear(in_features=emb_size, out_features=emb_size),
            BatchNorm1d(num_features=emb_size,track_running_stats=False,affine=False),
            Dropout(p=self.drop_rate),
            Linear(emb_size,emb_size),
            ReLU(), 

        )
        self.feature_module_final = torch.nn.Sequential(
            Linear(in_features=3*emb_size, out_features=emb_size),
            BatchNorm1d(num_features=emb_size,track_running_stats=False,affine=False),
            Dropout(p=self.drop_rate),
            Linear(emb_size,emb_size),
            ReLU(), 

        )

        self.post_conv_module = torch.nn.Sequential(
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            Linear(in_features=2*emb_size, out_features=emb_size),
            BatchNorm1d(num_features=emb_size,track_running_stats=False,affine=False),
            Dropout(p=self.drop_rate),
            Linear(emb_size,emb_size),
            ReLU(), 

        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        #print('output', output.shape)
        return self.output_module(torch.cat([output, right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(torch.cat((self.feature_module_right(node_features_i),
                                          self.feature_module_edge(edge_features),
                                        self.feature_module_right(node_features_j)),dim=1))
        
        #print('output', output.shape)
        return output
    

    
        
class ScalableFOGNN(torch.nn.Module):
    def __init__(self, drop_rate,num_obs_node_features,num_feat_node_features, num_layers, hidden_size, out_channels,heads,device,categorical=False, gnnType='SAGEConv',feat_val_thresh=0.01):
        super(ScalableFOGNN, self).__init__()
        self.device = device
        self.feat_value_thresh = feat_val_thresh
        print("setting threshold as ",self.feat_value_thresh)
        self.heads = heads
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.num_obs_node_features = num_obs_node_features
        self.num_feat_node_features = num_feat_node_features
        self.hidden_size = hidden_size
        #self.edge_hidden_size = edge_hidden_size
        self.projects_obs =torch.nn.ModuleList()
        self.projects_feat =torch.nn.ModuleList()
        self.categorical = categorical
        self.projects_obs.append(Sequential(
                    Linear(in_features=self.num_obs_node_features, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))
        self.projects_obs.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))

        
        self.projects_feat.append(Sequential(
                    Linear(in_features=self.num_feat_node_features, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))
        self.projects_feat.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))

        
        self.projects =torch.nn.ModuleList()
        self.projects.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.projects.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.num_otf_edge_features = 1
        self.edge_embedding =torch.nn.ModuleList()
        self.edge_embedding.append(Sequential(
                    Linear(in_features=self.num_otf_edge_features, out_features=self.hidden_size)))
        self.edge_embedding.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size)))
        
        self.edge_embedding_update = torch.nn.ModuleList()
        self.edge_embedding_update.append(Sequential(
                    Linear(in_features=3*self.hidden_size, out_features=self.hidden_size),
                    ReLU(),
                    Linear(self.hidden_size,self.hidden_size)
                ))
        self.edge_embedding_update.append(Sequential(
                    Linear(in_features=3*self.hidden_size, out_features=self.hidden_size),
                    ReLU(),
                    Linear(self.hidden_size,self.hidden_size)
                ))
        

        #### Convert these to lists as well seperate convolution per layer

        self.conv_f_to_o =torch.nn.ModuleList()
        self.conv_f_to_o.append(BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate))
        self.conv_f_to_o.append(BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate))
        
        
        self.conv_o_to_f = torch.nn.ModuleList()
        self.conv_o_to_f.append(BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate))
        self.conv_o_to_f.append(BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate))        

        
        self.out_channels = out_channels
        self.dropout = emb_size=hidden_size
        self.gin_convs = torch.nn.ModuleList()
        # self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size))
        # self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size))
        
        #print('gnnType', gnnType)
        
        if(gnnType=='SAGEConv'):
            self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
            self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
            #print('USING SAGEConvs ' )
        
        elif(gnnType=='GATConv'):
            print("using gat")
            self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size//4,heads= 4,concat=True))
            self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size,heads=1))
        elif(gnnType=='GINConv'):
            self.gin_convs.append(
                GINConv(
                    nn=Sequential(
                        Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                        ReLU(),
                        Linear(self.hidden_size, self.hidden_size),
                    ),eps= 0.01
                )
            )
            self.gin_convs.append(
                GINConv(
                    nn=Sequential(
                        Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                        ReLU(),
                        Linear(self.hidden_size, self.hidden_size)
                    ),eps= 0.01
                )
            )
        else:
            import sys
            print(gnnType, " not defined")
            sys.exit(1)

        self.project_Y = Sequential(
            Linear(self.hidden_size,self.hidden_size),
            LeakyReLU(negative_slope=0.01),
            Linear(self.hidden_size,self.out_channels),

        )
        self.repeat_conv_f_to_o = BipartiteGraphConvolution(emb_size=hidden_size)
        
        self.gat_convs_f_to_o = torch.nn.ModuleList()
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads, edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.gat_convs_o_to_f = torch.nn.ModuleList()
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.repeat_gat_convs_f_to_o  = GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2)
        
        self.obs_gat_project = torch.nn.ModuleList()
        self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))

        self.feat_gat_project = torch.nn.ModuleList()
        self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        self.repeat_obs_gat_project = Linear(2*self.hidden_size,self.hidden_size)

        self.gat_linear_project_obs = torch.nn.ModuleList()
        self.gat_linear_project_obs.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.gat_linear_project_obs.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.repeat_gat_linear_project_obs = Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                )
        self.gat_linear_project_feat = torch.nn.ModuleList()
        self.gat_linear_project_feat.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))        
        self.gat_linear_project_feat.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))    
        self.gat_linear_project = torch.nn.Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size)
 
        
       
       
    def forward(self, obs_features,feature_mask,feat_features,obs_adjs,data_x ,drop_rate=0,num_layers = 1): 
        ### Drop edges containing 0 value
        # if drop_rate >0 :
        #     #print(feature_mask.sum(),data_x[feature_mask].sum())
        #     feature_mask = drop_negative_edges_from_feature_mask(data_x,feature_mask,drop_rate=drop_rate)
        #     #print(feature_mask.sum(),data_x[feature_mask].sum())
        #### create obs feature graph 
        if self.categorical:
            feature_mask = data_x >self.feat_value_thresh#data_x.bool()   ### Only take those edges which has value >= 1
        if num_layers == 1:
            otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
            otf_edge_index = otf_edge_index.to(self.device)
            otf_edge_features = otf_edge_features.to(self.device)
            fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
            fto_edge_index = fto_edge_index
            otf_edge_features = otf_edge_features.double()

            obs_features = self.projects_obs[0](obs_features)
            feat_features = self.projects_feat[0](feat_features)
            otf_edge_features = self.edge_embedding[0](otf_edge_features)
            ### Message passing from feature to observation nodes ####
            #### Here fto_edge_index needs to be index from 0 as well since left side and 
            ### right side of bipartite graph is indexed seperately
            #obs_features = self.conv_f_to_o[0](feat_features, fto_edge_index, otf_edge_features, obs_features)
            obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_linear_project_obs[0](self.gat_convs_f_to_o[0]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))

            #print(obs_features.shape)
            #for i in range(self.num_layers):

            ##### Message passing between observation nodes ####

            for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
                obs_features = self.projects[i](obs_features)
                obs_features_target= obs_features[:size[1]]
                obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
                obs_features = F.elu(obs_features)
                obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
                  #### create obs feature graph 

            new_num_feature_edges = feature_mask[:size[1],:].sum()
            #print(new_num_feature_edges,otf_edge_index.shape)
            otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
            fto_edge_index = fto_edge_index[:,:new_num_feature_edges]

            otf_edge_features = otf_edge_features[:new_num_feature_edges,:]

            ### Message passing from observation to feature nodes 
            feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_linear_project_feat[0](self.gat_convs_o_to_f[0]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features))),dim=-1))

            #### Finally re-computing edge embeddings 
            otf_edge_features = self.edge_embedding_update[0](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))

            #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
            obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_linear_project_obs(self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))


            obs_features  = self.project_Y(obs_features)
            out = F.log_softmax(obs_features, dim=-1)
            return out,obs_features,feat_features
        if num_layers == 2:
        #### create obs feature graph 
            otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
            otf_edge_index = otf_edge_index.to(self.device)
            otf_edge_features = otf_edge_features.double().to(self.device)
            fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
            fto_edge_index = fto_edge_index

            for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):

                obs_features = self.projects_obs[i](obs_features)
                feat_features = self.projects_feat[i](feat_features)
                otf_edge_features = self.edge_embedding[i](otf_edge_features)
            ### Message passing from feature to observation nodes ####
            #### Here fto_edge_index needs to be index from 0 as well since left side and 
            ### right side of bipartite graph is indexed seperately
                #obs_features = self.conv_f_to_o[i](feat_features, fto_edge_index, otf_edge_features, obs_features)
                obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_linear_project_obs[i](self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))


            ##### Message passing between observation nodes ####

                obs_features = self.projects[i](obs_features)
                obs_features_target= obs_features[:size[1]]
                obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
                obs_features = F.elu(obs_features)
                obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
                  #### create obs feature graph 

                new_num_feature_edges = feature_mask[:size[1],:].sum()
                otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
                fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
                otf_edge_features = otf_edge_features[:new_num_feature_edges,:]
            ### Message passing from observation to feature nodes 
                #feat_features = self.conv_o_to_f[i](obs_features, otf_edge_index, otf_edge_features, feat_features)
                feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_linear_project_feat[i](self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features))),dim=-1))

            #### Finally re-computing edge embeddings 
                otf_edge_features = self.edge_embedding_update[i](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))



            #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
            #obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_linear_project_obs(self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))


            obs_features  = self.project_Y(obs_features)
            out = F.log_softmax(obs_features, dim=-1)
            return out,obs_features,feat_features
    
    
    
    
class FOGNN(torch.nn.Module):
    def __init__(self, drop_rate,num_obs_node_features,num_feat_node_features, num_layers, hidden_size, out_channels,heads =6):
        super(FOGNN, self).__init__()
        self.heads = heads
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        #print("Drop rate", self.drop_rate)
        self.num_obs_node_features = num_obs_node_features
        self.num_feat_node_features = num_feat_node_features
        self.hidden_size = hidden_size
        
        self.projects_obs =torch.nn.ModuleList()
        self.projects_feat =torch.nn.ModuleList()
        self.projects_obs.append(Sequential(
                    Linear(in_features=self.num_obs_node_features, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))
        self.projects_obs.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))

        
        self.projects_feat.append(Sequential(
                    Linear(in_features=self.num_feat_node_features, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),        

                ))
        self.projects_feat.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))

        
        self.projects =torch.nn.ModuleList()
        self.projects.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.projects.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))

        self.num_otf_edge_features = 1
        self.edge_embedding =torch.nn.ModuleList()
        self.edge_embedding.append(Sequential(
                    Linear(in_features=self.num_otf_edge_features, out_features=self.hidden_size)))
        self.edge_embedding.append(Sequential(
                    Linear(in_features=self.hidden_size, out_features=self.hidden_size)))

        self.edge_embedding_update = torch.nn.ModuleList()
        self.edge_embedding_update.append(Sequential(
                    Linear(in_features=3*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   

                ))
        self.edge_embedding_update.append(Sequential(
                    Linear(in_features=3*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   

                ))

        

        #### Convert these to lists as well seperate convolution per layer
        self.conv_f_to_o = BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate)
        self.conv_o_to_f = BipartiteGraphConvolution(emb_size=hidden_size,dropout=self.drop_rate)
        

        
        self.out_channels = out_channels

        self.gin_convs = torch.nn.ModuleList()
        self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        

        self.project_Y = Sequential(
            Linear(self.hidden_size,self.hidden_size),
            LeakyReLU(negative_slope=0.01),
            Linear(self.hidden_size,self.out_channels),

        )
        
        self.gat_convs_f_to_o = torch.nn.ModuleList()
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads, edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        
        self.gat_convs_o_to_f = torch.nn.ModuleList()
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size,heads=self.heads,edge_dim=self.hidden_size,add_self_loops=False,dropout=0.2))
        
#         self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=1))#,edge_dim=self.hidden_size))
#         self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size,heads=1))#, edge_dim=self.hidden_size))
        # self.conv1 = GATConv(in_channels, out_channels,heads=heads, edge_dim =1, add_self_loops=False)#, fill_value =1.0)#add_self_loops =False)
        # self.linear_transform = nn.Linear(out_channels*heads, out_channels)
        
    
    
        self.obs_gat_project = torch.nn.ModuleList()
        self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))

        self.feat_gat_project = torch.nn.ModuleList()
        self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))

        self.gat_linear_project_obs = torch.nn.ModuleList()
        self.gat_linear_project_obs.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.gat_linear_project_obs.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))
        self.gat_linear_project_feat = torch.nn.ModuleList()
        self.gat_linear_project_feat.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))        
        self.gat_linear_project_feat.append(Sequential(
                    Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size),
                    BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
                    Dropout(p=self.drop_rate),
                    Linear(self.hidden_size,self.hidden_size),
                    ReLU(),   
                ))    
        self.gat_linear_project = torch.nn.Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size)
        # self.gat_convs_o_to_f.append(nn.Linear(in_features=self.heads*self.hidden_size, out_features=self.hidden_size))#, edge_dim=self.hidden_size))
        
       
    def forward(self, obs_features,feat_features,otf_edge_index,otf_edge_attr, train_neigh_sampler ):  ## X contains features of observation and feature nodes , first observation and then features


        otf_edge_features = otf_edge_attr
        fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
        fto_edge_index = fto_edge_index

            
        for i in range(self.num_layers):
            
            obs_features = self.projects_obs[i](obs_features)
            feat_features = self.projects_feat[i](feat_features)
            otf_edge_features = self.edge_embedding[i](otf_edge_features)
            
            ### Message passing from feature to observation nodes ####
            #obs_features = self.conv_f_to_o(feat_features, fto_edge_index, otf_edge_features, obs_features)
            
            #### To add concateination
            obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_linear_project_obs[i](self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))
            
            #print(i, obs_features.shape)
            #obs_features = self.gat_linear_project_obs[i](obs_features)
            #### Here fto_edge_index needs to be index from 0 as well since left side and 
            ### right side of bipartite graph is indexed seperately
            

            ##### Message passing between observation nodes ####
            xs = []
            for batch_size, n_id, adj in train_neigh_sampler:
                adj = adj.to(device)
                edge_index, _, size = adj
                x = obs_features[n_id]
                x = self.projects[i](x)
                x_target = x[:size[1]]
                x = self.gin_convs[i]((x, x_target), edge_index)
                x = F.elu(x)
                # if i != self.num_layers - 1:  ### This seems to be wrong
                #     x = F.relu(x)
                xs.append(x.cpu())
                del x
                del x_target
                del adj
                del edge_index
                del size

                
            obs_features = torch.cat(xs, dim=0).to(device) ## new obs features
            del xs
            
            #print("Obs feature shape", obs_features.shape)
            ### Message passing from observation to feature nodes 
            
            #feat_features = self.conv_o_to_f(obs_features, otf_edge_index, otf_edge_features, feat_features)
            feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_linear_project_feat[i](self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features))),dim=-1))

            #feat_features = self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)#, otf_edge_features)
            #feat_features = self.gat_linear_project_feat[i](feat_features)

            
            ### But to compute otf_edge features, we need 
            otf_edge_features = self.edge_embedding_update[i](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))
            

        obs_features  = self.project_Y(obs_features)
        out = F.log_softmax(obs_features, dim=-1)
        return out#,obs_features,feat_features