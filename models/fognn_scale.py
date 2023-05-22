import torch
import sys
import os
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout,LeakyReLU
from torch_geometric.nn import GATConv,GINConv,SAGEConv, GATConv
import torch_geometric
src_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_dir)
from utils.utils import create_otf_edges,create_otf_edges_sample,sample_otf_edge_index,sample_otf_edge_index_v0


def temp_layer(input_size,output_size,drop_rate=0.2):
    return Sequential(
                    Linear(in_features=input_size, out_features=output_size),
                    BatchNorm1d(num_features=output_size,track_running_stats=False,affine=True),
                    Dropout(p=drop_rate),
                    Linear(output_size,output_size),
                    ReLU(),        
                )

def temp_gin_layer(input_size,output_size,drop_rate=0.2):
    return GINConv(nn=Sequential(
                        Linear(in_features=input_size, out_features=output_size),
                        #BatchNorm1d(num_features=output_size,track_running_stats=False,affine=True),
                        ReLU(),
                        Linear(output_size, output_size),
                    ),eps= 0.01
                )
class ScalableFOGNN(torch.nn.Module):
    def __init__(self, drop_rate,num_obs_node_features,num_feat_node_features,
        num_layers, hidden_size, out_channels,heads,device,categorical=False,
        gnnType='SAGEConv',feat_val_thresh=0.01,otf_sample=False,fto_sample=False,
        num_obs_samples = 30, num_feat_samples=30,use_data_x_otf= False,
        use_data_x_fto=False,otf_sample_testing=False,task_type="node"):
        super(ScalableFOGNN, self).__init__()
        self.device = device
        self.feat_value_thresh = feat_val_thresh
        print("setting threshold as ",self.feat_value_thresh)
        self.heads = heads
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.num_obs_node_features = num_obs_node_features
        self.num_feat_node_features = num_feat_node_features
        print("No. of features, ",self.num_feat_node_features  )
        self.hidden_size = hidden_size
        self.edge_hidden_size = self.hidden_size
        self.categorical = categorical
        self.otf_sample= otf_sample
        #print("Otf sampling, ", otf_sample)
        self.fto_sample = fto_sample
        #print("Fto sampling,", fto_sample )
        self.num_obs_samples = num_obs_samples
        self.num_feat_samples = num_feat_samples
        self.use_data_x_otf= use_data_x_otf
        self.use_data_x_fto= use_data_x_fto
        self.otf_sample_testing=otf_sample_testing
        self.task_type = task_type
        print(otf_sample,fto_sample,num_obs_samples,num_feat_samples,use_data_x_otf,use_data_x_fto,self.task_type)
        self.projects =torch.nn.ModuleList()
        self.projects.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))
        self.projects.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))

        self.projects_obs =torch.nn.ModuleList()
        self.projects_obs.append(temp_layer(self.num_obs_node_features,self.hidden_size,self.drop_rate))
        self.projects_obs.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))
        
        self.projects_feat =torch.nn.ModuleList()
        self.projects_feat.append(temp_layer(self.num_feat_node_features,self.hidden_size,self.drop_rate))
        self.projects_feat.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))


        self.num_otf_edge_features = 1
        self.otf_edge_embedding =torch.nn.ModuleList()
        self.otf_edge_embedding.append(Sequential(Linear(in_features=self.num_otf_edge_features, out_features=self.edge_hidden_size)))
        self.otf_edge_embedding.append(Sequential(Linear(in_features=self.edge_hidden_size, out_features=self.edge_hidden_size)))
    
        self.fto_edge_embedding =torch.nn.ModuleList()
        self.fto_edge_embedding.append(Sequential(Linear(in_features=self.num_otf_edge_features, out_features=self.edge_hidden_size)))
        self.fto_edge_embedding.append(Sequential(Linear(in_features=self.edge_hidden_size, out_features=self.edge_hidden_size)))
        
#         self.edge_embedding_update = torch.nn.ModuleList()
#         self.edge_embedding_update.append(Sequential(
#                     Linear(in_features=2*self.hidden_size+self.edge_hidden_size, out_features=self.edge_hidden_size),
#                     ReLU(),
#                     Linear(self.edge_hidden_size,self.edge_hidden_size)
#                 ))
#         self.edge_embedding_update.append(Sequential(
#                     Linear(in_features=2*self.hidden_size+self.edge_hidden_size, out_features=self.edge_hidden_size),
#                     ReLU(),
#                     Linear(self.edge_hidden_size,self.edge_hidden_size)
#                 ))
        

        
        self.out_channels = out_channels
        self.dropout = emb_size=hidden_size
        self.gin_convs = torch.nn.ModuleList()

        print('gnnType', gnnType)
        
        if(gnnType=='SAGEConv'):
            self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
            self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        
        elif(gnnType=='GATConv'):
            print("using gat")
            self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size//4,heads= 4,concat=True))
            self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size,heads=1))
        elif(gnnType=='GINConv'):
            self.gin_convs.append(temp_gin_layer(self.hidden_size,self.hidden_size,self.drop_rate))
            self.gin_convs.append(temp_gin_layer(self.hidden_size,self.hidden_size,self.drop_rate))

        else:
            import sys
            print(gnnType, " not defined")
            sys.exit(1)

        self.project_Y = Sequential(
            Linear(self.hidden_size,self.hidden_size),
            BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
            LeakyReLU(negative_slope=0.01),
            Linear(self.hidden_size,self.out_channels),

        )
        
        self.gat_convs_f_to_o = torch.nn.ModuleList()
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads, edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        
        self.gat_convs_o_to_f = torch.nn.ModuleList()
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        
        self.repeat_gat_convs_f_to_o  = GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2)
        
        self.obs_gat_project = torch.nn.ModuleList()
        self.obs_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))
        self.obs_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))

        
        
        # self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        # self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))

        self.feat_gat_project = torch.nn.ModuleList()
        self.feat_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))
        self.feat_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))

        
        # self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        # self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        

        self.repeat_obs_gat_project = Linear(2*self.hidden_size,self.hidden_size)
        

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # def decode_all(self, z):
    #     prob_adj = z @ z.t()
    #     return (prob_adj > 0).nonzero(as_tuple=False).t()
    def forward(self, obs_features,feature_mask,feat_features,obs_adjs,data_x ,drop_rate=0,num_layers = 1,sampling_in_loop=False): 
        #print("sampling in loop,", sampling_in_loop)
        if self.categorical:
            # print('inside ctr')
            feature_mask = data_x >self.feat_value_thresh   ### Only take those edges which has value >= 1
        
        
        otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
        otf_edge_index = otf_edge_index.to(self.device)
        otf_edge_features = otf_edge_features.to(self.device)
        fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
        fto_edge_features = otf_edge_features.detach().clone()
        fto_edge_features = fto_edge_features.to(self.device)
        #print(otf_edge_index.shape,otf_edge_features.shape)


        #### Add sampling from for every feature node

        if num_layers == 1:
            obs_features = self.projects_obs[0](obs_features)
            feat_features = self.projects_feat[0](feat_features)
            ### Message passing from feature to observation nodes ####
            #### Here fto_edge_index needs to be index from 0 as well since left side and 
            ### right side of bipartite graph is indexed seperately
            
            if self.fto_sample:
                sfto_edge_index,sfto_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,self.num_feat_node_features, info_flow="fto",num_samples=self.num_feat_samples,data_x = data_x,use_data_x_otf=self.use_data_x_fto,device=self.device)
                obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_convs_f_to_o[0]((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
            else:
                obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_convs_f_to_o[0]((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))

    
            ##### Message passing between observation nodes ####

            for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
                obs_features = self.projects[i](obs_features)
                obs_features_target= obs_features[:size[1]]
                obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
                obs_features = F.elu(obs_features)
                obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
                  #### create obs feature graph 
                #print(i, size)
            feature_mask = feature_mask[:size[1],:]
            new_num_feature_edges = feature_mask.sum()
            #print(new_num_feature_edges,otf_edge_index.shape)
            fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
            fto_edge_features = fto_edge_features[:new_num_feature_edges,:]
            
            otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
            otf_edge_features = otf_edge_features[:new_num_feature_edges,:]
            
            if self.otf_sample and sampling_in_loop: 
                #print("in sampling")
                sotf_edge_index,sotf_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,self.num_feat_node_features,info_flow="otf",num_samples=self.num_obs_samples,data_x = data_x,use_data_x_otf=self.use_data_x_otf,device=self.device)
                feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_convs_o_to_f[0]((obs_features, feat_features),sotf_edge_index,edge_attr=sotf_edge_features)),dim=-1))
            else:
                feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_convs_o_to_f[0]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

            #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
            

            if self.fto_sample and sampling_in_loop:
                sfto_edge_index,sfto_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,self.num_feat_node_features, info_flow="fto",num_samples=self.num_feat_samples,data_x = data_x,use_data_x_otf=self.use_data_x_fto,device=self.device)
                obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
            else:
                obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))
            
            if self.task_type=="link":
                return obs_features,feat_features
            
            obs_features  = self.project_Y(obs_features)
            out = F.log_softmax(obs_features, dim=-1)
            return out,obs_features,feat_features
        
        
        
        
        
        if num_layers == 2:

            for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
                obs_features = self.projects_obs[i](obs_features)
                feat_features = self.projects_feat[i](feat_features)
                
                
                if self.fto_sample and sampling_in_loop:
                    sfto_edge_index,sfto_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,self.num_feat_node_features, info_flow="fto",num_samples=self.num_feat_samples,data_x = data_x,use_data_x_otf=self.use_data_x_fto,device=self.device)
                    obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
                    #sfto_edge_index,sfto_edge_features = sample_otf_edge_index_v0(fto_edge_index,fto_edge_features,num_samples=self.num_feat_samples,data_x = data_x.T,use_data_x_otf=self.use_data_x_fto,device=self.device)   ### remember to send transpose of data.x
                    #obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
                else:
                    obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))

            ##### Message passing between observation nodes ####

                obs_features = self.projects[i](obs_features)
                obs_features_target= obs_features[:size[1]]
                obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
                obs_features = F.elu(obs_features)
                obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
                  #### create obs feature graph 
                
                feature_mask = feature_mask[:size[1],:]
                new_num_feature_edges = feature_mask.sum()
                #print(otf_edge_index.shape,new_num_feature_edges)
                fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
                fto_edge_features = fto_edge_features[:new_num_feature_edges,:]
                
                
                otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
                otf_edge_features = otf_edge_features[:new_num_feature_edges,:]

                if self.otf_sample and sampling_in_loop:#and (self.training or self.otf_sample_testing):  ### either training should be on or sampling is allowed in testing
                    sotf_edge_index,sotf_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,self.num_feat_node_features,info_flow="otf",num_samples=self.num_obs_samples,data_x = data_x,use_data_x_otf=self.use_data_x_otf,device=self.device)
                    feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),sotf_edge_index,edge_attr=sotf_edge_features)),dim=-1))
                #     sotf_edge_index,sotf_edge_features = sample_otf_edge_index_v0(otf_edge_index,otf_edge_features,num_samples=self.num_obs_samples,data_x = data_x,use_data_x_otf=self.use_data_x_otf,device=self.device)
                #     feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),sotf_edge_index,edge_attr=sotf_edge_features)),dim=-1))
                else:
                    feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

            #### Finally re-computing edge embeddings 
                #otf_edge_features = self.edge_embedding_update[i](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))



            #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
            #obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))
            if self.task_type=="link":
                return obs_features,feat_features

            obs_features  = self.project_Y(obs_features)
            out = F.log_softmax(obs_features, dim=-1)
            return out,obs_features,feat_features
        
        
# class ScalableFOGNN(torch.nn.Module):
#     def __init__(self, drop_rate,num_obs_node_features,num_feat_node_features,
#         num_layers, hidden_size, out_channels,heads,device,categorical=False,
#         gnnType='SAGEConv',feat_val_thresh=0.01,otf_sample=False,fto_sample=False,
#         num_obs_samples = 30, num_feat_samples=30,use_data_x_otf= False,use_data_x_fto=False,otf_sample_testing=False):
#         super(ScalableFOGNN, self).__init__()
#         self.device = device
#         self.feat_value_thresh = feat_val_thresh
#         print("setting threshold as ",self.feat_value_thresh)
#         self.heads = heads
#         self.num_layers = num_layers
#         self.drop_rate = drop_rate
#         self.num_obs_node_features = num_obs_node_features
#         self.num_feat_node_features = num_feat_node_features
#         self.hidden_size = hidden_size
#         self.edge_hidden_size = self.hidden_size
#         self.categorical = categorical
#         self.otf_sample= otf_sample
#         #print("Otf sampling, ", otf_sample)
#         self.fto_sample = fto_sample
#         #print("Fto sampling,", fto_sample )
#         self.num_obs_samples = num_obs_samples
#         self.num_feat_samples = num_feat_samples
#         self.use_data_x_otf= use_data_x_otf
#         self.use_data_x_fto= use_data_x_fto
#         self.otf_sample_testing=otf_sample_testing
#         print(otf_sample,fto_sample,num_obs_samples,num_feat_samples,use_data_x_otf,use_data_x_fto)
#         self.projects =torch.nn.ModuleList()
#         self.projects.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))
#         self.projects.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))

#         self.projects_obs =torch.nn.ModuleList()
#         self.projects_obs.append(temp_layer(self.num_obs_node_features,self.hidden_size,self.drop_rate))
#         self.projects_obs.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))
        
#         self.projects_feat =torch.nn.ModuleList()
#         self.projects_feat.append(temp_layer(self.num_feat_node_features,self.hidden_size,self.drop_rate))
#         self.projects_feat.append(temp_layer(self.hidden_size,self.hidden_size,self.drop_rate))


#         self.num_otf_edge_features = 1
#         self.otf_edge_embedding =torch.nn.ModuleList()
#         self.otf_edge_embedding.append(Sequential(Linear(in_features=self.num_otf_edge_features, out_features=self.edge_hidden_size)))
#         self.otf_edge_embedding.append(Sequential(Linear(in_features=self.edge_hidden_size, out_features=self.edge_hidden_size)))
    
#         self.fto_edge_embedding =torch.nn.ModuleList()
#         self.fto_edge_embedding.append(Sequential(Linear(in_features=self.num_otf_edge_features, out_features=self.edge_hidden_size)))
#         self.fto_edge_embedding.append(Sequential(Linear(in_features=self.edge_hidden_size, out_features=self.edge_hidden_size)))
        
# #         self.edge_embedding_update = torch.nn.ModuleList()
# #         self.edge_embedding_update.append(Sequential(
# #                     Linear(in_features=2*self.hidden_size+self.edge_hidden_size, out_features=self.edge_hidden_size),
# #                     ReLU(),
# #                     Linear(self.edge_hidden_size,self.edge_hidden_size)
# #                 ))
# #         self.edge_embedding_update.append(Sequential(
# #                     Linear(in_features=2*self.hidden_size+self.edge_hidden_size, out_features=self.edge_hidden_size),
# #                     ReLU(),
# #                     Linear(self.edge_hidden_size,self.edge_hidden_size)
# #                 ))
        

        
#         self.out_channels = out_channels
#         self.dropout = emb_size=hidden_size
#         self.gin_convs = torch.nn.ModuleList()

#         #print('gnnType', gnnType)
        
#         if(gnnType=='SAGEConv'):
#             self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
#             self.gin_convs.append(SAGEConv(self.hidden_size, self.hidden_size))
        
#         elif(gnnType=='GATConv'):
#             print("using gat")
#             self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size//4,heads= 4,concat=True))
#             self.gin_convs.append(GATConv(self.hidden_size, self.hidden_size,heads=1))
#         elif(gnnType=='GINConv'):
#             self.gin_convs.append(temp_gin_layer(self.hidden_size,self.hidden_size,self.drop_rate))
#             self.gin_convs.append(temp_gin_layer(self.hidden_size,self.hidden_size,self.drop_rate))

#         else:
#             import sys
#             print(gnnType, " not defined")
#             sys.exit(1)

#         self.project_Y = Sequential(
#             Linear(self.hidden_size,self.hidden_size),
#             BatchNorm1d(num_features=self.hidden_size,track_running_stats=False,affine=False),
#             LeakyReLU(negative_slope=0.01),
#             Linear(self.hidden_size,self.out_channels),

#         )
        
#         self.gat_convs_f_to_o = torch.nn.ModuleList()
#         self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
#         self.gat_convs_f_to_o.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads, edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        
#         self.gat_convs_o_to_f = torch.nn.ModuleList()
#         self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
#         self.gat_convs_o_to_f.append(GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2))
        
#         self.repeat_gat_convs_f_to_o  = GATConv(self.hidden_size, self.hidden_size//self.heads,heads=self.heads,edge_dim=self.num_otf_edge_features,add_self_loops=False,dropout=0.2)
        
#         self.obs_gat_project = torch.nn.ModuleList()
#         self.obs_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))
#         self.obs_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))

        
        
#         # self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
#         # self.obs_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))

#         self.feat_gat_project = torch.nn.ModuleList()
#         self.feat_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))
#         self.feat_gat_project.append(temp_layer(2*self.hidden_size,self.hidden_size,self.drop_rate))

        
#         # self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
#         # self.feat_gat_project.append(Linear(2*self.hidden_size,self.hidden_size))
        

#         self.repeat_obs_gat_project = Linear(2*self.hidden_size,self.hidden_size)
 
        
#     def forward(self, obs_features,feature_mask,feat_features,obs_adjs,data_x ,drop_rate=0,num_layers = 1): 

#         if self.categorical:
#             feature_mask = data_x >self.feat_value_thresh   ### Only take those edges which has value >= 1
        
        
#         otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
#         otf_edge_index = otf_edge_index.to(self.device)
#         otf_edge_features = otf_edge_features.to(self.device)
#         fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
#         fto_edge_features = otf_edge_features.detach().clone()
#         fto_edge_features = fto_edge_features.to(self.device)
#         #print(otf_edge_index.shape,otf_edge_features.shape)


#         #### Add sampling from for every feature node

#         if num_layers == 1:
#             obs_features = self.projects_obs[0](obs_features)
#             feat_features = self.projects_feat[0](feat_features)
#             ### Message passing from feature to observation nodes ####
#             #### Here fto_edge_index needs to be index from 0 as well since left side and 
#             ### right side of bipartite graph is indexed seperately
            
#             if self.fto_sample and (self.training or self.otf_sample_testing):
#                 sfto_edge_index,sfto_edge_features = sample_otf_edge_index(fto_edge_index,fto_edge_features,num_samples=self.num_feat_samples,data_x = data_x.T,use_data_x_otf=self.use_data_x_fto,device=self.device)
#                 obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_convs_f_to_o[0]((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
#             else:
#                 obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_convs_f_to_o[0]((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))

    
#             ##### Message passing between observation nodes ####

#             for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
#                 obs_features = self.projects[i](obs_features)
#                 obs_features_target= obs_features[:size[1]]
#                 obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
#                 obs_features = F.elu(obs_features)
#                 obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
#                   #### create obs feature graph 
#                 #print(i, size)
#             feature_mask = feature_mask[:size[1],:]
#             new_num_feature_edges = feature_mask.sum()
#             #print(new_num_feature_edges,otf_edge_index.shape)
#             fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
#             fto_edge_features = fto_edge_features[:new_num_feature_edges,:]
            
#             otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
#             otf_edge_features = otf_edge_features[:new_num_feature_edges,:]
            
#             if self.otf_sample and (self.training or self.otf_sample_testing):
#                 sotf_edge_index,sotf_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,num_samples=self.num_obs_samples,data_x = data_x,use_data_x_otf=self.use_data_x_otf,device=self.device)
#                 feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_convs_o_to_f[0]((obs_features, feat_features),sotf_edge_index,edge_attr=sotf_edge_features)),dim=-1))
#             else:
#                 feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_convs_o_to_f[0]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

#             #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
            

#             if self.fto_sample and (self.training or self.otf_sample_testing):
#                 sfto_edge_index,sfto_edge_features = sample_otf_edge_index(fto_edge_index,fto_edge_features,num_samples=self.num_feat_samples,data_x = data_x.T,use_data_x_otf=self.use_data_x_fto,device=self.device)
#                 obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
#             else:
#                 obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))

#             obs_features  = self.project_Y(obs_features)
#             out = F.log_softmax(obs_features, dim=-1)
#             return out,obs_features,feat_features
        
        
        
        
        
#         if num_layers == 2:

#             for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
#                 obs_features = self.projects_obs[i](obs_features)
#                 feat_features = self.projects_feat[i](feat_features)
                
                
#                 if self.fto_sample and (self.training or self.otf_sample_testing):
#                     sfto_edge_index,sfto_edge_features = sample_otf_edge_index(fto_edge_index,fto_edge_features,num_samples=self.num_feat_samples,data_x = data_x.T,use_data_x_otf=self.use_data_x_fto,device=self.device)   ### remember to send transpose of data.x
#                     obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),sfto_edge_index,edge_attr=sfto_edge_features)),dim=-1))
#                 else:
#                     obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))

#             ##### Message passing between observation nodes ####

#                 obs_features = self.projects[i](obs_features)
#                 obs_features_target= obs_features[:size[1]]
#                 obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
#                 obs_features = F.elu(obs_features)
#                 obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
#                   #### create obs feature graph 
                
#                 feature_mask = feature_mask[:size[1],:]
#                 new_num_feature_edges = feature_mask.sum()
#                 #print(otf_edge_index.shape,new_num_feature_edges)
#                 fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
#                 fto_edge_features = fto_edge_features[:new_num_feature_edges,:]
                
                
#                 otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
#                 otf_edge_features = otf_edge_features[:new_num_feature_edges,:]

#                 if self.otf_sample and (self.training or self.otf_sample_testing):  ### either training should be on or sampling is allowed in testing
#                     sotf_edge_index,sotf_edge_features = sample_otf_edge_index(otf_edge_index,otf_edge_features,num_samples=self.num_obs_samples,data_x = data_x,use_data_x_otf=self.use_data_x_otf,device=self.device)
#                     feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),sotf_edge_index,edge_attr=sotf_edge_features)),dim=-1))
#                 else:
#                     feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

#             #### Finally re-computing edge embeddings 
#                 #otf_edge_features = self.edge_embedding_update[i](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))



#             #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
#             #obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=fto_edge_features)),dim=-1))


#             obs_features  = self.project_Y(obs_features)
#             out = F.log_softmax(obs_features, dim=-1)
#             return out,obs_features,feat_features
       
# 
#    def forward(self, obs_features,feature_mask,feat_features,obs_adjs,data_x ,drop_rate=0,num_layers = 1): 

#         if self.categorical:
#             feature_mask = data_x >self.feat_value_thresh   ### Only take those edges which has value >= 1
        
#         if num_layers == 1:
#             otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
#             otf_edge_index = otf_edge_index.to(self.device)
#             otf_edge_features = otf_edge_features.to(self.device)
#             fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
#             fto_edge_index = fto_edge_index
#             fto_edge_features = otf_edge_features

#             obs_features = self.projects_obs[0](obs_features)
#             feat_features = self.projects_feat[0](feat_features)
#             otf_edge_features = self.edge_embedding[0](otf_edge_features)
#             ### Message passing from feature to observation nodes ####
#             #### Here fto_edge_index needs to be index from 0 as well since left side and 
#             ### right side of bipartite graph is indexed seperately

#             obs_features = self.obs_gat_project[0](torch.cat((obs_features,self.gat_convs_f_to_o[0]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features)),dim=-1))

    
#             ##### Message passing between observation nodes ####

#             for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):
#                 obs_features = self.projects[i](obs_features)
#                 obs_features_target= obs_features[:size[1]]
#                 obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
#                 obs_features = F.elu(obs_features)
#                 obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
#                   #### create obs feature graph 

#             new_num_feature_edges = feature_mask[:size[1],:].sum()
#             #print(new_num_feature_edges,otf_edge_index.shape)
#             otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
#             fto_edge_index = fto_edge_index[:,:new_num_feature_edges]

#             otf_edge_features = otf_edge_features[:new_num_feature_edges,:]

#             ### Message passing from observation to feature nodes 
#             #feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_linear_project_feat[0](self.gat_convs_o_to_f[0]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features))),dim=-1))
#             feat_features = self.feat_gat_project[0](torch.cat((feat_features,self.gat_convs_o_to_f[0]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

#             #### Finally re-computing edge embeddings 
#             otf_edge_features = self.edge_embedding_update[0](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))

#             #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
#             obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features)),dim=-1))


#             obs_features  = self.project_Y(obs_features)
#             out = F.log_softmax(obs_features, dim=-1)
#             return out,obs_features,feat_features
        
        
#         if num_layers == 2:
#         #### create obs feature graph 
#             otf_edge_index,otf_edge_features =  create_otf_edges(data_x,feature_mask)
#             otf_edge_index = otf_edge_index.to(self.device)
#             otf_edge_features = otf_edge_features.to(self.device)
#             fto_edge_index = torch.stack([otf_edge_index[1], otf_edge_index[0]], dim=0)
#             fto_edge_index = fto_edge_index
#             otf_edge_features = self.edge_embedding[0](otf_edge_features)
#             for i, (obs_edge_index,e_id,size) in enumerate(obs_adjs):

#                 obs_features = self.projects_obs[i](obs_features)
#                 feat_features = self.projects_feat[i](feat_features)
#                 #otf_edge_features = self.edge_embedding[i](otf_edge_features)
#             ### Message passing from feature to observation nodes ####
#             #### Here fto_edge_index needs to be index from 0 as well since left side and right side of bipartite graph is indexed seperately
# #                obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_linear_project_obs[i](self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features))),dim=-1))
#                 obs_features = self.obs_gat_project[i](torch.cat((obs_features,self.gat_convs_f_to_o[i]((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features)),dim=-1))


#             ##### Message passing between observation nodes ####

#                 obs_features = self.projects[i](obs_features)
#                 obs_features_target= obs_features[:size[1]]
#                 obs_features = self.gin_convs[i]((obs_features, obs_features_target), obs_edge_index)
#                 obs_features = F.elu(obs_features)
#                 obs_features = F.dropout(obs_features, p=self.drop_rate)  ## This will be equal to batch size of nodes
#                   #### create obs feature graph 

#                 new_num_feature_edges = feature_mask[:size[1],:].sum()
#                 otf_edge_index = otf_edge_index[:,:new_num_feature_edges]
#                 fto_edge_index = fto_edge_index[:,:new_num_feature_edges]
#                 otf_edge_features = otf_edge_features[:new_num_feature_edges,:]
#             ### Message passing from observation to feature nodes 
#                 #feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_linear_project_feat[i](self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features))),dim=-1))
#                 feat_features = self.feat_gat_project[i](torch.cat((feat_features,self.gat_convs_o_to_f[i]((obs_features, feat_features),otf_edge_index,edge_attr=otf_edge_features)),dim=-1))

#             #### Finally re-computing edge embeddings 
#                 #otf_edge_features = self.edge_embedding_update[i](torch.cat((otf_edge_features,obs_features[otf_edge_index[0]],feat_features[otf_edge_index[1]]),dim=1))



#             #### Sending messages back to observations nodes so that they have messages from nodes to which they have same feature enabled
#             #obs_features = self.repeat_obs_gat_project(torch.cat((obs_features,self.repeat_gat_convs_f_to_o((feat_features, obs_features),fto_edge_index,edge_attr=otf_edge_features)),dim=-1))


#             obs_features  = self.project_Y(obs_features)
#             out = F.log_softmax(obs_features, dim=-1)
#             return out,obs_features,feat_features
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
