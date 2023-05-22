from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch
from torch import Tensor
from torch_scatter import scatter_add

def row_normalize(edge_index, edge_weight, n_nodes):
    row_sum = get_adj_row_sum(edge_index, edge_weight, n_nodes)
    row_idx = edge_index[0]
    return edge_weight / row_sum[row_idx]
def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return DAD

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.adaptive_diffusion = False

    def propagate(
        self, x: Tensor, edge_index: Adj, mask: Tensor,
        edge_weight: OptTensor = None
    ) -> Tensor:
        
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]
        
        n_nodes = x.shape[0]
        adj = None
        for _ in range(self.num_iterations):
            if self.adaptive_diffusion or adj is None:
                adj = self.get_propagation_matrix(out, edge_index, edge_weight, n_nodes)
            # Diffuse current features
            #print(adj.shape,out.shape)
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, edge_weight, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_weight = edge_weight if edge_weight else torch.ones(edge_index.shape[1]).to(edge_index.device)
        edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)

        return adj


class LearnableFeaturePropagation(FeaturePropagation):
    def __init__(self, num_features: int, num_iterations: int, attention_dim: int, attention_type: str):
        super(LearnableFeaturePropagation, self).__init__(num_iterations)
        self.num_iterations = num_iterations
        self.attention_type = attention_type
        self.attention_layer = self._set_attention_layer(num_features, attention_dim)
        self.adaptive_diffusion = True

    def _set_attention_layer(self, num_features, attention_dim):
        if self.attention_type == 'transformer':
            layer = SpGraphTransAttentionLayer(num_features, attention_dim, concat=True, edge_weights=None)
        elif self.attention_type == 'restricted':
            layer = RestrictedAttentionLayer(num_features, attention_dim, concat=True, edge_weights=None)
        else:
            raise NotImplementedError
        return layer

    def get_propagation_matrix(self, x, edge_index, edge_weight, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_weight = edge_weight if edge_weight else torch.ones(edge_index.shape[1]).to(edge_index.device)
        self.attention_layer = self.attention_layer.to(edge_index.device)
        attention_score, _ = self.attention_layer(x, edge_index, edge_weight)
        edge_weight = attention_score.mean(dim=1).squeeze()
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)

        return adj
