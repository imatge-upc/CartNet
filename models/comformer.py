import torch
from torch import nn
from network.transformer import ComformerConv, ComformerConv_edge, ComformerConvEqui
from models.cartnet import Cholesky_head


# Modified implementation from Comfomer
# https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine




class eComformer(nn.Module): 
    """att pyg implementation."""

    def __init__(self, dim_in):
        """Set up att modules."""
        super().__init__()
        self.dim_in = dim_in
        self.embedding = nn.Embedding(119,self.dim_in)
        self.temperature_proj_atom = nn.Linear(1,self.dim_in, bias=True)
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=self.dim_in,
            ),
            nn.Linear(self.dim_in, self.dim_in),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=self.dim_in, out_channels=self.dim_in, heads=1, edge_dim=self.dim_in)
                for _ in range(3)
            ]
        )

        self.equi_update = ComformerConvEqui(in_channels=self.dim_in, out_channels=self.dim_in, edge_dim=self.dim_in, use_second_order_repr=True)

        self.cholesky = Cholesky_head(self.dim_in, 6)

    def forward(self, data) -> torch.Tensor:
        node_features = self.embedding(data.x) + self.temperature_proj_atom(data.temperature.unsqueeze(-1))[data.batch]
        n_nodes = node_features.shape[0]
        edge_feat = -0.75 / data.cart_dist
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 
        data.x = self.att_layers[2](node_features, data.edge_index, edge_features)

        return self.cholesky(data)




class iComformer(nn.Module): # iComFormer
    """att pyg implementation."""

    def __init__(self, dim_in):
        """Set up att modules."""
        super().__init__()
        self.dim_in = dim_in
        self.embedding = nn.Embedding(119,self.dim_in)
        self.temperature_proj_atom = nn.Linear(1,self.dim_in, bias=True)
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=self.dim_in,
            ),
            nn.Linear(self.dim_in, self.dim_in),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=self.dim_in,
            ),
            nn.Linear(self.dim_in, self.dim_in),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=self.dim_in, out_channels=self.dim_in, heads=1, edge_dim=self.dim_in)
                for _ in range(4)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(in_channels=self.dim_in, out_channels=self.dim_in, heads=1, edge_dim=self.dim_in)
        
        self.cholesky = Cholesky_head(self.dim_in, 6)

    def forward(self, data) -> torch.Tensor:
        node_features = self.embedding(data.x) + self.temperature_proj_atom(data.temperature.unsqueeze(-1))[data.batch]
        edge_feat = -0.75 / data.cart_dist # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.cell, dim=-1) # [num_batch, 3]
        edge_nei_len = edge_nei_len[data.batch[data.edge_index[0]]] # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.cell[data.batch[data.edge_index[0]]], data.cart_dir.unsqueeze(1).repeat(1, 3, 1)) # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat).squeeze(1)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features) 
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        data.x = self.att_layers[3](node_features, data.edge_index, edge_features)

        return self.cholesky(data)




