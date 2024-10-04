import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter
from models.utils import ExpNormalSmearing, CosineCutoff


class CartNet(torch.nn.Module):
    """
    CartNet's architecture. 
    From "CartNet: Cartesian Encoding for Anisotropic Displacement Parameters Estimation"

    """

    def __init__(self, 
        dim_in: int, 
        dim_rbf: int, 
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True, 
        use_envelope: bool = True,
        cholesky: bool = True):
        super().__init__()
    
        self.encoder = Encoder(dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, temperature=temperature)
        self.dim_in = dim_in

        layers = []
        for _ in range(num_layers):
            layers.append(CartNet_layer(
                dim_in=dim_in,
                use_envelope=use_envelope,
            ))
        self.layers = torch.nn.Sequential(*layers)

        if cholesky:
            self.head = Cholesky_head(dim_in)
        else:
            self.head = Scalar_head(dim_in)
        
    def forward(self, batch):
        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)
        
        pred, true = self.head(batch)
        
        return pred,true

class Encoder(torch.nn.Module):
    """
    Atom and Edge encoder from CartNet
    """
    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False, 
        temperature: bool = True,
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.temperature = temperature
        self.embedding = nn.Embedding(119, self.dim_in*2)
        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in*2, bias=True)
        else:
            self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)
        
       
        self.encoder_atom = nn.Sequential(self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3
        
        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0,radius,dim_rbf,False)  
        
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, batch):

        if self.temperature:
            x = self.embedding(batch.x) + self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        else:
            x = self.embedding(batch.x) + self.bias
        
        batch.x = self.encoder_atom(x)

        if cfg.invariant:
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch

class CartNet_layer(pyg_nn.conv.MessagePassing):
    """
    CartNet layer
    """
    
    def __init__(self, 
        dim_in: int, 
        use_envelope: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True) 
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        
        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, cfg.radius)
        

    def forward(self, batch):

        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        
        x_in = x
        e_in = e

        x, e = self.propagate(edge_index,
                                Xx=x, Ee=e,
                                He=dist,
                            )
 
        batch.x = self.activation(x) + x_in
        
        batch.edge_attr = e_in + e 

        return batch


    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))
        
        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1)*e_ij
        else:
            sigma_ij = e_ij
        
        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]  

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        

        out = scatter(sigma_ij*sender, index, 0, None, dim_size,
                                   reduce='sum')

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)
       
        e_out = self.e
        del self.e

        return x, e_out

class Cholesky_head(torch.nn.Module):
    """
    Cholesky head
    """
    def __init__(self, 
        dim_in: int
    ):
        super(Cholesky_head, self).__init__()
        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2),
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 6))

    def forward(self, batch):
        pred = self.MLP(batch.x[batch.non_H_mask])

        diag_elements = F.softplus(pred[:, :3])

        i,j = torch.tensor([0,1,2,0,0,1]), torch.tensor([0,1,2,1,2,2])
        L_matrix = torch.zeros(pred.size(0),3,3, device=pred.device)
        L_matrix[:,i[:3], i[:3]] = diag_elements
        L_matrix[:,i[3:], j[3:]] = pred[:,3:]

        U = torch.bmm(L_matrix.transpose(1, 2), L_matrix)
        
        return U, batch.y

class Scalar_head(torch.nn.Module):
    """
    Scalar head
    """
    def __init__(self,
        dim_in
    ):
        super(Scalar_head, self).__init__()

        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2), 
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 1))

    def forward(self, batch):
        dim_size = int(batch.batch.max().item() + 1)
        batch.x = self.MLP(batch.x)
        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size).squeeze(-1)
        return batch.x, batch.y

