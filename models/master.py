import torch
from torch_geometric.graphgym.config import cfg


def create_model(invariant=False):
   
    if cfg.model == "CartNet":
        from models.cartnet import CartNet
        model = CartNet(dim_in=cfg.dim_in,
                        dim_rbf=cfg.dim_rbf, 
                        num_layers=cfg.num_layers, 
                        invariant=cfg.invariant, 
                        temperature=cfg.use_temp, 
                        use_envelope=cfg.envelope, 
                        cholesky=True if cfg.dataset.name == "ADP" else False
                    ).to("cuda:0")
    
    elif cfg.model == "ecomformer":
        from models.comformer import eComformer
        assert cfg.dataset.name == "ADP", "eComformer only for ADP dataset"
        model = eComformer(dim_in=cfg.dim_in).to("cuda:0")

    elif cfg.model == "icomformer":
        from models.comformer import iComformer
        assert cfg.dataset.name == "ADP", "iComformer only for ADP dataset"
        model = iComformer(dim_in=cfg.dim_in).to("cuda:0")
    else:
        raise Exception("Model not implemented")
    return model