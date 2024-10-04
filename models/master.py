import torch
from torch_geometric.graphgym.config import cfg


def create_model(invariant=False):
   
    if cfg.model_name == "CartNet":
        from models.cartnet import CartNet
        model = CartNet(dim_in=cfg.dim_in, 
                        num_layers=cfg.num_layers, 
                        invariant=cfg.invariant, 
                        temperature=cfg.use_temp, 
                        use_cutoff=cfg.envelope, 
                        cholesky=True if cfg.dataset.name == "ADP" else False
                    ).to("cuda:0")
    
    elif cfg.model_name == "ecomformer":
        from models.comformer import eComformer
        model = eComformer(dim_in=cfg.dim_in).to("cuda:0")

    elif cfg.model_name == "icomformer":
        from models.comformer import iComformer
        model = iComformer(dim_in=cfg.dim_in).to("cuda:0")
    else:
        raise Exception("Model not implemented")
    return model