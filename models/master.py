# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch_geometric.graphgym.config import cfg


def create_model():
    """
    Creates and returns a model based on the configuration specified in `cfg`.

    Returns:
    model: An instance of the specified model class, moved to the CUDA device.
    Raises:
    Exception: If the specified model in `cfg.model` is not implemented.
    Notes:
    - If `cfg.model` is "CartNet", it imports and initializes a `CartNet` model with parameters from `cfg`.
    - If `cfg.model` is "ecomformer", it imports and initializes an `eComformer` model, ensuring the dataset is "ADP".
    - If `cfg.model` is "icomformer", it imports and initializes an `iComformer` model, ensuring the dataset is "ADP".
    """
   
    if cfg.model == "CartNet":
        from models.cartnet import CartNet
        model = CartNet(dim_in=cfg.dim_in,
                        dim_rbf=cfg.dim_rbf, 
                        num_layers=cfg.num_layers, 
                        invariant=cfg.invariant, 
                        temperature=cfg.use_temp, 
                        use_envelope=cfg.envelope,
                        atom_types=cfg.use_atom_types,
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