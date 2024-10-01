import torch.nn as nn
import logging
from dataset.datasetADP import DatasetADP
from logger.logger import create_logger
import time
from models.cartnet import CartNet
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from torch_geometric import seed_everything
import os
import os.path as osp
import glob
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing
from metrics import get_error_volume, compute_3D_IoU
from torch.utils.data import Subset
from sklearn.metrics import r2_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_number', type=int, default=0, help='Number of the experiment')
    parser.add_argument('--name', type=str, default="CartNet", help="name of the Wandb experiment" )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=16, help="Batch Accumulation")
    parser.add_argument("--dataset", type=str, default="ADP", help="Dataset name. Available: ADP, Javris, MaterialsProject")
    parser.add_argument("--figshare_name", type=str, default="dft_3d_2021", help="Figshare dataset name (Only for Jarvis and Materials Project)")
    parser.add_argument("--figshare_target", type=str, default="formation_energy_peratom", help="Figshare dataset target")
    parser.add_argument("--wandb_project", type=str, default="ADP", help="Wandb project name")
    parser.add_argument("--loss", type=str, default="MAE", help="Loss function")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup")
    parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=256, help="Input dimension")
    parser.add_argument("--num_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument('--augment', action='store_true', help='Hydrogens')
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
    parser.add_argument("--standarize_temp", action="store_true", help="Standarize temperature")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
    
    set_cfg(cfg)

    args, _ = parser.parse_known_args()
    cfg.exp_number = args.exp_number
    cfg.run_dir = "results/"+cfg.name+"/"+str(cfg.exp_number)
    cfg.dataset.task_type = "regression"
    cfg.name = args.name
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset = args.dataset
    cfg.figshare_name = args.figshare_name
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.loss = args.loss
    cfg.epochs = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.warmup = args.warmup
    cfg.model = args.model
    cfg.max_neighbours = args.max_neighbours
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.num_rbf = args.num_rbf
    cfg.augment = args.augment
    cfg.invariant = args.invariant
    cfg.disable_temp = args.disable_temp
    cfg.standarize_temp = args.standarize_temp
    cfg.disable_envelope = args.disable_envelope
    cfg.disable_H = args.disable_H

    set_printing()


