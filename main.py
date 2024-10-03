import torch
import logging
import argparse
import random
import numpy as np
from logger.logger import create_logger
from loader.loader import create_loader
from models.master import create_model
from train.train import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_number', type=int, default=0, help='Number of the experiment')
    parser.add_argument('--name', type=str, default="CartNet", help="name of the Wandb experiment" )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=16, help="Batch Accumulation")
    parser.add_argument("--dataset", type=str, default="ADP", help="Dataset name. Available: ADP, Jarvis, MaterialsProject")
    parser.add_argument("--dataset_path", type=str, default="./dataset/ADP_DATASET/")
    parser.add_argument("--figshare_target", type=str, default="formation_energy_peratom", help="Figshare dataset target")
    parser.add_argument("--wandb_project", type=str, default="ADP", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="aiquaneuro", help="Name of the wand entity")
    parser.add_argument("--loss", type=str, default="MAE", help="Loss function")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup")
    parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=256, help="Input dimension")
    parser.add_argument("--num_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument('--augment', action='store_true', help='Hydrogens')
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
    parser.add_argument("--no_standarize_temp", action="store_false", help="Standarize temperature")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
    parser.add_argument("--threads", type=int, default= 8, help="Number of threads")
    
    set_cfg(cfg)

    args, _ = parser.parse_known_args()
    cfg.exp_number = args.exp_number
    cfg.seed = cfg.exp_number
    cfg.run_dir = "results/"+cfg.name+"/"+str(cfg.exp_number)
    cfg.dataset.task_type = "regression"
    cfg.name = args.name
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset = args.dataset
    cfg.dataset_path = args.dataset_path
    cfg.figshare_name = args.figshare_name
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.loss = args.loss
    cfg.optim.max_epoch = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.warmup = args.warmup
    cfg.model = args.model
    cfg.max_neighbours = None if cfg.model== "CartNet" else args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.num_rbf = args.num_rbf
    cfg.augment = False if cfg.model in ["icomformer", "ecomformer"] else args.augment
    cfg.invariant = args.invariant
    cfg.use_temp = False if dataset != "ADP" else args.disable_temp
    cfg.standarize_temp = args.no_standarize_temp
    cfg.envelope = args.disable_envelope
    cfg.use_H = args.disable_H

    torch.set_num_threads(args.threads)

    set_printing()

    #Seed
    seed_everything(cfg.seed)

    logging.info(f"Experiment will be saved at: {cfg.run_dir}")

    loaders = create_loader()

    model = create_model()

    logging.info(model)
    cfg.params_count = params_count(model)
    logging.info(f"Number of parameters: {cfg.params_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    loggers = create_logger()
    train(model, loaders, optimizer, loggers)









