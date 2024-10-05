import torch
import logging
import argparse
import pickle
from tqdm import tqdm
from logger.logger import create_logger
from loader.loader import create_loader
from models.master import create_model
from train.train_irene import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing


def inference(model, loader):
    from train.metrics import compute_loss, compute_3D_IoU
    model.eval()
    
    with torch.no_grad():
        inference_output = {"pred": [], "true": [], "refcode": [], "pos": [], "atoms": [], "iou": [], "mae": []}
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to("cuda:0")
            inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
            inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
            inference_output["refcode"].append(batch.refcode)
            _pred, _true = model(batch)
            inference_output["pred"].append(_pred.detach().to("cpu"))
            inference_output["true"].append(_true.detach().to("cpu"))
            inference_output["iou"].append(compute_3D_IoU(_pred, _true).detach().to("cpu"))
            inference_output["mae"].append(compute_loss(_pred, _true)[0].detach().to("cpu"))
            
        pickle.dump(inference_output, open(cfg.inference_output, "wb"))

def montecarlo(model, loader):
    from train.metrics import compute_loss, compute_3D_IoU
    import roma

    model.eval()
    iou_montecarlo = []
    mae_montecarlo = []
    with torch.no_grad():
        for i in range(100):
            inference_output = {"pred": [], "true": [], "refcode": [], "pos": [], "atoms": [], "iou": [], "mae": []}
            for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
                batch_copy = batch.clone()
                batch.to("cuda:0")
                inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
                inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
                inference_output["refcode"].append(batch.refcode)
                pseudo_true, _ = model(batch)
                R = roma.utils.random_rotmat(size=1, device=batch.x.device).squeeze(0)
                pseudo_true =  R.transpose(-1,-2) @ pseudo_true @ R
                batch_copy.to("cuda:0")
                batch_copy.cart_dir = batch_copy.cart_dir @ R
                pred, _ = model(batch_copy)
                inference_output["pred"].append(pred.detach().to("cpu"))
                inference_output["true"].append(pseudo_true.detach().to("cpu"))
                inference_output["iou"].append(compute_3D_IoU(_pred, _true).detach().to("cpu"))
                inference_output["mae"].append(compute_loss(_pred, _true)[0].detach().to("cpu"))
            pickle.dump(inference_output, open(cfg.inference_output.replace(".pkl", "_montecarlo_"+str(i)+".pkl"), "wb"))
            iou_montecarlo+=inference_output["iou"]
            mae_montecarlo+=inference_output["mae"]
    
    iou_montecarlo = torch.cat(iou_montecarlo, dim=0)
    mae_montecarlo = torch.cat(mae_montecarlo, dim=0)

    logging.info(f"Montecarlo IoU: {iou_montecarlo.mean().item()}+/-{iou_montecarlo.std().item()}")
    logging.info(f"Montecarlo MAE: {mae_montecarlo.mean().item()}+/-{mae_montecarlo.std().item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment')
    parser.add_argument('--name', type=str, default="CartNet", help="name of the Wandb experiment" )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=16, help="Batch Accumulation")
    parser.add_argument("--dataset", type=str, default="ADP", help="Dataset name. Available: ADP, Jarvis, MaterialsProject")
    parser.add_argument("--dataset_path", type=str, default="./dataset/ADP_DATASET/")
    parser.add_argument("--inference", action="store_true", help="Inference")
    parser.add_argument("--montecarlo", action="store_true", help="Montecarlo")
    parser.add_argument("--weighs_path", type=str, default=None, help="Path to the weights of the model")
    parser.add_argument("--inference_output", type=str, default="./inference.pkl", help="Path to the inference output")
    parser.add_argument("--figshare_target", type=str, default="formation_energy_peratom", help="Figshare dataset target")
    parser.add_argument("--wandb_project", type=str, default="ADP", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="aiquaneuro", help="Name of the wandb entity")
    parser.add_argument("--loss", type=str, default="MAE", help="Loss function")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.01, help="Warmup")
    parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=256, help="Input dimension")
    parser.add_argument("--dim_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument('--augment', action='store_true', help='Hydrogens')
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
    parser.add_argument("--no_standarize_temp", action="store_false", help="Standarize temperature")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
    parser.add_argument("--threads", type=int, default= 8, help="Number of threads")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers")
    
    set_cfg(cfg)

    args, _ = parser.parse_known_args()
    cfg.seed = args.seed
    cfg.name = args.name
    cfg.run_dir = "results/"+cfg.name+"/"+str(cfg.seed)
    cfg.dataset.task_type = "regression"
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset.name = args.dataset
    cfg.dataset_path = args.dataset_path
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.loss = args.loss
    cfg.optim.max_epoch = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.warmup = args.warmup
    cfg.model = args.model
    cfg.max_neighbours = False if cfg.model== "CartNet" else args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.augment = False if cfg.model in ["icomformer", "ecomformer"] else args.augment
    cfg.invariant = args.invariant
    cfg.use_temp = False if cfg.dataset.name != "ADP" else args.disable_temp
    cfg.standarize_temp = args.no_standarize_temp
    cfg.envelope = args.disable_envelope
    cfg.use_H = args.disable_H
    cfg.workers = args.workers

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

    if args.inference:
        assert args.weighs_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Inference only for ADP dataset"
        ckpt = torch.load(args.weighs_path)
        model.load_state_dict(ckpt["model_state"])
        cfg.inference_output = args.inference_output
        inference(model, loaders[-1])
    elif args.montecarlo:
        assert args.weighs_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Montecarlo only for ADP dataset"
        montecarlo(model, loaders[-1])
    else:
        train(model, loaders, optimizer, loggers)










