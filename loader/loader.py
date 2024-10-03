import torch
from dataset.datasetADP import DatasetADP
from dataset.figshare_dataset import Figshare_Dataset
from dataset.utils import compute_knn
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
import random
import os.path as osp

def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    if cfg.dataset.name == "ADP":
        refcodes = [osp.join(cfg.dataset_path,"train_files.csv"), osp.join(cfg.dataset_path,"val_files.csv"), osp.join(cfg.dataset_path,"test_files.csv")]
        if cfg.model_name in ["icomformer", "ecomformer"]:
            assert cfg.max_neighbours is not None, "max_neighbours are needed for e/iComformer"
            cfg.dataset_path = compute_knn(cfg.max_neighbours, cfg.radius, cfg.path, refcodes)

        optimize_cell = True if cfg.model_name == "icomformer" else False
        dataset_train, dataset_val, dataset_test = (DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[0], Hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, augment=cfg.augment, optimize_cell=optimize_cell),
                                                    DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[1], Hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, optimize_cell=optimize_cell),
                                                    DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[2], Hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, optimize_cell=optimize_cell) 
                                                )
    elif cfg.dataset.name == "jarvis" or cfg.dataset.name=="megnet":
        from jarvis.db.figshare import data as jdata
        from dataset.figshare_dataset import Figshare_Dataset
        import math
        import pandas as pd


        if cfg.dataset.name == "jarvis":
            cfg.dataset.name = "dft_3d_2021"

        seed = 123 #PotNet uses seed=123 for the comparative table
        target = cfg.jarvis_target
        if cfg.jarvis_target in ["shear modulus", "bulk modulus"] and cfg.dataset.name == "megnet":
            import pickle as pk
            target = cfg.jarvis_target
            if cfg.jarvis_target == "bulk modulus":
                data_train = pk.load(open("./dataset/megnet/bulk_megnet_train.pkl", "rb"))
                data_val = pk.load(open("./dataset/megnet/bulk_megnet_val.pkl", "rb"))
                data_test = pk.load(open("./dataset/megnet/bulk_megnet_test.pkl", "rb"))
            elif cfg.jarvis_target == "shear modulus":
                data_train = pk.load(open("./dataset/megnet/shear_megnet_train.pkl", "rb"))
                data_val = pk.load(open("./dataset/megnet/shear_megnet_val.pkl", "rb"))
                data_test = pk.load(open("./dataset/megnet/shear_megnet_test.pkl", "rb"))
            targets_train = []
            dat_train = []
            targets_val = []
            dat_val = []
            targets_test = []
            dat_test = []
            for split, datalist, targets in zip([data_train, data_val, data_test], 
                                            [dat_train, dat_val, dat_test],
                                            [targets_train, targets_val, targets_test]):
                for i in split:
                    if (
                        i[target] is not None
                        and i[target] != "na"
                        and not math.isnan(i[target])
                    ):
                        datalist.append(i)
                        targets.append(i)
            
        else:
            data = jdata(cfg.jarvis_name)
            dat = []
            all_targets = []
            for i in data:
                if isinstance(i[target], list):
                    all_targets.append(torch.tensor(i[target]))
                    dat.append(i)

                elif (
                        i[target] is not None
                        and i[target] != "na"
                        and not math.isnan(i[target])
                ):
                    dat.append(i)
                    all_targets.append(i[target])
            
            ids_train, ids_val, ids_test = create_train_val_test(dat, seed=seed) 
            dat_train = [dat[i] for i in ids_train]
            dat_val = [dat[i] for i in ids_val]
            dat_test = [dat[i] for i in ids_test]
            targets_train = [all_targets[i] for i in ids_train]
            targets_val = [all_targets[i] for i in ids_val]
            targets_test = [all_targets[i] for i in ids_test]
        
        radius = cfg.cutoff
        prefix = cfg.jarvis_name+"_"+str(radius)+"_"+str(cfg.max_neighbours)+"_"+target+"_"+str(seed)
        dataset_train = Figshare_Dataset(root=cfg.dataset_path, data=dat_train, targets=targets_train, radius=radius, name=prefix+"_train")
        dataset_val = Figshare_Dataset(root=cfg.dataset_path, data=dat_val, targets=targets_val, radius=radius, name=prefix+"_val")
        dataset_test = Figshare_Dataset(root=cfg.dataset_path, data=dat_test, targets=targets_test, radius=radius, name=prefix+"_test")
    else:
        raise Exception("Dataset not implemented")
    
    # cfg.batch = 32
    loaders = [
        DataLoader(dataset_train, batch_size=cfg.batch, persistent_workers=True,
                                  shuffle=True, num_workers=6,
                                  pin_memory=True),
        DataLoader(dataset_val, batch_size=cfg.batch, persistent_workers=True,
                                    shuffle=False, num_workers=6,
                                    pin_memory=True),
        DataLoader(dataset_test, batch_size=1 if cfg.dataset.name == "ADP" else cfg.batch, persistent_workers=True,
                                    shuffle=False, num_workers=6,
                                    pin_memory=True)
    ]
    
    return loaders



def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test