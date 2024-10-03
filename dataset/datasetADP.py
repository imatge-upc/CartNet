import torch
from torch_geometric.data import Dataset, Batch
from tqdm import tqdm
import os.path as osp
import numpy as np
import torch.nn.functional as F
import roma
from dataset.utils import optmize_lattice
# from utils import radius_graph_pbc


class DatasetADP(Dataset):
    def __init__(self, root="/scratch/g1alexs/PBC_DATASET_SINGLE_MOL/", file_names=None, standarize_temp=True, hydrogens = True, augment = False, optimize_cell=False):
        self.original_root = root
        self.file_names = file_names
        self.standarize_temp = standarize_temp
        self.mean_temp = torch.tensor(192.1785) #training temp mean
        self.std_temp = torch.tensor(81.2135) #training temp std
        self.hydrogens = hydrogens
        self.augment = augment
        self.optimize_cell = optimize_cell

        with open(file_names, 'r') as file:
            self.file_names = [line.strip() for line in file.readlines()]

        super(DatasetPBC, self).__init__(self.original_root, None, None)
    def len(self):
        return len(self.file_names)
    
    def processed_file_names(self):
        return self.file_names
    
    def augment_data(self, data):
        
        R = roma.utils.random_rotmat(size=1, device=data.x.device).squeeze(0)
        data.y = R.transpose(-1,-2) @ data.y @ R      
        data.cart_dir = data.cart_dir @ R
        data.cell = data.cell @ R

        return data
    
    def get(self, idx):
        data = torch.load(osp.join(self.original_root,self.file_names[idx]+".pt"))

        if self.standarize_temp:
            data.temperature_og = data.temperature
            data.temperature = ((data.temperature - self.mean_temp) / self.std_temp)
        
        
        
        data.non_H_mask = data.x != 1

        
        if not self.hydrogens:
            #Remove hydrogens
            data.x = data.x[data.non_H_mask]
            data.y = data.y[data.non_H_mask]
            data.pos = data.pos[data.non_H_mask]
           
            atoms = torch.arange(0,data.non_H_mask.shape[0])[data.non_H_mask]
            bool_mask_source = torch.isin(data.edge_index[0], atoms )
            bool_mask_target = torch.isin(data.edge_index[1], atoms )
            bool_mask_combined = bool_mask_source & bool_mask_target
            data.edge_index = data.edge_index[:, bool_mask_combined]

            
            node_mapping = {old: new for new, old in enumerate(atoms.tolist())}
        
            
            data.edge_index = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] for edge in data.edge_index.t()]).t()

            
            data.edge_attr = data.edge_attr[bool_mask_combined, :]
            data.non_H_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        


        data.cart_dist = data.edge_attr[:,-1]
        data.cart_dir = data.edge_attr[:,:-1]
    
        if self.optimize_cell:
            data.cell_og = data.cell
            data.cell, rotation_matrix = optmize_lattice(data.cell.squeeze(0))
            data.cell = data.cell.unsqueeze(0)
            data.cart_dir = data.cart_dir @ rotation_matrix
            data.y = rotation_matrix.transpose(-1,-2) @ data.y @ rotation_matrix


        if self.augment:
            data = self.augment_data(data)
        
        return data

    
