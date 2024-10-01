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
    def __init__(self, root="/scratch/g1alexs/PBC_DATASET_SINGLE_MOL/", file_names=None, standarize_temp=True, Hydrogens = True, volume = True, augment = False, optimize_cell=False):
        self.original_root = root
        self.file_names = file_names
        self.standarize_temp = standarize_temp
        self.mean_temp = torch.tensor(192.1785)
        self.std_temp = torch.tensor(81.2135)
        self.Hydrogens = Hydrogens
        self.volume = volume
        self.augment = augment
        self.optimize_cell = optimize_cell

        with open(file_names, 'r') as file:
            self.file_names = [line.strip() for line in file.readlines()]

        super(DatasetPBC, self).__init__(self.original_root, None, None)
    def len(self):
        return len(self.file_names)
    
    def processed_file_names(self):
        return self.file_names
    
    def augmentation(self, data):
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
        
        ## Remove Hydrogen atoms
        # Create a mask for non-hydrogen atoms (atomic number != 1)
        data.non_H_mask = data.x != 1

        

        if not self.Hydrogens:
            # Filter out hydrogen atoms
            data.x = data.x[data.non_H_mask]
            data.y = data.y[data.non_H_mask]
            data.pos = data.pos[data.non_H_mask]
            # Create a boolean mask for the edge_index tensor based on the criteria using a different approach
            atoms = torch.arange(0,data.non_H_mask.shape[0])[data.non_H_mask]
            bool_mask_source = torch.isin(data.edge_index[0], atoms )
            bool_mask_target = torch.isin(data.edge_index[1], atoms )
            bool_mask_combined = bool_mask_source & bool_mask_target
            data.edge_index = data.edge_index[:, bool_mask_combined]

            # Create a mapping from old node indices to new node indices
            node_mapping = {old: new for new, old in enumerate(atoms.tolist())}
        
            # Update the filtered_edge_index using the node_mapping
            data.edge_index = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] for edge in data.edge_index.t()]).t()

            # Use the boolean mask to filter out the undesired edges
            # 
            data.edge_attr = data.edge_attr[bool_mask_combined, :]
            data.non_H_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        


        # data.x = torch.cat([F.one_hot(data.x,119), data.temperature.repeat(data.x.shape[0],1)], dim=-1)

        # batch = Batch.from_data_list([data])

        # data.edge_index, _, _, edge_attr = radius_graph_pbc(batch, 5.0, 32)

        # direct_norm = torch.norm(edge_attr, dim=-1).unsqueeze(-1)
        # data.edge_attr = torch.cat([edge_attr/direct_norm, direct_norm], dim=-1).detach()
        
        data.cart_dist = data.edge_attr[:,-1].unsqueeze(-1)
        data.cart_dir = data.edge_attr[:,:-1]
        # data.N = torch.diag(torch.linalg.norm(torch.linalg.inv(data.cell.transpose(-1,-2)).squeeze(0), dim=-1))
        if self.optimize_cell:
            data.cell_og = data.cell
            data.cell, rotation_matrix = optmize_lattice(data.cell.squeeze(0))
            data.cell = data.cell.unsqueeze(0)
            data.cart_dir = data.cart_dir @ rotation_matrix
            data.y = rotation_matrix.transpose(-1,-2) @ data.y @ rotation_matrix

        # cart_vector = data.cart_dir * data.cart_dist
        # # try:
        # inv_cell = torch.linalg.inv(data.cell)
        # # except Exception as e:
        # #     print("Error inverting cell matrix")
        # #     print(data.cell)
        # #     print(data.cell_og)
            
        # #     raise Exception
        # frac_coord = cart_vector@inv_cell
        # data.frac_dist = torch.norm(frac_coord, dim=-1).squeeze(0)
        # data.frac_dir = torch.nn.functional.normalize(frac_coord, dim=-1).squeeze(0)


        

        # theta = torch.atan2(data.edge_attr[:,1], data.edge_attr[:,0]).unsqueeze(-1)
        # phi = torch.acos(data.edge_attr[:,2]).unsqueeze(-1)
        # data.edge_attr = torch.cat([theta, phi], dim=-1)
        # M = data.cell
        # N = torch.diag(torch.linalg.norm(torch.linalg.inv(M.transpose(-1,-2)).squeeze(0), dim=-1))

        # data.y = N.transpose(-1,-2)@data.y@N
        # data.y = data.cell.transpose(-1,-2)@data.y@data.cell

        if self.augment:
            data = self.augmentation(data)
        ## Volume
        if self.volume:
            data.y = ((4.0 / 3.0) * np.pi * torch.sqrt(torch.det(data.y))).unsqueeze(-1)
        data.y = data.y[data.non_H_mask]
        
        # data.natoms = data.x.shape[0]
        # data.expander_edges = torch.ones(data.x.shape[0],data.x.shape[0]).nonzero().t().contiguous().t()

        # delattr(data, "pos")
        delattr(data, "natoms")
        delattr(data, "edge_attr")
        # delattr(data, "refcode")
        

        return data

if __name__ == "__main__":
    dataset = DatasetPBC(root="/scratch/g1alexs/PBC_DATASET_SINGLE_MOL/", file_names="./train_files_5A_ignore.csv", standarize_temp=False)

    # temp = torch.cat([data.temperature for data in tqdm(dataset)])
    # print(temp.mean())
    # print(temp.std())
    for data in dataset:

        print(data)
        break

    
