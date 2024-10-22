from torch_geometric.data import InMemoryDataset
from tqdm.auto import tqdm
from jarvis.core.specie import get_node_attributes
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, Batch
from dataset.utils import radius_graph_pbc, optmize_lattice
import torch
import roma


class Figshare_Dataset(InMemoryDataset):
    def __init__(self, root, data, targets, transform=None, pre_transform=None, name="jarvis", radius=5.0, max_neigh=-1, augment=False):
        
        self.data = data
        self.targets = targets
        self.name = name
        self.radius = radius
        self.max_neigh = max_neigh if max_neigh > 0 else None
        self.augment = augment
        super(Figshare_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.name + ".pt"

    def download(self):
        pass
    

    def get(self, idx):
        data = super().get(idx)
        
        if self.augment:
            data = self.augment_data(data)
        
        return data

    def augment_data(self, data):
        R = roma.utils.random_rotmat(size=1, device=data.x.device).squeeze(0)    
        data.cart_dir = data.cart_dir @ R
        data.cell = data.cell @ R

        return data

    def process(self):
        data_list = []
        for i, (ddat, target) in tqdm(enumerate(zip(self.data, self.targets)),total=len(self.data)):
            structure = Atoms.from_dict(ddat["atoms"])
            atomic_numbers = torch.tensor([get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]).squeeze(-1)
            target = torch.tensor(target)
            data = Data(x=atomic_numbers, y=target)
            data.pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([data.x.shape[0]])
            data.cell = data.cell.unsqueeze(0)

            #Compute PBC
            batch = Batch.from_data_list([data])
            edge_index, _, _, cart_vector = radius_graph_pbc(batch, self.radius, self.max_neigh)
            
            data.cart_dist = torch.norm(cart_vector, p=2, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(cart_vector, p=2, dim=-1)
            

            data.edge_index = edge_index
            delattr(data, "pbc")
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])