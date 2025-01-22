import re
import queue
import argparse
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import os.path as osp
from ccdc import io
from gemmi import cif 
import pandas as pd
from utils import radius_graph_pbc


def frac_to_cart_matrix(abc, angles):
    a, b, c = abc
    alpha, beta, gamma = torch.tensor(angles) * (torch.pi / 180.0)  # Convert to radians
    volume = 1 - torch.cos(alpha)**2 - torch.cos(beta)**2 - torch.cos(gamma)**2 + 2 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma)
    volume = c * torch.sqrt(volume) * b * a
    M = torch.tensor([
        [a, b * torch.cos(gamma), c * torch.cos(beta)],
        [0, b * torch.sin(gamma), c * (torch.cos(alpha) - torch.cos(beta)*torch.cos(gamma)) / torch.sin(gamma)],
        [0, 0, volume / (a * b * torch.sin(gamma))]
    ])
    return M.t()


def delete_repeated(coord, threshold=1e-4):    
    coord = torch.where(coord < 0 , coord+1, coord)
    coord = torch.where(coord > 1 , coord-1, coord)
    coord = torch.where(torch.isclose(coord, torch.ones(coord.shape, dtype=torch.float32), atol=1e-4), torch.zeros_like(coord, dtype=torch.float32), coord)
    distance = coord.unsqueeze(0) - coord.unsqueeze(1)
    distance = torch.linalg.norm(distance, dim=-1)
    
    duplicates = distance < threshold
    first_true_indices = torch.argmax(duplicates.float(), dim=1)

    mask_to_keep = torch.arange(coord.shape[0]) <= first_true_indices

    return mask_to_keep

possible_atomic_num_list = list(range(1, 119))

def refcsd2graph(refcode, output_folder):
    try:
        csd_reader = io.EntryReader("CSD")
        entry = csd_reader.entry(refcode)

        if entry.pressure is not None:
            return None
        
        if entry.remarks is not None:
            return None
        
        if entry.crystal.has_disorder:
            return None
        
        if entry.temperature is None:
            doc = cif.read_string(entry.to_string(format='cif'))


            try: # copy all the data from mmCIF file
                block = doc.sole_block()  # mmCIF has exactly one block
                temperature = block.find_pair("_diffrn_ambient_temperature")[1]
                temperature = re.findall(r'\d+\.?\d*',string=str(temperature))
                assert(len(temperature)==1)
                assert(temperature[0] is not None)
                temperature = float(temperature[0])
            except Exception as e:
                return None
        else:
            temperature = entry.temperature
            
            temp = re.findall(r'\d+\.?\d*',string=str(entry.temperature))
            try:
                assert(len(temp)==1)
            except:
                return None
            
            temperature = float(temp[0])
        
        data = Data()
        entry = csd_reader.entry(refcode)
        packing = entry.crystal.packing(inclusion="OnlyAtomsIncluded")
        keep_mask = delete_repeated(torch.tensor([[atom.fractional_coordinates.x, atom.fractional_coordinates.y, atom.fractional_coordinates.z] for atom in packing.atoms]))
        adp = []
        data.x = torch.tensor([possible_atomic_num_list.index(atom.atomic_number)+1 for atom in packing.atoms])[keep_mask]
        data.pos = torch.tensor([[atom.coordinates.x, atom.coordinates.y, atom.coordinates.z] for atom in packing.atoms])[keep_mask]

        for atom in packing.atoms:
            if atom.displacement_parameters is None:
                if atom.atomic_number == 1:
                    adp.append(torch.eye(3).unsqueeze(0)*0.01)
                    continue
                elif atom.atomic_number != 1:
                    print("istrotropic")
                    return
            
            if atom.displacement_parameters.type == "Isotropic" and atom.atomic_number == 1:
                adp.append(torch.eye(3).unsqueeze(0)*atom.displacement_parameters.isotropic_equivalent)
                
            elif atom.displacement_parameters.type == "Anisotropic":
                adp.append(torch.tensor(atom.displacement_parameters.values).unsqueeze(0))

            else:
                raise NotImplementedError
        
        data.y = torch.cat(adp, dim=0)[keep_mask]
        
        data.y = data.y[data.x != 1]

        abc = entry.crystal.cell_lengths.a, entry.crystal.cell_lengths.b, entry.crystal.cell_lengths.c
        angles = entry.crystal.cell_angles.alpha, entry.crystal.cell_angles.beta, entry.crystal.cell_angles.gamma

        M = frac_to_cart_matrix(abc, angles)

        N = torch.diag(torch.linalg.norm(torch.linalg.inv(M.transpose(-1,-2)).squeeze(0), dim=-1))

        data.cell = M.unsqueeze(0)
       

        data.y = N.transpose(-1,-2)@data.y@N
        data.y = data.cell.transpose(-1,-2)@data.y@data.cell

        assert torch.allclose(torch.tensor(entry.crystal.orthogonal_to_fractional.translation), torch.zeros(3)), refcode

        
        data.pbc = torch.tensor([[True, True, True]])
        data.natoms = torch.tensor([data.x.shape[0]])
        data.temperature = torch.tensor([temperature])
        data.refcode = refcode

        batch = Batch.from_data_list([data])

        edge_index, _, _, edge_attr = radius_graph_pbc(batch, 5.0, None)

        data.edge_index = edge_index
        direct_norm = torch.norm(edge_attr, dim=-1).unsqueeze(-1)
        data.edge_attr = torch.cat([edge_attr/direct_norm, direct_norm], dim=-1)
        
        

        torch.save(data, osp.join(output_folder,str(refcode)+".pt"))
    except Exception as e:
        
        raise Exception(f"Error occurred for refcode: {refcode} Error message: {str(e)}")


def worker_process(task_queue, results_queue, counter, error_event):
    while True:
        if error_event.is_set():  # Check if the error event is set at the start of each iteration
            break
        try:
            refcode, output_folder = task_queue.get_nowait()
            
            error_occurred = mp.Value('i', 0)  # 0 means no error, 1 means error occurred
            
            def target(error_flag):
                try:
                    refcsd2graph(refcode, output_folder)
                except Exception as e:
                    error_flag.value = 1
                    results_queue.put(str(e))

            process = mp.Process(target=target, args=(error_occurred,))
            process.start()
            process.join(timeout=600)  # 10 minutes timeout
            
            if error_occurred.value == 1:  # If an error occurred in the target function
                error_event.set()
                break

            if process.is_alive():
                process.terminate()
                process.join()
                results_queue.put(f"Timeout occurred for refcode: {refcode}!")

            # Increment the shared counter
            with counter.get_lock():
                counter.value += 1

        except queue.Empty:  
            break


if __name__ == '__main__':
    torch.set_num_threads(10)
    parser = argparse.ArgumentParser(description='Process CSD data into graphs')
    parser.add_argument('--output', type=str, default="ADP_DATASET/data/", help='Output folder path')
    args = parser.parse_args()
    output_folder = args.output
    data_df = pd.read_csv('./csv/all_dataset.csv', header=None)
    
    res = [refcsd2graph(refcode, output_folder) for refcode in tqdm(data_df[0].tolist())]
    res = [r for r in res if r is None]
    with open("errors.txt", "w") as f:
        f.write("\n".join(res))