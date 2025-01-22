import torch
import json
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='Process some metrics.')
parser.add_argument('--path', type=str, default="/home/g1alexs/Sublimation/results/CartNet", help='Path to the results directory')
args = parser.parse_args()

path = args.path
print(path)

MAE = []
similarity_index = []
iou = []

for i in range(4):
    try:
        # Open and load the JSON file
        with open(osp.join(path,str(i),"test/stats.json"), 'r') as file:
            data = json.load(file)
            MAE.append(data['MAE'])
            similarity_index.append(data['similarity_index'])
            iou.append(data['iou'])
    except Exception as e:
        print(e)

print(similarity_index)
print(iou)
print(MAE)


print("MAE")
MAE = torch.tensor(MAE)
print(MAE.mean().item(), MAE.std().item(), MAE.max().item(), MAE.min().item())

print("similarity_index")
similarity_index = torch.tensor(similarity_index)
print(similarity_index.mean().item(), similarity_index.std().item(), similarity_index.max().item(), similarity_index.min().item())

print("iou")
iou = torch.tensor(iou)
print(iou.mean().item(), iou.std().item(), iou.max().item(), iou.min().item())

