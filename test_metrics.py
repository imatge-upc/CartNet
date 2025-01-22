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


for i in range(4):
    try:
        # Open and load the JSON file
        with open(osp.join(path,str(i),"test/stats.json"), 'r') as file:
            data = json.load(file)
            MAE.append(data['MAE'])
    except Exception as e:
        print(e)


print("MAE")
MAE = torch.tensor(MAE)
print(MAE.mean().item(), MAE.std().item(), MAE.max().item(), MAE.min().item())
