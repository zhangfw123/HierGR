import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="waimai", help='dataset')
    parser.add_argument("--data_path", type=str, default="", help="Input data path.")
    parser.add_argument("--data_name", type=str, default="semantic_embs_", help="Input data path.")
    parser.add_argument("--data_split", type=int, default=6, help="Input data path.")
    parser.add_argument("--output_dir", type=str,default="../checkpoint/", help='output_dir')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='best_collision_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
ckpt_path = args_setting.checkpoint


device = torch.device("cuda")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]
print(args)

model = RQVAE(in_dim=256,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_key = k[7:]
    else:
        new_key = k
    new_state_dict[new_key] = v
print(new_state_dict.keys())
model.load_state_dict(new_state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)


all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

data_path = args_setting.data_path
data_name = args_setting.data_name

split_idx=args_setting.data_split
print(data_path+'/'+data_name+f'{split_idx}.npy')
dataset = EmbDataset(data_path+'/'+data_name+f'{split_idx}.npy')
data_loader = DataLoader(
    dataset,
    batch_size=4096,
    num_workers=args.num_workers,
    pin_memory=True,
    shuffle=False  
)
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)



print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

output_dir = args_setting.output_dir
output_file = f"{args_setting.dataset}.{split_idx}.index.json"
output_file = os.path.join(output_dir,output_file)
print(f"save to file:{output_file}")

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
