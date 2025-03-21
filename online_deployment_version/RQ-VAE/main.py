import argparse
import random
import torch
import numpy as np
from time import time
import logging
# import wandb
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="", help="Input data path.")
    parser.add_argument("--data_name", type=str, default="semantic_embs_", help="Input data path.")
    parser.add_argument("--data_split", type=int, default=6, help="Input data path.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--layers', type=int, nargs='+', default=[4096,2048,1024,512,256,128], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")
    parser.add_argument('--job_name', type=str, default="worker", required=False, help='')
    parser.add_argument('--task_index', type=int, default=0, required=False, help='')
    parser.add_argument('--worker_hosts', type=str, default="", required=False, help='')
    parser.add_argument('--local_rank', type=int, default=0, 
                      help='number of gpus to use')
    parser.add_argument('--world_size', type=int, default=0, 
                      help='number of gpus to use')
    parser.add_argument('--dist_url', type=str, default=0,
                      help='local rank for distributed training')
    parser.add_argument("--gpu_count",type=int,default=1,help="")

    return parser.parse_args()

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
    
def main_worker(rank, world_size, args):
    print(rank, world_size)
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=world_size, rank=rank)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    local_rank = int(os.environ["LOCAL_RANK"])
    model = RQVAE(
      in_dim=256,
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
      beta = args.beta,
      alpha = args.alpha,
      n_clusters= args.n_clusters,
      sample_strategy =args.sample_strategy
    )
    
    model = DDP(model.to(local_rank), device_ids=[local_rank])
    print(model)
    trainer = Trainer(args, model, local_rank, world_size)
    
    try:
        best_loss, best_collision_rate = trainer.fit(args.data_path, args.data_name, args.data_split, args)
        if rank == 0:
            print("Best Loss", best_loss)
            print("Best Collision Rate", best_collision_rate)
    finally:
        dist.destroy_process_group()
    

if __name__ == '__main__':
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main_worker(rank, world_size, args)



