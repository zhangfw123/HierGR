import logging
import json
import numpy as np
import torch
import torch.distributed as dist
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import EmbDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class Trainer(object):
    def __init__(self, args, model, rank, world_size):
        self.args = args
        self.model = model
        self.rank = int(os.environ["RANK"])
        self.world_size = world_size
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}
        
        self.sampler = None

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.module.vq_initialization(data)    

    def _train_epoch(self, train_data, epoch_idx):
        if isinstance(train_data.sampler, DistributedSampler):
            train_data.sampler.set_epoch(epoch_idx)

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out = self.model(data)
            loss, cf_loss, loss_recon, quant_loss = self.model.module.compute_loss(out, rq_loss, emb_idx, dense_out, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += 0
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_cf_loss, quant_loss.item()
    
    @torch.no_grad()
    def _valid_epoch(self, valid_data,epoch_idx):

        self.model.eval()
        if isinstance(valid_data.sampler, DistributedSampler):
            valid_data.sampler.set_epoch(epoch_idx)
        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        
        indices_list = []
        num_sample = 0
        
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.module.get_indices(data)
            
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            indices_list.append(indices)
        all_indices = [None] * self.world_size
        dist.all_gather_object(all_indices, indices_list)
        all_num_sample = [None] * self.world_size
        dist.all_gather_object(all_num_sample, num_sample)
        if self.rank == 0:
            
            indices_set = set()
            total_indices = []
            for indices in all_indices:
                total_indices.extend(np.concatenate(indices, axis=0).tolist())
            print("total_indices:", len(total_indices))
            for index in tqdm(total_indices):
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
            collision_rate = (len(total_indices) - len(indices_set))/len(total_indices)
        else:
            collision_rate = 1
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cf_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data_path, data_name, data_split, args):

        cur_eval_step = 0
        # self.vq_init()
        for epoch_idx in range(self.epochs):
            total_train_loss = 0
            total_train_recon_loss = 0
            total_quant_loss = 0
            training_start_time = time()
            print(dist.get_rank(), dist.get_world_size())
            for split_idx in range(1, data_split+1):
                dataset = EmbDataset(data_path+'/'+data_name+f'{split_idx}.npy')
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True
                )

                batch_size = args.batch_size // dist.get_world_size()
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    shuffle=True  
                )
                # train
                train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data_loader, epoch_idx)
                total_train_loss += train_loss
                total_train_recon_loss += train_recon_loss
                total_quant_loss += quant_loss

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, total_train_loss, total_train_recon_loss, cf_loss
            )
            self.logger.info(train_loss_output)

            if total_train_loss < self.best_loss:
                self.best_loss = total_train_loss

            # eval
            
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                total_collision_rate = 0
                for split_idx in range(1, data_split+1):
                    dataset = EmbDataset(data_path+'/'+data_name+f'{split_idx}.npy')
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=True
                    )
                    batch_size = args.batch_size // dist.get_world_size()
                    data_loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        shuffle=False  
                    )
                    collision_rate = self._valid_epoch(data_loader, epoch_idx)
                    print("collision_rate:", collision_rate)
                    total_collision_rate += collision_rate
                total_collision_rate /= data_split
                print(total_collision_rate)
                if self.rank==0:
                    if total_collision_rate < self.best_collision_rate:
                        self.best_collision_rate = total_collision_rate
                        cur_eval_step = 0
                        self._save_checkpoint(epoch_idx, collision_rate=total_collision_rate,
                                              ckpt_file=self.best_collision_ckpt)
                    else:
                        cur_eval_step += 1


                    valid_end_time = time()
                    valid_score_output = (
                        set_color("epoch %d evaluating", "green")
                        + " ["
                        + set_color("time", "blue")
                        + ": %.2fs, "
                        + set_color("collision_rate", "blue")
                        + ": %f]"
                    ) % (epoch_idx, valid_end_time - valid_start_time, total_collision_rate)

                    self.logger.info(valid_score_output)

                    if epoch_idx>1:
                        self._save_checkpoint(epoch_idx, collision_rate=total_collision_rate)


        return self.best_loss, self.best_collision_rate




