# 保持原有实现不变
import numpy as np
import torch
import torch.utils.data as data

class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.embeddings = np.load(data_path)
        print(self.embeddings.shape)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        return torch.FloatTensor(emb), index

    def __len__(self):
        return len(self.embeddings)
