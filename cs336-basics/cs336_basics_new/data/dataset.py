import torch
from torch.utils.data import Dataset
import numpy as np

class LMDataset(Dataset):
    def __init__(self, data_path: str, context_length: int):
        self.dataset = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
    def __len__(self):
        return len(self.dataset) // self.context_length
    def __getitem__(self, idx):
        start = idx * self.context_length
        x = torch.from_numpy(self.dataset[start:start+self.context_length].astype(np.int64))
        y = torch.from_numpy(self.dataset[start+1:start+self.context_length+1].astype(np.int64))
        return (x, y)

