import torch
from torch.utils.data import Dataset
import numpy as np

def load_batch(batch_file):
    with np.load(batch_file) as data:
        features = data['features']
        labels = data['labels']
    
    return features, labels

class CreamTorchData(Dataset):

    def __init__(self, batch_files):

        self.batch_files = batch_files

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        with np.load(batch_file) as data:
            features = data['features']
            labels = data['labels']

        return torch.tensor(features,dtype=torch.float32), torch.tensor(labels,dtype=torch.long)
    