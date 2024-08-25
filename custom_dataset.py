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
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for batch_file in self.batch_files:
            features, labels = load_batch(batch_file)
            self.data.append(features)
            self.labels.append(labels)
        
        self.data = np.concatenate(self.data, axis = 0)
        self.data = np.concatenate(self.labels , axis = 0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index],dtype=torch.float32), torch.tensor(self.labels[index],dtype=torch.long)
    