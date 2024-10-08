import torch
from torch.utils.data import Dataset
import numpy as np
import os 

# Custom dataset to enable loading batch by batch using DataLoader, so we don't need to load whole dataset into RAM

class CreamTorchData(Dataset):

    def __init__(self, batch_files_paths):
        self.batch_files_path = batch_files_paths
        self.batch_files = [os.path.join(files_path, file) 
                            for files_path in batch_files_paths
                            for file in os.listdir(files_path) 
                            if file.endswith('.npz')]
    
        self.emotion_map = {"angry" : 0, "disgust" : 1, "fear" : 2, "happy" : 3, "sad" : 4, "neutral" : 5}

    def __len__(self):
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        with np.load(batch_file) as data:
            features = data['features']
            labels = data['labels']
            mapped_labels = np.array([self.emotion_map[label] for label in labels])
            

        return torch.tensor(features,dtype=torch.float32), torch.tensor(mapped_labels,dtype=torch.int32)
    