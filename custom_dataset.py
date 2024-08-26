import torch
from torch.utils.data import Dataset
import numpy as np
import os 

def load_batch(batch_file):
    with np.load(batch_file) as data:
        features = data['features']
        labels = data['labels']
    
    return features, labels

class CreamTorchData(Dataset):

    def __init__(self, batch_files_path):
        self.batch_files_path = batch_files_path
        # Get a list of all .npz files in the directory
        self.batch_files = [os.path.join(batch_files_path, file) 
                            for file in os.listdir(batch_files_path) 
                            if file.endswith('.npz')]
    
        self.emotion_map = {"angry" : 0, "disgust" : 1, "fear" : 2, "happy" : 3, "sad" : 4, "neutral" : 5}

    def __len__(self):
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        batch_file = self.batch_files[idx]
        with np.load(batch_file) as data:
            features = data['features']
            labels = data['labels']
            print(f'Loading batch: {features.shape}')
            mapped_labels = np.array([self.emotion_map[label] for label in labels])

        return torch.tensor(features,dtype=torch.float32), torch.tensor(mapped_labels,dtype=torch.int32)
    