
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

class ERecogClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ERecogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        
        # Pooling and Dropout
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=61440, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # First convolution layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second convolution layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Third convolution layer
        x = self.conv3(x)
        x = F.relu(x)
        x = F.pad(x, (1, 0, 0, 1), mode='constant', value=0)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Fourth convolution layer
        x = self.conv4(x)
        x = F.relu(x)
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Fifth convolution layer
        x = self.conv5(x)
        x = F.relu(x)
        x = F.pad(x, (1, 0, 0, 1), mode='constant', value=0)
        #x = self.pool1(x)
        x = self.dropout1(x)

        #print(x.shape)
        # Sixth convolution layer
        x = self.conv6(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)  # Flatten the tensor for fully connected layers
        #print(x.shape)
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        
        output = F.log_softmax(x, dim= 1)
        return output