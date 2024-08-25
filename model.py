
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

class ERecogClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ERecogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(in_features=32640, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # 128 x 1379
        x = self.conv1(x)
        # 126 x 1377
        x = F.relu(x)
        x = F.pad(x, (0, 1, 0, 0), mode='constant', value=0)
        # 126 x 1378
        x = self.pool1(x)
        x = self.dropout1(x)

        # 63 x 689

        x = self.conv2(x)
        # 61 x 687
        x = F.relu(x)
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        # 62 x 688
        x = self.pool1(x)
        x = self.dropout1(x)

        # 31 x 344

        x = self.conv3(x)
        # 29 x 342
        x = F.relu(x)
        x = F.pad(x, (0, 1, 0, 0), mode='constant', value=0)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 15 x 171

        x = self.conv4(x)
        # 13 x 169
        x = F.relu(x)
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        # 14 x 170
        x = self.pool1(x)
        x = self.dropout1(x)

        # 7 x 85

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        output = F.log_softmax(x, dim=1)
        return output


