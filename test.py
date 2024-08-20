import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import random
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import preprocessing_CREMA as prep


female_ids = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,
              1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,1052,1053,1054,
              1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,
              1082,1084,1089,1091]
male_ids = list(set(list(range(1001,1092))) - set(female_ids))

creamData = prep.CreamData(
    path = 'data/CREAM-D_wav/AudioWAV/',
    female = female_ids,
    male = male_ids,
    path_to_standardize_audio_data='ProcessedData',
)

data = pd.read_csv('extracted_features.csv')
loaded_matrices = np.load('extracted_features_matrices.npy')
print(loaded_matrices.shape)

data.drop('X', axis=1, inplace=True)
data_map = {path : matrix for path in data['path'] for matrix in loaded_matrices}
print(len(data_map))

y_train, y_validation, y_test = creamData.train_test_split(data)
print(len(y_train), len(y_validation), len(y_test))

def get_data_loader(X_data, y_data, batch_size, shuffle):
  y_data = pd.Series(y_data)
  X_tensor = torch.FloatTensor(X_data)
  if isinstance(y_data, pd.Series):
    y_data = y_data.values
  y_tensor = torch.LongTensor(y_data)
  dataset = TensorDataset(X_tensor, y_tensor)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

X_train = []
X_validation = []
X_test = []

for path in y_train['path']:
   X_train.append(data_map[path])

for path in y_validation['path']:
   X_validation.append(data_map[path])

for path in y_test['path']:
   X_test.append(data_map[path])

X_train = np.array(X_train)
X_validation = np.array(X_validation)
X_test = np.array(X_test)

y_train = pd.Series(y_train['emotion'])
y_validation = pd.Series(y_validation['emotion'])
y_test = pd.Series(y_test['emotion'])

print(y_train.value_counts())
print(len(y_train), len(y_validation), len(y_test))

emotion_map = {"angry" : 0, "disgust" : 1, "fear" : 2, "happy" : 3, "sad" : 4, "neutral" : 5}

y_train = y_train.apply(lambda x : emotion_map[x])
y_validation = y_validation.apply(lambda x : emotion_map[x])
y_test = y_test.apply(lambda x : emotion_map[x])

class ERecogClassifier(nn.Module):
  def __init__(self, num_classes):
    super(ERecogClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
    self.fc1 = nn.Linear(in_features=4*126*1377, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)

    return output
    
num_classes = data['emotion'].nunique()
model = ERecogClassifier(num_classes=num_classes)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

X_train = X_train.reshape(len(X_train), 1, 128, 1379)
train_loader = get_data_loader(X_train, y_train, batch_size=6, shuffle=False)

def train_classification(model, criterion, optimizer, number_of_epochs, train_loader):
  losses = []
  accuracies = []

  for epoch in range(number_of_epochs):
      running_loss = 0.0
      correct = 0
      total = 0

      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs.squeeze(), labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          
          predicted = torch.argmax(outputs, dim=1)
            
          correct += (predicted.squeeze() == labels).sum().item()
          total += labels.size(0)

      epoch_loss = running_loss / len(train_loader)
      epoch_accuracy = correct / total
      losses.append(epoch_loss)
      accuracies.append(epoch_accuracy)
      print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

  return losses, accuracies

train_losses, train_accuaracies = train_classification(model, criterion, optimizer, 10, train_loader)