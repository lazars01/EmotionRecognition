import os
import pandas as pd
import numpy as np

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import ERecogClassifier
import preprocessing_CREMA as prep
from custom_dataset import CreamTorchData

# Ids for  dataset
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


def create_data_loader(batch_files, batch_size, shuffle= True):
    dataset = CreamTorchData(batch_files)
    return DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,num_workers=2)


emotion_map = {"angry" : 0, "disgust" : 1, "fear" : 2, "happy" : 3, "sad" : 4, "neutral" : 5}


    
num_classes = len(emotion_map)
model = ERecogClassifier(num_classes=num_classes)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

train_loader = create_data_loader('batches/train', batch_size=32, shuffle=True)
val_loader = create_data_loader('batches/validation', batch_size=32, shuffle=False)
test_loader = create_data_loader('batches/test', batch_size=32, shuffle=False)

def train_classification(model, criterion, optimizer, number_of_epochs, train_loader, val_loader, scaler):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Use mixed precision
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            print(f"Running Loss : {running_loss}")

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_accuracy = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")

        #torch.cuda.empty_cache() mozda

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:

                with torch.cuda.amp.autocast():
                  val_outputs = model(val_inputs)
                  val_loss = criterion(val_outputs.squeeze(), val_labels)
                val_running_loss += val_loss.item()


                val_predicted = torch.argmax(val_outputs, dim=1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted.squeeze() == val_labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")


        # Clear cache to free memory
    torch.cuda.empty_cache()

    return train_losses, train_accuracies, val_losses, val_accuracies

train_losses, train_accuaracies = train_classification(model, criterion, optimizer, 10, train_loader)