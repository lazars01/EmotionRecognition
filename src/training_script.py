import utils as utils
import pickle
from model import ERecogClassifier

import os

import numpy as np

import torch.nn as nn
import torch.optim as optim

num_classes = 6
model = ERecogClassifier(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

train_loader = utils.create_data_loader('../batches/train', batch_size=1, shuffle=True)
val_loader = utils.create_data_loader('../batches/validation', batch_size=1, shuffle=False)

train_loader_augment = utils.create_data_loader('../batches_augment/train', batch_size=1, shuffle=True)
val_loader_augment = utils.create_data_loader('../batches_augment/validation', batch_size=1, shuffle=False)

train_losses, train_accuracies, val_losses, val_accuracies = utils.train_classification(model, criterion, optimizer, 50, [train_loader], val_loader)

os.makedirs('models', exist_ok=True)

with open('models/noaugment.model.pickle','wb') as model_file:
     pickle.dump(model, model_file)

os.makedirs('results', exist_ok=True)

np.save('results/train_loss', train_losses)
np.save('results/train_accuracies', train_accuracies)
np.save('results/val_loss', val_losses)
np.save('results/val_acc', val_accuracies)


model = ERecogClassifier(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
train_losses, train_accuracies, val_losses, val_accuracies = utils.train_classification(model, criterion, optimizer, 50, [train_loader_augment], val_loader_augment)



with open('models/augment.model.pickle','wb') as model_file:
     pickle.dump(model, model_file)


np.save('results/augment_train_loss', train_losses)
np.save('results/augment_train_accuracies', train_accuracies)
np.save('results/augment_val_loss', val_losses)
np.save('results/augment_val_acc', val_accuracies)