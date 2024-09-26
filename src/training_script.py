from . import utils as utils
import pickle
from .model import ERecogClassifier

import os

import numpy as np

import torch.nn as nn
import torch.optim as optim


train_loader = utils.create_data_loader(['batches/train'], batch_size=1, shuffle=True)
train_augment_loader = utils.create_data_loader(['batches/train_augment', 'batches/train'], batch_size=1, shuffle=True)
train_spec_augment_loader = utils.create_data_loader(['batches/train_spec_augment','batches/train'], batch_size = 1, shuffle=True)


loaders = [ train_loader,train_augment_loader, train_spec_augment_loader]
paths = ['train','train_augment','train_spec_augment']


val_loader = utils.create_data_loader(['batches/validation'], batch_size=1, shuffle=False)
num_classes = 6

for i, path in enumerate(paths):
     
     model = ERecogClassifier(num_classes=num_classes)
     optimizer = optim.Adam(model.parameters(), lr=5e-4)
     criterion = nn.NLLLoss()

     loader = loaders[i]
     train_losses, train_accuracies, val_losses, val_accuracies = utils.train_classification(model, criterion, optimizer, 40, [loader], val_loader)

     os.makedirs('models',exist_ok=True)
     with open(f'models/{path}.model.pickle', 'wb') as model_file:
          pickle.dump(model, model_file)

     os.makedirs('results', exist_ok=True)
     np.save(f'results/{path}_loss', train_losses)
     np.save(f'results/{path}_accuracies', train_accuracies)
     np.save(f'results/{path}_val_loss', val_losses)
     np.save(f'results/{path}_val_acc', val_accuracies)

