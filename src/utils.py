import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from custom_dataset import CreamTorchData

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def create_data_loader(batch_files, batch_size, shuffle=True):
    dataset = CreamTorchData(batch_files)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def plot_classification(train_loss, train_accuracy, val_loss, val_accuracy, label=''):
    number_of_epochs = len(train_loss)
    epochs = range(1, number_of_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plotting Training Loss
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, train_loss, label= label + ' Training Loss')
    if len(val_loss):
        plt.plot(epochs, val_loss, label= label + ' Validation Loss')

    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accuracy, label= label + ' Training Accuracy')
    if len(val_accuracy):
        plt.plot(epochs, val_accuracy, label= label + ' Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_classification(model, criterion, optimizer, number_of_epochs, train_loaders, val_loader):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        i = 0
        for train_loader in train_loaders:
            for inputs, labels in train_loader:
                i +=1
                inputs = inputs.transpose(0, 1)
                labels = labels.transpose(0, 1)
                labels = labels.squeeze()
                labels = labels.to(torch.long)
                optimizer.zero_grad()            
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted.squeeze() == labels).sum().item()
                total += labels.size(0)
                #print(f'Running loss: {running_loss}')

        epoch_train_loss = running_loss / total
        epoch_train_accuracy = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")

        torch.cuda.empty_cache()

        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                print(len(val_loader))
                for val_inputs, val_labels in val_loader:

                    val_inputs = val_inputs.transpose(0, 1)
                    val_labels = val_labels.transpose(0, 1)
                    val_labels = val_labels.squeeze()
                    val_labels = val_labels.to(torch.long)
                    
                    #with torch.cuda.amp.autocast():
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs.squeeze(), val_labels)
                    val_running_loss += val_loss.item()


                    val_predicted = torch.argmax(val_outputs, dim=1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted.squeeze() == val_labels).sum().item()

            epoch_val_loss = val_running_loss / val_total
            epoch_val_accuracy = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            print(f"Epoch [{epoch + 1}/{number_of_epochs}], Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")
            torch.cuda.empty_cache()
        

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_classification(model, criterion, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predicted_labels, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:

            inputs = inputs.transpose(0, 1)
            labels = labels.transpose(0, 1)
            labels = labels.squeeze()
            labels = labels.to(torch.long)

            outputs = model(inputs)
            total_loss += criterion(outputs.squeeze(), labels).item()
            
            predicted = torch.argmax(outputs, dim=1)
             
            predicted_labels.extend(predicted.squeeze().tolist())
            true_labels.extend(labels.tolist())

            total_samples += labels.size(0)
            total_correct += (predicted.squeeze() == labels).sum().item()

    # compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print(f'Model evaluation on: {loader}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # plot
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
