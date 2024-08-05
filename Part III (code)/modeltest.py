#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:15:03 2024

@author: miro
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Model Definitions
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Added 3rd convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Added 4th convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Increased dropout to prevent overfitting
        # The input size for fc1 needs to be adjusted. Placeholder for now.
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Adjusted based on new architecture and image size
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)  # Adjusted to match the new size
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class VariantModel1(nn.Module):
    def __init__(self):
        super(VariantModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Additional convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  #Additional pooling layer for the new conv layer
        x = x.view(-1, 128 * 6 * 6)  
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VariantModel2(nn.Module):
    def __init__(self):
        super(VariantModel2, self).__init__()
        # Using a larger kernel size for the first layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        # Using a smaller kernel size for the second layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_dataset_by_gender():
    dataset_path = './new_dataset'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # Debugging: Print out class names and associated indices
    # print("Classes found:", dataset.classes)
    # for idx, (path, class_index) in enumerate(dataset.samples):
    #     print(f"Index: {idx}, Path: {path}, Class Index: {class_index}, Class Name: {dataset.classes[class_index]}")

    male_indices = [i for i, (path, label) in enumerate(dataset.samples) if 'male' in path.lower()]
    female_indices = [i for i, (path, label) in enumerate(dataset.samples) if 'female' in path.lower()]

    loaders = {
        'male': DataLoader(Subset(dataset, male_indices), batch_size=32, shuffle=True),
        'female': DataLoader(Subset(dataset, female_indices), batch_size=32, shuffle=True)
    }
    return loaders

def train_model(model, train_loader, val_loader, device, criterion, optimizer, modelName, n_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{modelName}.pth')

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    plt.show()
    
def evaluate_model_by_group(model, gender_loader, device, modelName, class_names):
    model_path = f'best_model_{modelName}.pth'
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    results = {}
    criterion = nn.CrossEntropyLoss()

    for group, loader in gender_loader.items():
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        results[group] = {
            'Loss': total_loss / len(loader),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }

        # Plot confusion matrix for the group
        print(f"Confusion Matrix for {group}:")
        plot_confusion_matrix(all_labels, all_preds, class_names)

    for group, metrics in results.items():
        print(f"Results for {group}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

def evaluate_model(model, test_loader, device, class_names, modelName):
    model_path = f'best_model_{modelName}.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #Calculate macro and micro evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    #Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)

    print(f'Accuracy: {accuracy}\nMacro Precision: {macro_precision}, Macro Recall: {macro_recall}, Macro F1: {macro_f1}\nMicro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}')



# Main Execution
def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #DATA PREP
    train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resizing the image to 128x128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resizing the image to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset_path = './new_dataset_level_3'
    #Apply the training transform for the training dataset
    train_dataset = ImageFolder(root=dataset_path, transform=train_transform)
    #Apply the test transform for validation and test datasets
    val_dataset = ImageFolder(root=dataset_path, transform=test_transform)
    test_dataset = ImageFolder(root=dataset_path, transform=test_transform)

    targets = np.array([sample[1] for sample in train_dataset.samples])

    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.15,
        random_state=42,
        stratify=targets)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.176,
        random_state=42,
        stratify=targets[train_val_idx])

    train_dataset = Subset(train_dataset, indices=train_idx)
    val_dataset = Subset(val_dataset, indices=val_idx)
    test_dataset = Subset(test_dataset, indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    #DATA PREP END
    class_names = ['focused/engaged', 'happy', 'neutral', 'surprised']

    ##!!!IMPORTANT!!!

    #The training for each model is commented, you can just run the program and it will evaluate the models since they will
    #already be included in the file. You can remove the comment and train the models again.

    main_model = MainModel().to(device)
    optimizer = optim.Adam(main_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # train_model(main_model, train_loader, val_loader, device, criterion, optimizer, "main", n_epochs=10)
    # evaluate_model(main_model, test_loader, device, class_names, "main")

    # main model bias and mitigation
    gender_loader = load_dataset_by_gender()
    evaluate_model_by_group(main_model, gender_loader, device, 'main', class_names)

    #VariantModel1 and VariantModel2
    # variant_model1 = VariantModel1().to(device)
    # optimizer_v1 = optim.Adam(variant_model1.parameters(), lr=0.001)
    # train_model(variant_model1, train_loader, val_loader, device, criterion, optimizer_v1, "variant1", n_epochs=10)
    # evaluate_model(variant_model1, test_loader, device, class_names, "variant1")

    # variant_model2 = VariantModel2().to(device)
    # optimizer_v2 = optim.Adam(variant_model2.parameters(), lr=0.001)
    # train_model(variant_model2, train_loader, val_loader, device, criterion, optimizer_v2, "variant2", n_epochs=10)
    # evaluate_model(variant_model2, test_loader, device, class_names, "variant2")

if __name__ == '__main__':
    main()