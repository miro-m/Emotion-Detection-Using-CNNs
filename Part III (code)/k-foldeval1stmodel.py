#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:01:00 2024

@author: miro
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = '/Users/miro/Documents/GitHub/COMP472PROJECT-AK_4/dataset'
    dataset = ImageFolder(root=dataset_path, transform=transform)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold+1}/10")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

        model = MainModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
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

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        micro_recall = recall_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')

        fold_results.append({
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'micro_precision': micro_precision,
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        })

        print(f"Results for fold {fold+1}: Accuracy: {accuracy}, Macro Precision: {macro_precision}, Micro Precision: {micro_precision}, Macro Recall: {macro_recall}, Micro Recall: {micro_recall}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

    avg_accuracy = np.mean([f['accuracy'] for f in fold_results])
    avg_macro_precision = np.mean([f['macro_precision'] for f in fold_results])
    avg_micro_precision = np.mean([f['micro_precision'] for f in fold_results])
    avg_macro_recall = np.mean([f['macro_recall'] for f in fold_results])
    avg_micro_recall = np.mean([f['micro_recall'] for f in fold_results])
    avg_macro_f1 = np.mean([f['macro_f1'] for f in fold_results])
    avg_micro_f1 = np.mean([f['micro_f1'] for f in fold_results])

    print(f"Average scores across all folds: Accuracy: {avg_accuracy}, Macro Precision: {avg_macro_precision}, Micro Precision: {avg_micro_precision}, Macro Recall: {avg_macro_recall}, Micro Recall: {avg_micro_recall}, Macro F1: {avg_macro_f1}, Micro F1: {avg_micro_f1}")

if __name__ == '__main__':
    main()
