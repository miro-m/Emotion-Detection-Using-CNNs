#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:49:29 2024

@author: miro
"""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import numpy as np

from modeltest import MainModel, VariantModel1, VariantModel2

def load_model(model_path, model_type):
    if model_type == 'main':
        model = MainModel()
    elif model_type == 'variant1':
        model = VariantModel1()
    elif model_type == 'variant2':
        model = VariantModel2()
    else:
        raise ValueError('Invalid model type provided. Choose from "main", "variant1", "variant2".')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_dataset(model, dataset_loader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataset_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_preds)

def predict_single_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation and Application')
    parser.add_argument('--mode', type=str, choices=['dataset', 'single'], required=True,
                        help='Operational mode: "dataset" for evaluating on a complete dataset, "single" for a single image prediction.')
    parser.add_argument('--model_type', type=str, choices=['main', 'variant1', 'variant2'], required=True,
                        help='Type of model to load for prediction.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file.')
    parser.add_argument('--data_path', type=str, help='Path to the dataset or image, depending on the mode.')
    args = parser.parse_args()

    # Transformations must match those used during training
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_model(args.model_path, args.model_type)
## results: ['focused/engaged', 'happy', 'neutral', 'surprised']
    if args.mode == 'dataset':
        dataset = ImageFolder(root=args.data_path, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        predictions = evaluate_dataset(model, loader)
        print("Predictions for the dataset:", predictions)
    elif args.mode == 'single':
        prediction = predict_single_image(model, args.data_path, transform)
        print("Prediction for the single image:", prediction)

if __name__ == '__main__':
    main()
