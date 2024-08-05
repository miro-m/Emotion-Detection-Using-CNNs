#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
import os

"""
Code explanation:
    This code provides the bar chart for class distribution, 25 images diplay for each class, and the pixel intensity distribution for each class.
"""


# Bar chart for the number of images for each class
folders = ['/Users/miro/Desktop/472 datastets/Emotion recognition 3/happy_images', '/Users/miro/Desktop/472 datastets/Emotion recognition 3/neutral_images', '/Users/miro/Desktop/472 datastets/Emotion recognition 3/surprised_images', '/Users/miro/Desktop/472 datastets/Emotion recognition 3/focused:engaged_images']
classes = ['Happy', 'Neutral', 'Surprised', 'Engaged/Focused']
class_counts = [1125, 1124, 1030, 500]

plt.figure(figsize=(10, 6))
plt.bar(classes, class_counts, color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Emotion Categories')
plt.ylabel('Number of Images')
plt.show()

# Display Sample Images and Their Pixel Intensity Distribution
def display_samples_and_intensity(folder, class_name):
    # Randomly shuffle the list of image files in the given folder and select the first 25
    image_files = shuffle(os.listdir(folder))[:25]
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(f'Sample Images for {class_name}')
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path)
        # Display the image in the ith subplot (arranged in a 5x5 grid)
        axs[i // 5, i % 5].imshow(img)
        axs[i // 5, i % 5].axis('off')
    
    # Pixel Intensity Distribution
    # Combine pixel values of all selected images for the histogram
    pixels_r = np.array([])
    pixels_g = np.array([])
    pixels_b = np.array([])
    for img_file in image_files:
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path)
        img_np = np.array(img)
        # Append the flattened pixel values of the red channel to pixels_r and repeat the same process for green and blue
        pixels_r = np.append(pixels_r, img_np[:, :, 0].ravel())
        pixels_g = np.append(pixels_g, img_np[:, :, 1].ravel())
        pixels_b = np.append(pixels_b, img_np[:, :, 2].ravel())
    
    # Figure for the pixel intensity distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pixels_r, bins=256, color='red', alpha=0.5, label='Red', density=True)
    plt.hist(pixels_g, bins=256, color='green', alpha=0.5, label='Green', density=True)
    plt.hist(pixels_b, bins=256, color='blue', alpha=0.5, label='Blue', density=True)
    plt.title(f'Pixel Intensity Distribution for {class_name}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


for folder, class_name in zip(folders, classes):
    display_samples_and_intensity(folder, class_name)