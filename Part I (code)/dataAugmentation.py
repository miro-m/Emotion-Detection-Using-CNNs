#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle


"""
Code explanation:
    This code right here performs some data augmentation on the engaged images as a test, it does not change the images permenantly.
    During the training/testing process we might do some permanant changes to our images, but for this was just used to check the distribution on the pixel intensity after the changes for now.
"""
# Path to your local machine where the images are
folder_path = '/Users/miro/Desktop/472 datastets/Emotion recognition 3/focused:engaged_images'

# Load image paths
image_paths = shuffle([
    os.path.join(folder_path, image_file) for image_file in os.listdir(folder_path) 
    if image_file.lower().endswith(('.jpg', '.jpeg'))
])

# Function to perform data augmentation
def augment_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Randomly adjust brightness (70-130% of the original image)
    brightness_factor = np.random.uniform(0.7, 1.3)
    image_bright = ImageEnhance.Brightness(image).enhance(brightness_factor)
    
    # Randomly adjust contrast (70-130% of the original image)
    contrast_factor = np.random.uniform(0.7, 1.3)
    image_contrast = ImageEnhance.Contrast(image_bright).enhance(contrast_factor)
    
    return image_contrast

# Accumulate pixel values for the augmented images
pixels_r = np.array([])
pixels_g = np.array([])
pixels_b = np.array([])

# Augment images and calculate pixel intensity distribution
for image_path in image_paths:  
    augmented_image = augment_image(image_path)
    augmented_image_np = np.array(augmented_image)
    
    # Accumulate the pixel values for each color channel
    pixels_r = np.append(pixels_r, augmented_image_np[:, :, 0].flatten())
    pixels_g = np.append(pixels_g, augmented_image_np[:, :, 1].flatten())
    pixels_b = np.append(pixels_b, augmented_image_np[:, :, 2].flatten())

# Plot the pixel intensity distribution for the augmented images
plt.figure(figsize=(10, 6))
plt.hist(pixels_r, bins=256, color='red', alpha=0.5, label='Red')
plt.hist(pixels_g, bins=256, color='green', alpha=0.5, label='Green')
plt.hist(pixels_b, bins=256, color='blue', alpha=0.5, label='Blue')
plt.title('Pixel Intensity Distribution for Augmented Engaged/Focused Images')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

