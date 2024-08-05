#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import shutil

"""
Code explanation:
Here we read over the first Dataset and we check for images for the given classes we want(Happy, Neutral, and surprised)
 and we store the ones that we need in a respective folder. They are labeled by class in a respective folder.

"""
#paths for the docs and files
#for happy images
annotation_file_path = "/Users/miro/Desktop/472 datastets/Emotion recognition 3/train/_annotations.txt"
source_directory = "/Users/miro/Desktop/472 datastets/Emotion recognition 3/train"
happy_directory = "/Users/miro/Desktop/472 datastets/Emotion recognition 3/happy_images"


#verif to see if our folder exists
os.makedirs(happy_directory, exist_ok=True)

with open(annotation_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        # Split the line by space and comma to find the emotion label and filename
        parts = line.split(',')
        # The last part is the emotion label
        emotion_label = parts[-1]
        if emotion_label == '4':
            # Assuming the filename is correctly identified before this point
            image_filename = parts[0].split(' ')[0]  # Adjust based on actual format
            # Check if the filename contains '.jpg'
            if '.jpg' in image_filename:
                # Construct full paths
                source_path = os.path.join(source_directory, image_filename)
                destination_path = os.path.join(happy_directory, image_filename)
                # Copy the image if it exists
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                else:
                    print(f"File not found: {source_path}")
                    
#for surprised images

surprised_directory = "/Users/miro/Desktop/472 datastets/Emotion recognition 3/surprised_images"



os.makedirs(surprised_directory, exist_ok=True)

with open(annotation_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        # Split the line by space and comma to find the emotion label and filename
        parts = line.split(',')
        # The last part is the emotion label
        emotion_label = parts[-1]
        if emotion_label == '7':
            # Assuming the filename is correctly identified before this point
            image_filename = parts[0].split(' ')[0]  # Adjust based on actual format
            # Check if the filename contains '.jpg'
            if '.jpg' in image_filename:
                # Construct full paths
                source_path = os.path.join(source_directory, image_filename)
                destination_path = os.path.join(surprised_directory, image_filename)
                # Copy the image if it exists
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                else:
                    print(f"File not found: {source_path}")

#for neutral pictures

neutral_directory = "/Users/miro/Desktop/472 datastets/Emotion recognition 3/neutral_images"


#verif to see if our folder exists
os.makedirs(neutral_directory, exist_ok=True)

with open(annotation_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        # Split the line by space and comma to find the emotion label and filename
        parts = line.split(',')
        # The last part is the emotion label
        emotion_label = parts[-1]
        if emotion_label == '5':
            # Assuming the filename is correctly identified before this point
            image_filename = parts[0].split(' ')[0]  # Adjust based on actual format
            # Check if the filename contains '.jpg'
            if '.jpg' in image_filename:
                # Construct full paths
                source_path = os.path.join(source_directory, image_filename)
                destination_path = os.path.join(neutral_directory, image_filename)
                # Copy the image if it exists
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                else:
                    print(f"File not found: {source_path}")