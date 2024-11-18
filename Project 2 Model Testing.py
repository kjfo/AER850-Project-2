# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils, models, preprocessing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


""" Model Testing """
# Define Input Image Shape and Batch Size.
image_shape = (128, 128, 1)     # Recommended: 500, 500, 3
batch_size = 32

test_directory = r'C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\AER850-Project-2\AER850-Project-2\Project 2 Data\Data\test'
test_directory = os.path.realpath(test_directory)

# # Turn RGB Images to Grayscale.
# def preprocess_image(image, label):
#     # Images Don't Need to be RGB. Just Need Key Features.
#     image = tf.image.rgb_to_grayscale(image)
#     return image, label

# Create Test Generator.
test_generator = utils.image_dataset_from_directory(
    directory = test_directory,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1]),
    seed = 10,
    color_mode = "grayscale",
    batch_size = batch_size,
    shuffle = False,
)
# ).map(preprocess_image)

# Load Model
model = models.load_model("model_20241117_225601.keras", safe_mode = False)

# Evaluate Test Images Using the Model.
test_loss, test_acc = model.evaluate(x = test_generator)

# Load the Test Images in an Array.
test_directory1 = r'C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\AER850-Project-2\AER850-Project-2\Project 2 Data\Data\test\crack\test_crack.jpg'
test_directory1 = os.path.realpath(test_directory1)
test_directory2 = r'C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\AER850-Project-2\AER850-Project-2\Project 2 Data\Data\test\missing-head\test_missinghead.jpg'
test_directory2 = os.path.realpath(test_directory2)
test_directory3 = r'C:\Users\kjfo0\OneDrive\Desktop\AER850 - Python Codes\AER850-Project-2\AER850-Project-2\Project 2 Data\Data\test\paint-off\test_paintoff.jpg'
test_directory3 = os.path.realpath(test_directory3)

test_image_paths = [
    test_directory1,
    test_directory2,
    test_directory3,
]

# Preprocess the Selected Test Images.
test_image_arrays = []

for test_image_path in test_image_paths:
    image = preprocessing.image.load_img(
        test_image_path,
        target_size = (image_shape[0], image_shape[1]),
        color_mode = "grayscale"          # Again, just need key features.
    )
    test_image_arr = preprocessing.image.img_to_array(image)
    test_image_arr = np.expand_dims(test_image_arr, axis = 0)
    test_image_arrays.append(test_image_arr)

# Stack the Test Images.
test_image_single_arr = np.vstack(test_image_arrays)

# Get Predictions Using the Saved Model.
predictions = model.predict(test_image_single_arr)

# Crack Classes.
crack_classes = ['Crack', 'Missing Head', 'Paint-off']

# Display Final Results.
for i, test_image_path in enumerate(test_image_paths):
    test_image_disp = cv2.imread(test_image_path)
    test_image_disp = cv2.resize(test_image_disp, (500, 500))
    
    
    # Showing the Probabilities in the Test Image.
    for j, (label, probability) in enumerate(zip(crack_classes, predictions[i])):
        text = f"{label}: {probability * 100:.2f}%"
        position = (5, 50 + j * 25)
        cv2.putText(test_image_disp, text, position, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Show the image with matplotlib
    plt.figure(figsize=(4,4))
    plt.imshow(test_image_disp)
    plt.axis('off')
    plt.title(f"True Class: {crack_classes[i]}\nPredicted Class: {crack_classes[np.argmax(predictions[i])]}")
    plt.show()