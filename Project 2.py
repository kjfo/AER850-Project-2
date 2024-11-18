# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils, layers, models, callbacks, regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import save_model
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import constants

print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


""" Data Processing """
# Define Input Image Shape and Batch Size.
image_shape = (128, 128, 1)     # Recommended: 500, 500, 3
batch_size = 32

# Establish Train, Validation, and Test Directories.
data_folder = os.path.join("..", "data")
model_folder = os.path.join("..", "models")
tune_folder = os.path.join("..", "tuning")

train_directory = os.path.join(data_folder, "train")
validation_directory = os.path.join(data_folder, "valid")
test_directory = os.path.join(data_folder, "test")

# Turn RGB Images to Grayscale.
def preprocess_image(image, label):
    # Images Don't Need to be RGB. Just Need Key Features.
    image = tf.image.rgb_to_grayscale(image)
    return image, label

# Create Train and Validation Generators.
train_generator = utils.image_dataset_from_directory(
    directory = train_directory,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1]),
    seed = 10,
    batch_size = batch_size,
).map(preprocess_image)

validation_generator = utils.image_dataset_from_directory(
    directory = validation_directory,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1]),
    seed = 10,
    batch_size = batch_size,
).map(preprocess_image)

# test_generator = utils.image_dataset_from_directory(
#     directory = test_directory,
#     labels = "inferred",
#     image_size = (image_shape[0], image_shape[1]),
#     seed = 10,
#     batch_size = batch_size,
# ).map(preprocess_image)


""" Neural Network Architecture Design """
# Build the Neural Network
model = models.Sequential(
    
    [
    # Perform Data Augmentation.
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomZoom(0.2),
    layers.Rescaling(scale = 1.0/255),

    layers.Conv2D(64, (3, 3), strides = (2, 2), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (1, 1)),
    layers.Conv2D(64, (3, 3), strides = (2, 2), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (1, 1)),
    layers.Conv2D(128, (3, 3), strides = (2, 2), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (2, 2), strides = (1, 1)),
    layers.Conv2D(256, (3, 3), strides = (2, 2), activation = 'relu'),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    
    layers.Dense(100, activation = 'relu', kernel_regularizer = regularizers.L1L2()),
    layers.Dropout(0.35),                                                             # Can adjust dropout rate.
    layers.Dense(100, activation = 'relu', kernel_regularizer = regularizers.L1L2()),
    layers.Dropout(0.35), 
    layers.Dense(3, activation = 'softmax', kernel_regularizer = regularizers.L1L2()),
    ]
)


""" Model Evaluation """
model.compile(optimizer = Adam(),
              loss = SparseCategoricalCrossentropy(),
              metrics = [Accuracy()]
)

# To Interrupt the Training Process When Validation Loss is No Longer Improving.
early_stopping = callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 15,
    restore_best_weights = True,
)

history = model.fit(train_generator,                             # Aim is to get 70-80% accuracy
                    validation_data = validation_generator,
                    epochs = 75,
                    callbacks = [early_stopping],
                    steps_per_epoch = 55,
)

# Save the Model.
model_path = os.path.join(model_folder, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
model.save(model_path)

# Plot Accuracy and Loss for Both Training and Validation Data.
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'Training Data Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'Training Data Loss')
plt.plot(history.history['val_loss'], label = 'Validation Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title('Model Loss')

plt.tight_layout()
plt.show()
