import tensorflow as tf
import numpy as np
import os

IMG_SIZE = 64
BATCH_SIZE = 32
DATA_DIR = "output/grouped_legs/group_0/"

# Load images as a TensorFlow dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None,
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Normalize the images from [0, 255] to [0, 1]
def normalize_img(image):
    image = tf.cast(image, tf.float32) / 255.0
    return (image, image)

train_dataset = train_dataset.map(normalize_img)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1) # (64, 64, 1) for grayscale

# --- 1. The Encoder ---
# This compresses the image
inputs = Input(shape=INPUT_SHAPE)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x) # Now 32x32
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x) # Now 16x16
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) # "Latent Vector" - Now 8x8x8

# --- 2. The Decoder ---
# This reconstructs the image from the latent vector
x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(encoded) # Now 16x16
x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x) # Now 32x32
x = Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x) # Now 64x64
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # Back to 1 channel

# --- 3. The Autoencoder Model ---
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.summary()

autoencoder.fit(
    train_dataset,
    epochs=50,
    validation_data=train_dataset
)

autoencoder.save("models/leg_autoencoder.h5")
