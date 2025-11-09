import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

IMG_SIZE = 64
BATCH_SIZE = 32
DATA_DIR = "./output/cow_hsv_pruned_1h/cow_01/"

# Load images as a TensorFlow dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None,
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Normalize the images
def normalize_img(image):
    image = tf.cast(image, tf.float32) / 255.0
    return (image, image)

train_dataset = train_dataset.map(normalize_img)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

# Encoder
inputs = Input(shape=INPUT_SHAPE)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(encoded)
x = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.summary()

# Train model & store history
history = autoencoder.fit(
    train_dataset,
    epochs=50,
    validation_data=train_dataset
)

# Save model
os.makedirs("models", exist_ok=True)
autoencoder.save("models/leg_autoencoder.h5")

# Create output folder for plots
os.makedirs("training_plots", exist_ok=True)

# Save training loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Autoencoder Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.savefig("training_plots/loss_curve.png")
plt.close()

# Save reconstructed samples
sample_imgs, _ = next(iter(train_dataset))
reconstructed = autoencoder.predict(sample_imgs)

n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(sample_imgs[i].numpy().squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.savefig("training_plots/reconstructions.png")
plt.close()

print("Training complete!")
print("Saved plots to ./training_plots/")
print("Saved model to ./models/leg_autoencoder.h5")

