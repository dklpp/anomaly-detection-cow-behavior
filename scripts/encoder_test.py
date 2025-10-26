import cv2
import sys
import tensorflow as tf
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 32

def get_reconstruction_error(img_path, model):
    # Load and preprocess the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=[0, -1])

    # Get the reconstructed image
    reconstructed_img = model.predict(img)

    # Calculate the reconstruction error (MSE)
    error = np.mean(np.square(img - reconstructed_img))
    return error

model = tf.keras.models.load_model("models/leg_autoencoder.h5")

# 1. Find a good threshold by checking errors on images
normal_image_path = "output/grouped_legs/group_0/leg_1106_471_151.png"
normal_error = get_reconstruction_error(normal_image_path, model)
print(f"Normal image error: {normal_error}") 

ANOMALY_THRESHOLD = 0.01 # Need to change when validated

# 2. Test an anomalous image
anomalous_image_path = "output/grouped_legs/group_0/leg_1126_469_133.png"
anomaly_error = get_reconstruction_error(anomalous_image_path, model)
print(f"Anomalous image error: {anomaly_error}")

if anomaly_error > ANOMALY_THRESHOLD:
    print("Anomaly Detected!")
