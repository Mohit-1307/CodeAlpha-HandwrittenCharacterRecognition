import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

MODEL_NAME = "best_model.keras"

IMAGE_PATH = "digit.png"

IMAGE_SIZE = 28


# Load trained model
model = tf.keras.models.load_model(MODEL_NAME)

print(f"Model loaded: {MODEL_NAME}")

# Check image exists
if not os.path.exists(IMAGE_PATH):
    
    raise FileNotFoundError(f"Image not found: {os.path.abspath(IMAGE_PATH)}")

print(f"Reading image: {os.path.abspath(IMAGE_PATH)}")

# Load image in grayscale
image = cv2.imread(
    
    IMAGE_PATH, 
    
    cv2.IMREAD_GRAYSCALE
    
    )

if image is None:
    
    raise ValueError("Failed to load image")

# Reduce image noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Convert image to binary
threshold = cv2.adaptiveThreshold(

    blurred,

    255,

    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

    cv2.THRESH_BINARY_INV,

    11,

    2
)

# Improve digit shape
kernel = np.ones((3, 3), np.uint8)

threshold = cv2.dilate(

    threshold,

    kernel,

    iterations = 1
)

# Find digit contour
contours, _ = cv2.findContours(
    threshold,

    cv2.RETR_EXTERNAL,

    cv2.CHAIN_APPROX_SIMPLE

)

if not contours:
    
    raise ValueError("No digit detected in image")

# Select largest contour
largest_contour = max(
    
    contours,
    
    key = cv2.contourArea
    
)

x, y, width, height = cv2.boundingRect(largest_contour)

# Crop digit region
digit = threshold[
    
    y : y + height,
    
    x : x + width
    
]

# Resize while preserving aspect ratio
height, width = digit.shape

scale = 20 / max(height, width)

new_width = int(width * scale)

new_height = int(height * scale)

digit = cv2.resize(
    
    digit,
    
    (new_width, new_height)
    
)

# Create black canvas
canvas = np.zeros(
    
    (IMAGE_SIZE, IMAGE_SIZE),
    
    dtype = np.uint8
    
)

# Center digit on canvas
x_offset = (IMAGE_SIZE - new_width) // 2

y_offset = (IMAGE_SIZE - new_height) // 2

canvas[
    
    y_offset:y_offset + new_height,
    
    x_offset:x_offset + new_width
    
] = digit

# Normalize image
processed_image = (
    
    canvas.astype("float32") / 255.0
    
)

# Prepare input for CNN
model_input = processed_image.reshape(

    1,

    IMAGE_SIZE,

    IMAGE_SIZE,

    1

)

# Predict digit
prediction = model.predict(

    model_input,

    verbose = 0
    
    )

predicted_digit = int(np.argmax(prediction))

confidence = float(np.max(prediction))

# Show processed digit
plt.figure(figsize = (5, 5))

plt.imshow(processed_image, cmap = "gray")

plt.title(f"Prediction: {predicted_digit}")

plt.axis("off")

plt.show()

# Show prediction confidence
plt.figure(figsize = (8, 4))

plt.bar(range(10), prediction[0])

plt.xlabel("Digit")

plt.ylabel("Confidence")

plt.title("Prediction Confidence")

plt.xticks(range(10))

plt.show()

# Print results
print("\nPrediction Result")

print(f"Predicted Digit : {predicted_digit}")

print(f"Confidence Score: {confidence:.4f}")