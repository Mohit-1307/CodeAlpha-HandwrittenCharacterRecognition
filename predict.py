import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# =========================
# Load Trained CNN Model
# =========================

model = tf.keras.models.load_model("model.keras")

# =========================
# Image Path
# =========================

image_path = "digit.png"

print("Looking for image at:")
print(os.path.abspath(image_path))

# =========================
# Check File Exists
# =========================

if not os.path.exists(image_path):
    print("\nERROR: Image file not found!")
    exit()

# =========================
# Load Image
# =========================

img = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

# =========================
# Verify Image Loaded
# =========================

if img is None:
    print("\nERROR: Unable to load image!")
    exit()

# =========================
# Gaussian Blur
# =========================

img = cv2.GaussianBlur(
    img,
    (5, 5),
    0
)

# =========================
# Adaptive Threshold
# =========================

thresh = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# =========================
# Morphological Dilation
# =========================

kernel = np.ones((3, 3), np.uint8)

thresh = cv2.dilate(
    thresh,
    kernel,
    iterations=1
)

# =========================
# Find Contours
# =========================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# =========================
# Ensure Digit Exists
# =========================

if len(contours) == 0:
    print("\nNo digit detected!")
    exit()

# =========================
# Largest Contour
# =========================

largest_contour = max(
    contours,
    key=cv2.contourArea
)

# =========================
# Bounding Box
# =========================

x, y, w, h = cv2.boundingRect(
    largest_contour
)

# =========================
# Crop Digit
# =========================

digit = thresh[y:y+h, x:x+w]

# =========================
# Resize Digit
# =========================

digit = cv2.resize(
    digit,
    (20, 20)
)

# =========================
# Create 28x28 Canvas
# =========================

canvas = np.zeros(
    (28, 28),
    dtype=np.uint8
)

# =========================
# Center Digit
# =========================

canvas[4:24, 4:24] = digit

# =========================
# Normalize
# =========================

canvas = canvas.astype("float32") / 255.0

# =========================
# Display Processed Digit
# =========================

plt.imshow(canvas, cmap="gray")

plt.title("Processed Digit")

plt.axis("off")

plt.show()

# =========================
# Reshape for CNN
# =========================

canvas = canvas.reshape(
    1,
    28,
    28,
    1
)

# =========================
# Predict Digit
# =========================

prediction = model.predict(canvas)

predicted_digit = np.argmax(
    prediction
)

confidence = np.max(
    prediction
)

# =========================
# Output Results
# =========================

print("\nPrediction Results")
print("-------------------")

print(f"Predicted Digit: {predicted_digit}")

print(f"Confidence: {confidence:.4f}")