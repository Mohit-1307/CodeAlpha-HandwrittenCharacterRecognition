import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load Dataset
# =========================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =========================
# Normalize
# =========================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# =========================
# Reshape for CNN
# =========================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# =========================
# Save original labels
# =========================

y_test_original = y_test

# =========================
# One-hot encoding
# =========================

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =========================
# Data Augmentation
# =========================

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train)

# =========================
# Build CNN Model
# =========================

model = models.Sequential([

    layers.Input(shape=(28,28,1)),

    layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.3),

    layers.Dense(
        10,
        activation='softmax'
    )
])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Early Stopping
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# Train Model
# =========================

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# =========================
# Evaluate Model
# =========================

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")

# =========================
# Predictions
# =========================

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)

# =========================
# Classification Report
# =========================

print("\nClassification Report:\n")

print(classification_report(
    y_test_original,
    y_pred_classes
))

# =========================
# Save Model
# =========================

model.save("model.keras")

print("\nModel saved successfully!")

# =========================
# Plot Accuracy
# =========================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.legend()

plt.show()

# =========================
# Plot Loss
# =========================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")

plt.legend()

plt.show()