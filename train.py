import random

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers, models

import json

from sklearn.metrics import(accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score )

from tensorflow.keras.callbacks import(EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)


SEED = 42

NUM_CLASSES = 10

BATCH_SIZE = 64

EPOCHS = 20

INPUT_SHAPE = (28, 28, 1)

MODEL_NAME = "best_model.keras"

# Set seed for reproducible results
random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN
x_train = np.expand_dims(x_train, axis = -1)

x_test = np.expand_dims(x_test, axis = -1)

# Save original labels for evaluation
y_test_labels = y_test.copy()

# Convert labels to categorical format
y_train = to_categorical(y_train, NUM_CLASSES)

y_test = to_categorical(y_test, NUM_CLASSES)

# Data augmentation
train_datagen = ImageDataGenerator(

    rotation_range = 8,

    zoom_range = 0.08,

    width_shift_range = 0.08,

    height_shift_range = 0.08,

)

# Build CNN model
model = models.Sequential(
    
    [
        
        layers.Input(shape=INPUT_SHAPE),
        
        layers.Conv2D(

            32,

            (3, 3),

            padding = "same",

            activation = "relu"

        ),
        
        layers.BatchNormalization(),

        layers.Conv2D(

            32,

            (3, 3),

            padding = "same",

            activation = "relu"

        ),
        
        layers.MaxPooling2D((2, 2)),
        
        layers.Dropout(0.2),

        layers.Conv2D(

            64,

            (3, 3),

            padding = "same",

            activation = "relu"

        ),
        
        layers.BatchNormalization(),

        layers.Conv2D(

            64,

            (3, 3),

            padding = "same",

            activation = "relu"

        ),
        
        layers.MaxPooling2D((2, 2)),
        
        layers.Dropout(0.3),

        layers.Flatten(),

        layers.Dense(

            128,

            activation = "relu"

        ),
        
        layers.Dropout(0.4),

        layers.Dense(

            NUM_CLASSES,

            activation = "softmax"

        )
        
    ]
    
)

# Compile model
model.compile(

    optimizer = "adam",

    loss = "categorical_crossentropy",

    metrics = ["accuracy"]

)

# Stop training if validation loss stops improving
early_stopping = EarlyStopping(

    monitor = "val_loss",

    patience = 4,

    restore_best_weights = True

)

# Reduce learning rate if model stops improving
reduce_lr = ReduceLROnPlateau(

    monitor = "val_loss",

    factor = 0.5,

    patience = 2,

    min_lr = 1e-6,

    verbose = 1

)

# Save best model automatically
checkpoint = ModelCheckpoint(

    filepath = MODEL_NAME,

    monitor = "val_accuracy",

    save_best_only = True,

    verbose = 1
)

# Train model
history = model.fit(

    train_datagen.flow(

        x_train,

        y_train,

        batch_size = BATCH_SIZE

    ),
    
    epochs = EPOCHS,
    
    validation_data = (x_test, y_test),
    
    callbacks = [

        early_stopping,

        reduce_lr,

        checkpoint

    ],
    
    verbose = 1

)

# Evaluate model
test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test,

    verbose = 0

)

# Generate predictions
predictions = model.predict(

    x_test,

    verbose = 0

)

predicted_labels = np.argmax(

    predictions,

    axis = 1

)

# Calculate evaluation metrics
accuracy = accuracy_score(

    y_test_labels,

    predicted_labels

)

precision = precision_score(

    y_test_labels,

    predicted_labels,

    average = "weighted"

)

recall = recall_score(

    y_test_labels,

    predicted_labels,

    average = "weighted"

)

f1 = f1_score(

    y_test_labels,

    predicted_labels,

    average = "weighted"

)

roc_auc = roc_auc_score(

    y_test,

    predictions,

    multi_class = "ovr"
    
)

# Print model performance
print("\nModel Performance")  

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"ROC-AUC : {roc_auc:.4f}")

# Print classification report
print("\nClassification Report\n")

print(classification_report(y_test_labels, predicted_labels))

# Plot training accuracy
plt.figure(figsize = (8, 4))

plt.plot(history.history["accuracy"], label = "Training")

plt.plot(history.history["val_accuracy"], label = "Validation")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Model Accuracy")

plt.legend()

plt.tight_layout()

plt.show()

# Plot training loss
plt.figure(figsize = (8, 4))

plt.plot(history.history["loss"], label = "Training")

plt.plot(history.history["val_loss"], label = "Validation")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Model Loss")

plt.legend()

plt.tight_layout()

plt.show()

# Show confusion matrix
conf_matrix = confusion_matrix(y_test_labels, predicted_labels)

fig, ax = plt.subplots(figsize = (8, 8))

ConfusionMatrixDisplay(confusion_matrix = conf_matrix,).plot(ax = ax)

plt.title("Confusion Matrix")

plt.show()

# Show sample predictions
plt.figure(figsize = (12, 6))

for index in range(10):
    
    plt.subplot(2, 5, index + 1)

    plt.imshow(x_test[index].squeeze(), cmap = "gray")

    predicted = predicted_labels[index]
    
    actual = y_test_labels[index]

    plt.title(f"Pred : {predicted}\nTrue: {actual}")

    plt.axis("off")

plt.tight_layout()

plt.show()

print(f"\nBest model saved as '{MODEL_NAME}'")


metrics = {

    "accuracy" : float(accuracy),

    "precision" : float(precision),

    "recall" : float(recall),

    "f1_score" : float(f1),

    "roc_auc" : float(roc_auc)

}

# Save Metrics
with open("metrics.json", "w") as file:
    
    json.dump(metrics, file, indent = 4)

print("\nMetrics saved to metrics.json")