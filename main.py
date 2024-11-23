# Temp fixes for encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load and preprocess MNIST data
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load MNIST data functions
def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load training and testing data
train_images = load_images('./archive/train-images.idx3-ubyte')
train_labels = load_labels('./archive/train-labels.idx1-ubyte')
test_images = load_images('./archive/t10k-images.idx3-ubyte')
test_labels = load_labels('./archive/t10k-labels.idx1-ubyte')

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand dimensions
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# 2. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

# 3. Define CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# 4. Train the model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=100,  # Low for faster training
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, reduce_lr]
)

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save model in recommended Keras format
model.save('./saved_models/mnist_model.keras')

# 6. Generate predictions
y_pred = model.predict(test_images)
y_pred_classes = y_pred.argmax(axis=1)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(test_labels, y_pred_classes, average='weighted')
recall = recall_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Directory for plots
output_dir = './output_plots'
os.makedirs(output_dir, exist_ok=True)

# Plot accuracy and loss
def plot_and_save(history, output_dir):
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix_and_save(test_labels, y_pred_classes, output_dir):
    conf_matrix = confusion_matrix(test_labels, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Generate and save plots
plot_and_save(history, output_dir)
plot_confusion_matrix_and_save(test_labels, y_pred_classes, output_dir)
