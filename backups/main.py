# Temp fixes for encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load and preprocess MNIST data
import numpy as np
from tensorflow.python.keras.layers import Input, Dense

def load_images(file_path):
    with open(file_path, 'rb') as f:
        # Read and parse IDX file headers
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # Load image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read and parse IDX file headers
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # Load label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load training data
train_images = load_images('./archive/train-images.idx3-ubyte')
train_labels = load_labels('./archive/train-labels.idx1-ubyte')

# Load testing data
test_images = load_images('./archive/t10k-images.idx3-ubyte')
test_labels = load_labels('./archive/t10k-labels.idx1-ubyte')

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing images shape: {test_images.shape}")
print(f"Testing labels shape: {test_labels.shape}")

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand dimensions for convolutional input
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.applications import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

# Define a regularized and dropout-augmented model
import tensorflow as tf
from tensorflow.keras import layers, regularizers

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

# Compile the model with learning rate scheduling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=20,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save the model in TensorFlow format
model.save('./saved_models/mnist_model.h5')

# Save using Joblib for scikit-learn integration
import joblib
joblib.dump(model, './saved_models/mnist_model.pkl')



# 6 Plotting
import os
import matplotlib.pyplot as plt

# Directory to save plots
output_dir = './output_plots'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Example plot function with saving
def plot_and_save(history):
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plot_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to: {plot_path}")
    plt.close()  # Close the plot

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the plot
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to: {plot_path}")
    plt.close()  # Close the plot

def plot_confusion_matrix_and_save(test_labels, y_pred_classes, output_dir):
    from sklearn.metrics import confusion_matrix
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

    # Save the plot
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close()

# Example usage
plot_confusion_matrix_and_save(test_labels, y_pred_classes, output_dir)



# Generate and save plots
plot_and_save(history)
