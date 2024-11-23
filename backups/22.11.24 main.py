#temp

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
sys.stdout.reconfigure(encoding='utf-8')

#end temp

import numpy as np

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


#2

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# If using a dense model, flatten the images
train_images_flat = train_images.reshape(-1, 28 * 28)
test_images_flat = test_images.reshape(-1, 28 * 28)

#3

import tensorflow as tf

# Define a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

#4

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

#5

# Save the model in TensorFlow format
import joblib
joblib.dump(model, './saved_models/minst_model.pkl')

# Plotting
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Assume training history from model.fit() is stored in `history`
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 1. Plot Training and Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 2. Plot Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 3. Calculate Precision, Recall, and F1 Score
y_pred = model.predict(test_images)
y_pred_classes = y_pred.argmax(axis=1)

precision = precision_score(test_labels, y_pred_classes, average='weighted')
recall = recall_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

# Print metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 4. Plot Precision, Recall, F1 Score
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Precision, Recall, F1 Score')
plt.ylabel('Score')
plt.show()

# 5. Plot Confusion Matrix
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

plt.show()
