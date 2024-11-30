from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploaded_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
MODEL_PATH = './saved_models/mnist_model.keras'
model = load_model(MODEL_PATH)

# Load test data (assuming preprocessed and ready for evaluation)
def load_test_data():
    test_images = np.load('./archive/test_images.npy')
    test_labels = np.load('./archive/test_labels.npy')
    return test_images, test_labels

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading files
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', message="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message="No selected file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')

# Route for making predictions
@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Preprocess image
    img = Image.open(file_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return render_template('result.html', 
                           filename=filename, 
                           predicted_class=predicted_class, 
                           confidence=confidence)

# Route for retraining

def load_training_data():
    train_images = np.load('./archive/train_images.npy')
    train_labels = np.load('./archive/train_labels.npy')
    return train_images, train_labels

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        # Load training data
        train_images, train_labels = load_training_data()

        # Set up a checkpoint to save the model during retraining
        checkpoint_path = './saved_models/retrained_model.keras'
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
        
        # Prevent overfitting using early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Retrain the model
        history = model.fit(
            train_images, train_labels,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[checkpoint]
        )

        # Save the final model
        model.save(checkpoint_path)

        return jsonify({
            'message': 'Model retrained successfully!',
            'training_history': {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            }
        })
    else:
        # Return a message for GET requests
        return render_template('retrain.html')


# Route for evaluation
@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        # Load test data
        test_images, test_labels = load_test_data()

        # Make predictions on test data
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, predicted_classes)
        precision = precision_score(test_labels, predicted_classes, average='weighted')
        recall = recall_score(test_labels, predicted_classes, average='weighted')
        f1 = f1_score(test_labels, predicted_classes, average='weighted')
        conf_matrix = confusion_matrix(test_labels, predicted_classes)
        # Plot 1
        #plot_path = plot_per_class_metrics(test_labels, predicted_classes, class_names)
        # Plot 2
        #misclassified_plot_path = plot_misclassified_images(test_images, test_labels, predicted_classes, class_names)

        # Pass the results to the template
        return render_template('evaluate_results.html', 
                               accuracy=accuracy, 
                               precision=precision, 
                               recall=recall, 
                               f1_score=f1, 
                               confusion_matrix=conf_matrix.tolist())

    return render_template('evaluate.html')
"""
    import matplotlib.pyplot as plt

    def plot_per_class_metrics(test_labels, y_pred_classes, class_names, output_dir='./static/plots'):
        from sklearn.metrics import precision_recall_fscore_support
        import os

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred_classes, labels=range(len(class_names)))

        # Create the plot
        x = np.arange(len(class_names))
        width = 0.25
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-Score')
        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Per-Class Metrics (Precision, Recall, F1-Score)')
        plt.xticks(x, class_names)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'per_class_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    # Call the function 
    plot_path = plot_per_class_metrics(test_labels, predicted_classes, class_names)

    def plot_misclassified_images(test_images, test_labels, y_pred_classes, class_names, num_images=10, output_dir='./static/plots'):

        # Find misclassified indices
        misclassified_indices = np.where(test_labels != y_pred_classes)[0]

        # Select a subset to display
        num_images = min(num_images, len(misclassified_indices))
        selected_indices = np.random.choice(misclassified_indices, num_images, replace=False)

        # Plot the images
        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(selected_indices):
            plt.subplot(2, (num_images + 1) // 2, i + 1)
            plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {class_names[test_labels[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
            plt.axis('off')
        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'misclassified_examples.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    # Call the function in the evaluate route
    misclassified_plot_path = plot_misclassified_images(test_images, test_labels, predicted_classes, class_names)
"""


if __name__ == '__main__':
    # Get the PORT from environment variables
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)
