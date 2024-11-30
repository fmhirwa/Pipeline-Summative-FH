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

        # Pass the results to the template
        return render_template('evaluate_results.html', 
                               accuracy=accuracy, 
                               precision=precision, 
                               recall=recall, 
                               f1_score=f1, 
                               confusion_matrix=conf_matrix.tolist())

    return render_template('evaluate.html')


if __name__ == '__main__':
    app.run(debug=True)
