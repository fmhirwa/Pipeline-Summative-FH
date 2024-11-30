# ML Pipeline App

## Overview
This Flask-based web application demonstrates an end-to-end Machine Learning pipeline using the MNIST dataset. The app allows users to:

- Upload images for prediction.
- Visualize model performance.
- Trigger model retraining with new data.

The project emphasizes scalability, ease of use, and performance testing.

## Features
1. **Prediction**: Upload an image and get the predicted class with confidence.
2. **Visualizations**: Explore dataset features and model evaluation metrics (e.g., accuracy, confusion matrix).
3. **Retraining**: Upload new datasets (e.g., images or CSV) and retrain the model.
4. **Performance Testing**: Flood testing with simulated requests. (WIP)

## Requirements
The following Python libraries are required:

- `Flask==3.1.0`
- `tensorflow-cpu==2.18.0`
- `Pillow==11.0.0`
- `Werkzeug==3.1.3`
- `numpy==2.0.2`
- `SciPy >= 1.6.0`
- `joblib >= 1.2.0`
- `threadpoolctl >= 3.1.0`
- `scikit-learn == 1.5.2`

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application locally:
   ```bash
   python app.py
   ```

5. Access the app:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Deployment
The app is deployed on Heroku and accessible at:
[Heroku App URL](https://ml-pipeline-minst-y3t2.herokuapp.com)

## Key Functionalities

### 1. **Prediction**
- Upload a single image (e.g., `.png` or `.jpg`).
- Dark background. Check the test_images folder for examples
- The app preprocesses the image and predicts its class using the trained model.

### 2. **Visualization**
- View training/validation accuracy and loss plots.
- Explore confusion matrix and feature distributions.

### 3. **Retraining**
- Upload a new dataset (e.g., multiple images or a CSV file).
- Retrain the model and update predictions with the latest weights.

### 4. **Performance Testing**
- Simulate multiple requests to evaluate app latency and scalability.
- Results available for review.

## Folder Structure
```
project/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── index.html         # Homepage
│   ├── upload.html        # File upload page
│   ├── result.html        # Prediction results
│   ├── evaluate_results.html # Evaluation metrics
├── uploaded_data/         # Folder to store uploaded files
├── saved_models/          # Folder to store trained/retrained models
├── static/                # Static assets (e.g., CSS, JS)
├── requirements.txt       # List of Python dependencies
├── main.py                # Main trained model 
├── README.md              # Project documentation
```

## Future Enhancements
- Add support for additional datasets.
- Enable model deployment on other platforms (e.g., AWS, GCP).
- Optimize for mobile-friendly usage.

## Authors
- **Florent Hirwa**

For any inquiries, please contact [m.hirwa@alustudent.com].
