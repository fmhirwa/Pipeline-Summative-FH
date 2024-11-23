import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def plot_accuracy_and_loss(history):
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_and_plot_metrics(test_labels, y_pred_classes):
    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(test_labels, y_pred_classes, average='weighted')
    recall = recall_score(test_labels, y_pred_classes, average='weighted')
    f1 = f1_score(test_labels, y_pred_classes, average='weighted')

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Plot Precision, Recall, F1 Score
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.ylim(0, 1)
    plt.title('Precision, Recall, F1 Score')
    plt.ylabel('Score')
    plt.show()

def plot_confusion_matrix(test_labels, y_pred_classes):
    # Generate Confusion Matrix
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
