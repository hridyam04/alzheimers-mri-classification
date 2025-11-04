import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data_sequences  # same function used in training

# --- Paths ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimers_cnn_lstm.h5")
DATA_DIR = r"C:\Users\ASUS\Downloads\ADNI_preprocessed"  

def evaluate_model():
    print(" Loading validation/test data...")
    _, X_val, _, y_val = load_data_sequences(DATA_DIR)

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate
    print("\n Evaluating model on validation data...")
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
    print(f" Validation Loss: {loss:.4f}")

    # Predictions
    print("\n Generating predictions...")
    y_pred = model.predict(X_val)
    y_true_classes = np.argmax(y_val, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # --- Classification report ---
    print("\n Classification Report:")
    target_names = ["CN", "MCI", "AD"]
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.show()

    print(f" Confusion matrix saved at {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

if __name__ == "__main__":
    evaluate_model()
