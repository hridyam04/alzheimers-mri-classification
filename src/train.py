
import os
import matplotlib.pyplot as plt
from data_loader import load_data_sequences
from model import build_cnn_lstm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Paths ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimers_cnn_lstm.h5")

def train_model():
    # 1. Load data
    print("📂 Loading data...")
    data_dir = r"C:\Users\ASUS\Downloads\ADNI_preprocessed"  
    X_train, X_val, y_train, y_val = load_data_sequences(data_dir)

    # 2. Build model
    print(" Building CNN+LSTM model...")
    model = build_cnn_lstm(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

    # 3. Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    # 4. Train
    print("🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        #epochs=20,
        epochs=15,
        batch_size=8,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # 5. Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # 6. Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss")

    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
    plt.show()

    return history


if __name__ == "__main__":
    train_model()
