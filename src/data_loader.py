# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def load_data(data_dir, img_size=(128,128), batch_size=16):
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.2
#     )

#     train_gen = datagen.flow_from_directory(
#         data_dir,
#         target_size=img_size,
#         color_mode="grayscale",
#         batch_size=batch_size,
#         class_mode="categorical",
#         subset="training"
#     )

#     val_gen = datagen.flow_from_directory(
#         data_dir,
#         target_size=img_size,
#         color_mode="grayscale",
#         batch_size=batch_size,
#         class_mode="categorical",
#         subset="validation"
#     )

#     return train_gen, val_gen

# data_loader.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data_sequences(data_dir, seq_len=5, img_size=(128,128)):
    """
    Load MRI slices as sequences when data is structured as:
    data_dir / {CN,MCI,AD} / slices.jpg (no subject subfolders).
    Groups slices by subject_id prefix (before '_slice').
    """
    X, y = [], []
    classes = ["CN", "MCI", "AD"]
    class_map = {cls: i for i, cls in enumerate(classes)}

    print("📂 Loading data sequences...")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)

        # Collect slices per subject (group by prefix before '_slice')
        subject_map = {}
        for f in os.listdir(cls_dir):
            if f.endswith(".jpg"):
                subject_id = f.split("_slice")[0]
                subject_map.setdefault(subject_id, []).append(os.path.join(cls_dir, f))

        # For each subject, make sequences
        for subj, files in subject_map.items():
            all_files = sorted(files)  # keep slice order
            for i in range(0, len(all_files) - seq_len + 1, seq_len):
                seq_files = all_files[i:i+seq_len]
                seq_imgs = []
                for f in seq_files:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, img_size)
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=-1)
                    seq_imgs.append(img)
                X.append(np.array(seq_imgs))
                y.append(class_map[cls])

        print(f"✅ {cls}: {len(subject_map)} subjects, {len(X)} total sequences so far")

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(classes))

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Final shapes → X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    data_dir = r"C:\Users\ASUS\Downloads\ADNI_preprocessed"
    X_train, X_val, y_train, y_val = load_data_sequences(data_dir)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
