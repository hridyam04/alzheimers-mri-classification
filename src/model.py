
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def build_cnn_lstm(input_shape=(5,128,128,1), num_classes=3):
#     """
#     CNN extracts slice-level features,
#     LSTM learns dependencies between slices in a sequence.
#     """

#     # CNN feature extractor
#     cnn = models.Sequential([
#         layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
#         layers.MaxPooling2D((2,2)),
#         layers.Conv2D(64, (3,3), activation="relu"),
#         layers.MaxPooling2D((2,2)),
#         layers.Flatten()
#     ])

#     # Combine with LSTM
#     model = models.Sequential([
#         layers.TimeDistributed(cnn, input_shape=input_shape),
#         layers.LSTM(64, return_sequences=False),
#         layers.Dense(64, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax")
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     return model

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def build_cnn_lstm(input_shape=(5, 128, 128, 1), num_classes=3):
    """
    CNN (VGG16) extracts slice-level spatial features,
    LSTM learns inter-slice dependencies.
    """

    # --- Base CNN feature extractor (VGG16) ---
    base_cnn = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base_cnn.trainable = False  # Freeze pretrained layers to save computation

    # Convert grayscale (1 channel) to 3-channel to match VGG16 input
    cnn_input = layers.Input(shape=(128,128,1))
    x = layers.Concatenate()([cnn_input, cnn_input, cnn_input])  # (128,128,3)
    x = base_cnn(x)
    x = layers.GlobalAveragePooling2D()(x)
    cnn_extractor = models.Model(cnn_input, x)

    # --- Combine with LSTM ---
    model = models.Sequential([
        layers.TimeDistributed(cnn_extractor, input_shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
