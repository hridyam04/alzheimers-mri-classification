

# def build_cnn_lstm(input_shape=(5,128,128,1), num_classes=3):
#     cnn = models.Sequential([
#         layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
#         layers.MaxPooling2D(2,2),
#         layers.Conv2D(64, (3,3), activation="relu"),
#         layers.MaxPooling2D(2,2),
#         layers.Flatten()
#     ])

#     model = models.Sequential([
#         layers.TimeDistributed(cnn, input_shape=input_shape),
#         layers.LSTM(64),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(num_classes, activation="softmax")
#     ])

#     return model


# # model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape=(5,128,128,1), num_classes=3):
    """
    CNN extracts slice-level features,
    LSTM learns dependencies between slices in a sequence.
    """

    # CNN feature extractor
    cnn = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten()
    ])

    # Combine with LSTM
    model = models.Sequential([
        layers.TimeDistributed(cnn, input_shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
