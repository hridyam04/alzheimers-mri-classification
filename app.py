# import os
# from flask import Flask, request, render_template, jsonify
# import tensorflow as tf
# import cv2
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('models/alzheimers_cnn_lstm.h5')

# def preprocess_image(image_file):
#     """Preprocess the uploaded image."""
#     # Read image file
#     image = Image.open(image_file)
#     # Convert to grayscale
#     image = image.convert('L')
#     # Resize to match training dimensions
#     image = image.resize((128, 128))
#     # Convert to numpy array
#     img_array = np.array(image)
#     # Normalize
#     img_array = img_array / 255.0
#     # Duplicate the single slice 5 times to match the expected sequence length
#     img_array = np.stack([img_array] * 5)
#     # Reshape for model input (adding batch and channel dimensions)
#     img_array = img_array.reshape(1, 5, 128, 128, 1)
#     return img_array

# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file uploaded'})
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'})

#         # Process the image
#         processed_image = preprocess_image(file)
        
#         # Make prediction
#         prediction = model.predict(processed_image)
        
#         # Get the class with highest probability
#         classes = ['AD', 'CN', 'MCI']
#         predicted_class = classes[np.argmax(prediction)]
#         confidence = float(np.max(prediction)) * 100
        
#         return jsonify({
#             'prediction': predicted_class,
#             'confidence': f'{confidence:.2f}%'
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     # Create templates directory if it doesn't exist
#     os.makedirs('templates', exist_ok=True)
#     app.run(debug=True)
# app.py
# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
import sys
import subprocess
import tensorflow as tf

st.set_page_config(page_title="Alzheimer's MRI Classifier", layout="wide")
# Add src/ to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
st.write("Current working directory:", os.getcwd())

# ---------- Config ----------
MODEL_PATHS = [
    os.path.join("models", "alzheimers_2dcnn.h5"),
    os.path.join("models", "alzheimers_cnn_lstm.h5")  # fallback
]
TRAIN_SCRIPT_CANDIDATES = [os.path.join("src", "train.py")]
EVAL_SCRIPT_CANDIDATES = [os.path.join("src", "evaluate.py")]

IMG_SIZE = (128, 128)
CLASS_NAMES = ["CN", "MCI", "AD"]

# ---------- Page setup ----------

st.title("🧠 Alzheimer's MRI Classification System")
st.caption(
    "Demonstration of a deep learning–based MRI classification system for Alzheimer's detection — "
    "integrating training, evaluation, and real-time image prediction."
)

# ---------- Utility functions ----------
@st.cache_resource
def load_trained_model():
    """Load pre-trained model if available."""
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = tf.keras.models.load_model(p)
                st.success(f"✅ Model loaded successfully from: {p}")
                return model, p
            except Exception as e:
                st.error(f"Found model at {p} but failed to load: {e}")
                return None, p
    return None, None


def preprocess_image_2d(pil_image, img_size=IMG_SIZE, model=None):
    """
    Converts a single uploaded JPEG to the correct format for the model.
    Works for both 2D CNNs and CNN-LSTM (VGG16-based) models.
    If model expects sequences, the same image is repeated across timesteps.
    """
    # 1️⃣ Convert to grayscale, resize, normalize
    img = pil_image.convert("L")
    arr = np.array(img)
    arr = cv2.resize(arr, img_size)
    arr = arr.astype("float32") / 255.0  # (128, 128)

    # 2️⃣ Add channel dimension → (128, 128, 1)
    arr = np.expand_dims(arr, axis=-1)

    # 3️⃣ Adjust based on model input
    if model is not None:
        input_shape = model.input_shape

        # If model expects 3 channels
        if input_shape[-1] == 3:
            arr = np.repeat(arr, 3, axis=-1)  # (128, 128, 3)

        # If model expects sequence input (CNN-LSTM)
        if len(input_shape) == 5:
            time_steps = input_shape[1] or 5
            # repeat the same image across the time dimension
            arr = np.repeat(arr[np.newaxis, ...], time_steps, axis=0)  # (time_steps, 128, 128, 3)
            arr = np.expand_dims(arr, axis=0)  # (1, time_steps, 128, 128, 3)
            return arr

    # 4️⃣ For simple 2D CNNs → (1, 128, 128, C)
    arr = np.expand_dims(arr, axis=0)
    return arr




def find_existing_script(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def stream_subprocess(cmd, placeholder):
    """Stream output of subprocess (used for training/evaluation)."""
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    try:
        for line in p.stdout:
            placeholder.text(line)
        p.wait()
        return p.returncode
    except Exception as e:
        p.kill()
        raise e


# ---------- Layout ----------
left, right = st.columns([2, 1])

# ---------- Right column: Model + Training ----------
with right:
    st.header("Model & Actions")
    model, model_path = load_trained_model()
    if model is None:
        st.warning("⚠ No trained model found in models/. You can still run training below.")
    else:
        st.write(f"*Model loaded from:* {model_path}")
        # st.write(f"*Model input shape:* {getattr(model, 'input_shape', 'unknown')}")

    st.markdown("---")
    st.subheader("Actions")

    # Training
    train_script = find_existing_script(TRAIN_SCRIPT_CANDIDATES)
    if train_script:
        if st.button("🚀 Start training (run train.py)"):
            st.info(f"Starting training using {train_script}. Logs will appear below.")
            log_box = st.empty()
            cmd = [sys.executable, train_script]
            with st.spinner("Training started... this may take some time ⏳"):
                try:
                    returncode = stream_subprocess(cmd, log_box)
                    if returncode == 0:
                        st.success("✅ Training completed successfully.")
                    else:
                        st.error(f"❌ Training exited with code {returncode}. Check logs.")
                except Exception as e:
                    st.error(f"Error during training: {e}")
    else:
        st.info("No train.py found in src/ directory.")

    # Evaluation
    eval_script = find_existing_script(EVAL_SCRIPT_CANDIDATES)
    if eval_script:
        if st.button("📊 Run evaluation (run evaluate.py)"):
            st.info(f"Running {eval_script} — this will generate evaluation results.")
            log_box = st.empty()
            cmd = [sys.executable, eval_script]
            with st.spinner("Running evaluation..."):
                try:
                    returncode = stream_subprocess(cmd, log_box)
                    if returncode == 0:
                        st.success("✅ Evaluation completed successfully.")
                    else:
                        st.error(f"❌ Evaluation exited with code {returncode}.")
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
    else:
        st.info("No evaluate.py found in src/ directory.")

    st.markdown("---")
    st.caption("💡 Tip: Copy your trained .h5 model into models/ for instant prediction access.")

# ---------- Left column: Image Upload + Prediction ----------
with left:
    st.header("Upload & Predict")
    uploaded_file = st.file_uploader(
        "Upload MRI image (.jpg/.png/.jpeg)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        col1, col2 = st.columns([1, 1])

        # Predict
        with col1:
            if st.button("🔬 Predict using model"):
                if model is None:
                    st.error("No model loaded. Please place a trained model in models/ or train one.")
                else:
                    st.info("Preprocessing image...")
                    img_input = preprocess_image_2d(image, model=model)
                    st.write(f"Processed input shape: {img_input.shape}")
                    with st.spinner("Running inference..."):
                        try:
                            pred = model.predict(img_input)
                            class_idx = int(np.argmax(pred))
                            confidence = float(np.max(pred)) * 100
                            st.success(f"🧠 Predicted Class: *{CLASS_NAMES[class_idx]}* ({confidence:.2f}% confidence)")
                            st.write("Raw probabilities:", np.round(pred[0], 4).tolist())
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

        # Save image
        with col2:
            if st.button("💾 Save to dataset"):
                save_dir = st.text_input("Save folder (e.g., dataset/CN)", value="dataset/UNKNOWN")
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{int(time.time())}.jpg"
                out_path = os.path.join(save_dir, fname)
                image.save(out_path)
                st.success(f"✅ Image saved to {out_path} for later training use.")

# ---------- Display Confusion Matrix ----------
# cm_path = os.path.join("models", "confusion_matrix.png")
# if os.path.exists(cm_path):
#     st.header("📈 Confusion Matrix (Latest Evaluation)")
#     st.image(cm_path, use_column_width=True)