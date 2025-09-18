# demo_AI.py
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from gtts import gTTS
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(layout="centered")
st.title("Gesture → Text → Speech (Prototype)")

# -----------------------------
# 1) Load dataset & quick train (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_and_train_demo_model():
    train_df = pd.read_csv("Data/sign_mnist_train.csv")
    test_df = pd.read_csv("Data/sign_mnist_test.csv")

    y_train = train_df["label"].values
    X_train = train_df.drop("label", axis=1).values
    y_test = test_df["label"].values
    X_test = test_df.drop("label", axis=1).values

    # Normalize + reshape
    X_train = (X_train / 255.0).reshape(-1, 28, 28, 1).astype("float32")
    X_test  = (X_test  / 255.0).reshape(-1, 28, 28, 1).astype("float32")

    # Small CNN for demo
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(25, activation='softmax')  # Sign-MNIST has 25 classes (no J/Z)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Train briefly (demo)
    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), verbose=0)
    return model

with st.spinner("Loading dataset and training demo model (this may take ~1-2 minutes)..."):
    model = load_and_train_demo_model()

# -----------------------------
# 2) Label & word mapping
# -----------------------------
label_map = {i: chr(65 + i) for i in range(25)}    # 0->'A', 1->'B', ... 'Y' (Sign-MNIST)
# Demo mapping: letter -> word (you can change these)
word_map = {"A": "Hi", "B": "Bye", "N": "Name"}

st.markdown("**How to test:** use `Capture image` (camera) or upload an image of a single sign (28×28 or similar).")
st.write("Note: this is a prototype using the Sign-MNIST dataset (letters). We map a few letters to demo words.")

# -----------------------------
# 3) Input: camera (browser) or upload
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    camera_img = st.camera_input("Capture image with your camera")  # works in browser (local + cloud)
with col2:
    upload = st.file_uploader("Or upload an image file", type=["png","jpg","jpeg"])

img_bytes = None
if camera_img is not None:
    img_bytes = camera_img.getvalue()
elif upload is not None:
    img_bytes = upload.getvalue()

if img_bytes:
    # Convert to PIL image, grayscale and resize to 28x28
    pil = Image.open(tf.io.BytesIO(img_bytes)).convert("L")  # L = grayscale
    pil = pil.resize((28,28))
    st.image(pil, caption="Input (resized to 28x28)", use_column_width=False)

    # Convert to array expected by model
    arr = np.array(pil).reshape(1,28,28,1).astype("float32") / 255.0

    # Prediction
    pred = model.predict(arr, verbose=0)
    pred_label = int(np.argmax(pred, axis=1)[0])
    letter = label_map[pred_label]

    st.success(f"Predicted letter: **{letter}**")

    if letter in word_map:
        mapped_word = word_map[letter]
        st.info(f"Mapped word: **{mapped_word}**")

        # Convert to speech and play in browser
        tts = gTTS(mapped_word)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        st.audio(tmp.name, format='audio/mp3')
    else:
        st.write("No demo mapping defined for this letter.")
else:
    st.write("No image yet. Capture or upload a sample image to test.")
