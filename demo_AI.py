# demo_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from gtts import gTTS
import tempfile

st.title("Gesture to Speech Demo (Prototype)")

# -----------------------------
# 1. Load dataset & train model
# -----------------------------
@st.cache_data
def load_and_train():
    train_df = pd.read_csv("Data/sign_mnist_train.csv")
    test_df = pd.read_csv("Data/sign_mnist_test.csv")

    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values

    # Normalize & reshape
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Build model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(25, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train briefly for demo
    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test))
    
    return model

model = load_and_train()

# -----------------------------
# 2. Mapping letters & words
# -----------------------------
label_map = {i: chr(65+i) for i in range(25)}       # A-Y
word_map = {"A": "Hi", "B": "Bye", "N": "Name"}    # map letters to demo words

# -----------------------------
# 3. Webcam input
# -----------------------------
st.write("Start your camera to test gestures.")
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not available")
        break

    # Show camera feed
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert to grayscale & resize to 28x28
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (28,28))
    small = small.reshape(1,28,28,1) / 255.0

    # Predict
    pred = np.argmax(model.predict(small))
    letter = label_map[pred]

    st.text(f"Predicted Letter: {letter}")
    
    if letter in word_map:
        word = word_map[letter]
        st.text(f"Mapped Word: {word}")

        # Convert to speech and play in browser
        tts = gTTS(word)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmpfile.name)
        st.audio(tmpfile.name, format='audio/mp3')

cap.release()
