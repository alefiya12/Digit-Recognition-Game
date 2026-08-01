import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit Recognition Game",
    page_icon="🎮",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model.h5")

# ---------------- SIDEBAR MODE SELECT ----------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Single Digit Recognition", "Multi Digit Recognition"]
)

# ---------------- HEADER ----------------
st.title("🎮 AI Digit Recognition Game")

st.write("Draw digits and let the AI guess them!")

st.divider()

# =====================================================
# SINGLE DIGIT MODE
# =====================================================

if mode == "Single Digit Recognition":

    st.subheader("✏ Draw a Single Digit (0-9)")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="single_canvas"
    )

    col1, col2 = st.columns(2)

    predict = col1.button("🔍 Predict Digit")
    clear = col2.button("🧹 Clear Canvas")

    if clear:
        st.rerun()

    if predict and canvas.image_data is not None:

        img = canvas.image_data

        img = Image.fromarray((img[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))

        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"🤖 AI Prediction: {digit}")
        st.info(f"🎯 Confidence Score: {confidence:.2f}%")

        # Probability graph
        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots()

        ax.bar(range(10), prediction[0])

        ax.set_xlabel("Digits")
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")
        ax.set_xticks(range(10))

        st.pyplot(fig)

# =====================================================
# MULTI DIGIT MODE
# =====================================================

elif mode == "Multi Digit Recognition":

    st.subheader("✏ Draw Multiple Digits (Example: 25, 7106)")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="multi_canvas"
    )

    predict = st.button("🔍 Predict Number")

    if predict and canvas.image_data is not None:

        img = canvas.image_data.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 10 and h > 10:
                boxes.append((x, y, w, h))

        boxes = sorted(boxes, key=lambda b: b[0])

        predictions = []
        confidences = []
        probabilities_list = []

        for box in boxes:

            x, y, w, h = box

            digit = thresh[y:y+h, x:x+w]

            # Padding
            digit = cv2.copyMakeBorder(
                digit, 10,10,10,10,
                cv2.BORDER_CONSTANT,
                value=0
            )

            digit = cv2.resize(digit, (28,28))
            digit = digit / 255.0
            digit = digit.reshape(1,28,28,1)

            pred = model.predict(digit)

            digit_pred = np.argmax(pred)
            confidence = np.max(pred) * 100

            predictions.append(str(digit_pred))
            confidences.append(confidence)
            probabilities_list.append(pred[0])

        # Final number
        number = "".join(predictions)

        # Average confidence
        avg_conf = sum(confidences) / len(confidences)

        st.success(f"🤖 AI Prediction: {number}")
        st.info(f"🎯 Average Confidence Score: {avg_conf:.2f}%")

        # -------- Per Digit Confidence --------
        st.subheader("🔍 Per Digit Confidence")

        for i, conf in enumerate(confidences):
            st.write(f"Digit {predictions[i]} → {conf:.2f}%")

        # -------- Probability Graphs --------
        st.subheader("📊 Prediction Probability (Per Digit)")

        for i, probs in enumerate(probabilities_list):

            st.write(f"Digit {i+1} ({predictions[i]})")

            fig, ax = plt.subplots()

            ax.bar(range(10), probs)
            ax.set_xlabel("Digits")
            ax.set_ylabel("Probability")
            ax.set_title(f"Digit {predictions[i]} Confidence")
            ax.set_xticks(range(10))

            st.pyplot(fig)