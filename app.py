
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload an image containing one handwritten digit "
    "from 0 to 9."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, width=300)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Threshold the image
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Make the background black
    border = np.concatenate([
        binary[0, :],
        binary[-1, :],
        binary[:, 0],
        binary[:, -1]
    ])

    if np.mean(border) > 127:
        binary = cv2.bitwise_not(binary)

    # Find foreground components
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(binary, 8)
    )

    if count <= 1:
        st.error("No digit detected. Please try another image.")
    else:
        # Select the largest component
        component = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        x, y, w, h, area = stats[component]

        if area < 5:
            st.error("The detected digit is too small.")
        else:
            mask = np.zeros_like(binary)
            mask[labels == component] = 255

            digit = mask[y:y+h, x:x+w]

            # Resize while preserving aspect ratio
            scale = 20.0 / max(h, w)
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))

            digit = cv2.resize(
                digit,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # Center the digit
            canvas = np.zeros((28, 28), dtype=np.uint8)

            start_x = (28 - new_w) // 2
            start_y = (28 - new_h) // 2

            canvas[
                start_y:start_y + new_h,
                start_x:start_x + new_w
            ] = digit

            st.subheader("Processed Image")
            st.image(
                canvas,
                width=180,
                clamp=True
            )

            # Prepare CNN input
            input_image = (
                canvas.astype("float32") / 255.0
            ).reshape(1, 28, 28, 1)

            # Predict
            with st.spinner("Recognizing digit..."):
                scores = model.predict(
                    input_image, verbose=0
                )[0]

            prediction = int(np.argmax(scores))
            confidence = float(scores[prediction]) * 100

            st.success(f"Predicted Digit: {prediction}")
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Prediction Scores")

            st.bar_chart({
                "Score": {
                    str(i): float(scores[i])
                    for i in range(10)
                }
            })

            st.caption(
                "Confidence is the model's score, "
                "not a guarantee of correctness."
            )
