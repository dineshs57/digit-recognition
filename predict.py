
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Select an image
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a handwritten digit",
    filetypes=[
        ("Image files", "*.png *.jpg *.jpeg *.bmp")
    ]
)

root.destroy()

if not file_path:
    print("No image selected.")
    raise SystemExit

# Read image
img = cv2.imread(file_path)

if img is None:
    raise ValueError("Could not read the selected image.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold image
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Detect foreground polarity using image borders
border = np.concatenate([
    binary[0, :],
    binary[-1, :],
    binary[:, 0],
    binary[:, -1]
])

# Make background black and digit white
if np.mean(border) > 127:
    binary = cv2.bitwise_not(binary)

# Find connected components
count, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(binary, 8)
)

if count <= 1:
    raise ValueError("No digit detected.")

# Select the largest foreground component
component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
x, y, w, h, area = stats[component]

if area < 5:
    raise ValueError("Detected foreground is too small.")

# Create a mask for the selected component
mask = np.zeros_like(binary)
mask[labels == component] = 255

# Crop digit
digit = mask[y:y+h, x:x+w]

# Resize while preserving aspect ratio
scale = 20.0 / max(h, w)
new_w = max(1, round(w * scale))
new_h = max(1, round(h * scale))

digit = cv2.resize(
    digit, (new_w, new_h),
    interpolation=cv2.INTER_AREA
)

# Center on a 28x28 canvas
canvas = np.zeros((28, 28), dtype=np.uint8)

start_x = (28 - new_w) // 2
start_y = (28 - new_h) // 2

canvas[
    start_y:start_y + new_h,
    start_x:start_x + new_w
] = digit

# Convert to MNIST-style input
input_image = canvas.astype("float32") / 255.0
input_image = input_image.reshape(1, 28, 28, 1)

# Predict
scores = model.predict(input_image, verbose=0)[0]
prediction = int(np.argmax(scores))
confidence = float(scores[prediction]) * 100

print("Predicted digit:", prediction)
print(f"Model confidence: {confidence:.2f}%")

# Display the processed image and prediction scores
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(canvas, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Processed input: {prediction}")
axes[0].axis("off")

axes[1].bar(range(10), scores)
axes[1].set_xticks(range(10))
axes[1].set_xlabel("Digit")
axes[1].set_ylabel("Prediction score")
axes[1].set_title("CNN prediction scores")

plt.tight_layout()
plt.show()
