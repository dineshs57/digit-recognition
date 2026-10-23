
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Keep CPU training simple and reproducible
tf.keras.utils.set_random_seed(42)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu", padding="same"
    ),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the CNN
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate on unseen test data
loss, accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# Save trained model
model.save("digit_model.keras")
print("Saved model as digit_model.keras")

# Plot training and validation accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.show()

# Confusion matrix
predictions = model.predict(X_test, batch_size=256, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)
display.plot(cmap="Blues", values_format="d")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
