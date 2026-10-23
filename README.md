# 🔢 Handwritten Digit Recognition

An interactive web application built with **Streamlit** and **TensorFlow / Keras** that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 🚀 Features

- **CNN Architecture**: Trained on the MNIST dataset with high accuracy.
- **Interactive UI**: Built with Streamlit for uploading digit images and viewing predictions in real-time.
- **Image Preprocessing**: Preprocesses uploaded images (grayscale, resizing to 28x28, normalization) using OpenCV.
- **Visual Analytics**: Includes training accuracy charts and confusion matrix analysis.

---

## 📁 Project Structure

```text
digit-recognition/
├── app.py                   # Streamlit web application
├── train.py                 # Script to train CNN on MNIST dataset
├── predict.py               # Inference helper script
├── digit_model.keras        # Trained Keras model file
├── confusion_matrix.png     # Evaluation confusion matrix plot
├── training_accuracy.png    # Training & validation accuracy plot
├── requirements.txt         # Required Python dependencies
├── .gitignore               # Files excluded from git tracking
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dineshs57/digit-recognition.git
   cd digit-recognition
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the App

To launch the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser at the displayed local URL (usually `http://localhost:8501`) and upload a handwritten digit image.

---

## 🧠 Model Training (Optional)

To retrain the model from scratch:

```bash
python train.py
```

This will save the trained model as `digit_model.keras` and generate `confusion_matrix.png` and `training_accuracy.png`.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
