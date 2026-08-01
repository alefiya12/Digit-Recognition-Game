# 🎮 Digit Recognition Game

An interactive **Machine Learning** application that recognizes handwritten digits drawn on a digital canvas. Built using **TensorFlow/Keras** and **Streamlit**, the project supports both **Single Digit Recognition** and **Multi-Digit Recognition** with confidence scores and prediction probability visualization.

---

## 📌 Features

* ✏️ Draw handwritten digits on an interactive canvas
* 🔢 Single Digit Recognition (0–9)
* 🔢🔢 Multi-Digit Recognition (e.g., 25, 7106)
* 🎯 Confidence Score for predictions
* 📊 Prediction Probability Graph
* 🧠 CNN model trained on the MNIST dataset
* 🖥️ Modern and interactive Streamlit interface

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frontend/UI:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Image Processing:** Pillow
* **Data Processing:** NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```text
digit-recognition-game/
│
├── app.py                 # Streamlit application
├── train_model.py         # CNN model training script
├── model.h5               # Trained model
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── venv/                  # Virtual environment (optional)
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Digit-Recognition-Game.git
cd Digit-Recognition-Game
```

### 2. Create a Virtual Environment

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit tensorflow numpy pillow matplotlib opencv-python seaborn scikit-learn streamlit-drawable-canvas
```

---

## 🧠 Train the Model

Run the following command to train the CNN model and generate the trained model file:

```bash
python train_model.py
```

This will:

* Load the MNIST dataset
* Train the CNN model
* Display training and validation accuracy graphs
* Display training and validation loss graphs
* Generate a confusion matrix
* Save the trained model as `model.h5`

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL displayed in your terminal (typically `http://localhost:8501`) in your web browser.

---

## 🧠 Model Architecture

```text
Input Image (28×28×1)
        │
        ▼
Conv2D (32 Filters)
        │
        ▼
MaxPooling2D
        │
        ▼
Conv2D (64 Filters)
        │
        ▼
MaxPooling2D
        │
        ▼
Flatten
        │
        ▼
Dense (64 Neurons)
        │
        ▼
Dense (10 Neurons - Softmax)
```

---

## ⚙️ How It Works

### Single Digit Recognition

```text
User Draws Digit
        │
        ▼
Canvas Image
        │
        ▼
Image Preprocessing
        │
        ▼
CNN Model
        │
        ▼
Digit Prediction
        │
        ▼
Confidence Score + Probability Graph
```

### Multi-Digit Recognition

```text
User Draws Number
        │
        ▼
Canvas Image
        │
        ▼
Image Thresholding
        │
        ▼
Contour Detection (OpenCV)
        │
        ▼
Digit Segmentation
        │
        ▼
CNN Prediction (Each Digit)
        │
        ▼
Combine Predictions
        │
        ▼
Final Number
```

---

## 📊 Results

* Test Accuracy: Approximately **98–99%** on the MNIST test dataset
* Supports interactive handwritten digit prediction
* Displays confidence scores for predictions
* Visualizes prediction probabilities using bar charts
* Includes confusion matrix and training graphs for model evaluation

---

## 📚 References

* TensorFlow Documentation – https://www.tensorflow.org
* Keras Documentation – https://keras.io
* Streamlit Documentation – https://docs.streamlit.io
* OpenCV Documentation – https://docs.opencv.org
* MNIST Dataset – http://yann.lecun.com/exdb/mnist/
* Deep Learning (Ian Goodfellow, Yoshua Bengio, Aaron Courville) – https://www.deeplearningbook.org

---

## 👩‍💻 Author

**Alefiya Mithiborwala**

---
