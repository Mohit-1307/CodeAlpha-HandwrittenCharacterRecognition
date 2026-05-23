<div align="center">

# ✍️ Handwritten Character Recognition using CNN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit"/>
</p>

<p align="center">
  Deep Learning based handwritten digit recognition system using Convolutional Neural Networks (CNNs).
</p>

---

</div>

# 📌 Project Overview

This project implements a **Handwritten Character Recognition System** using:

- 🧠 Convolutional Neural Networks (CNN)
- 👁️ Computer Vision with OpenCV
- 📊 Deep Learning using TensorFlow/Keras
- 🌐 Streamlit Web Application

The model is trained on the **MNIST handwritten digit dataset** and can predict handwritten digits from uploaded images.

---

# 🚀 Features

✅ CNN-based digit recognition  
✅ Image preprocessing pipeline  
✅ Noise reduction using Gaussian Blur  
✅ Adaptive thresholding  
✅ Contour detection and cropping  
✅ Real-time digit prediction  
✅ Confidence score display  
✅ Streamlit interactive UI  
✅ High accuracy (~99%)  

---

# 🧠 Deep Learning Architecture

```text
Input Image (28x28)
        ↓
Convolution Layer
        ↓
ReLU Activation
        ↓
Max Pooling
        ↓
Convolution Layer
        ↓
ReLU Activation
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer
        ↓
Softmax Output
```

---

# 📂 Project Structure

```bash
Handwritten Character Recognition/
│
├── dataset/
├── venv/
│
├── train.py
├── predict.py
├── app.py
├── model.keras
├── digit.png
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core Programming |
| TensorFlow / Keras | Deep Learning |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Streamlit | Web Application |

---

# 📊 Dataset

## MNIST Dataset

The project uses the famous **MNIST handwritten digit dataset**.

### Dataset Details

| Property | Value |
|---|---|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28×28 |
| Classes | 10 Digits (0–9) |

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Mohit-1307/CodeAlpha-HandwrittenCharacterRecognition
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd CodeAlpha-HandwrittenCharacterRecognition
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python train.py
```

### Output

- Trained CNN model
- Accuracy graph
- Saved model file

```text
model.keras
```

---

# 🔍 Run Prediction

Place a handwritten image:

```text
digit.png
```

inside project folder.

Run:

```bash
python predict.py
```

---

# 🌐 Run Streamlit Web App

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

# 🖼️ Image Preprocessing Pipeline

```text
Input Image
      ↓
Grayscale Conversion
      ↓
Gaussian Blur
      ↓
Adaptive Thresholding
      ↓
Morphological Dilation
      ↓
Contour Detection
      ↓
Digit Cropping
      ↓
Resize to 28×28
      ↓
Normalization
      ↓
CNN Prediction
```

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | ~98% |
| Validation Accuracy | ~99% |
| Test Accuracy | ~99% |

---

# 🧮 Mathematical Concepts

## Convolution Operation

```math
(I * K)(x,y)=\sum_m\sum_n I(m,n)K(x-m,y-n)
```

---

## ReLU Activation

```math
ReLU(x)=max(0,x)
```

---

## Softmax Function

```math
P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
```

---

## Cross Entropy Loss

```math
L=-\sum_i y_i\log(\hat{y_i})
```

---

# 📸 Screenshots

## Training Accuracy Graph

_Add your screenshot here_

---

## Prediction Output

_Add your screenshot here_

---

# 🔥 Future Improvements

- 🔠 Alphabet Recognition
- 📄 Full OCR System
- 🎥 Real-time Webcam Recognition
- 🌍 Multi-language Handwriting Support
- 🤖 Transformer-based OCR

---

# 🧪 Example Prediction

```text
Prediction Results
-------------------
Predicted Digit: 7
Confidence: 0.9981
```

---

# 📚 Learning Outcomes

Through this project, concepts learned include:

- Convolutional Neural Networks
- Image Processing
- Feature Extraction
- Computer Vision
- OCR Pipeline
- Deep Learning Deployment
- Streamlit Application Development

---

# 🤝 Contributing

Contributions are welcome!

Feel free to:
- Fork repository
- Create feature branch
- Submit pull requests

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

### Mohit

Machine Learning & AI Enthusiast 🚀

---

# ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!
