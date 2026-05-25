<div align="center">

# Handwritten Character Recognition — CNN

[![Python](https://img.shields.io/badge/Python-3.12-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=flat-square)]()

**CNN-based handwritten digit recognition trained on MNIST with a real-time Streamlit inference interface.**

[Overview](#-overview) · [Architecture](#-cnn-architecture) · [Preprocessing](#-image-preprocessing-pipeline) · [Math](#-key-math) · [Setup](#-installation) · [Results](#-results)

</div>

---

## Overview

This project implements an end-to-end deep learning system for recognizing handwritten digits (0–9) from uploaded images.

The pipeline covers CNN model design, image preprocessing with OpenCV (grayscale conversion, Gaussian blur, adaptive thresholding, contour detection), model training on MNIST, and deployment as an interactive Streamlit web application.

| Capability | Details |
|---|---|
| Model | Custom CNN trained on MNIST |
| Input | Uploaded handwritten digit image |
| Preprocessing | Grayscale → Blur → Threshold → Contour → Crop → 28×28 → Normalize |
| Output | Predicted digit (0–9) + confidence score |
| Deployment | Streamlit web app (`localhost:8501`) |
| Test Accuracy | ~99% |

---

## CNN Architecture

```
Input Image (28×28×1)
        │
        ▼
Conv2D → ReLU            ← Feature map extraction (edges, curves)
        │
        ▼
MaxPooling2D             ← Spatial downsampling, reduce computation
        │
        ▼
Conv2D → ReLU            ← Higher-level feature extraction
        │
        ▼
MaxPooling2D
        │
        ▼
Flatten
        │
        ▼
Dense (128) → ReLU       ← Fully connected classification head
        │
        ▼
Dense (10) → Softmax     ← Class probability distribution (digits 0–9)
```

---

## Image Preprocessing Pipeline

Raw images from users are rarely clean 28×28 grayscale inputs. The preprocessing pipeline normalizes them before inference:

```
Input Image (any size, any channel)
        │
        ▼
Grayscale Conversion
        │
        ▼
Gaussian Blur             ← Noise reduction
        │
        ▼
Adaptive Thresholding     ← Binarization robust to lighting variation
        │
        ▼
Morphological Dilation    ← Fill gaps in digit strokes
        │
        ▼
Contour Detection         ← Isolate individual digit regions
        │
        ▼
Digit Cropping
        │
        ▼
Resize to 28×28           ← Match MNIST input dimensions
        │
        ▼
Normalization (÷255)      ← Scale pixel values to [0, 1]
        │
        ▼
CNN Inference
```

---

## Key Math

**Convolution** — extracts spatial features by sliding a learned kernel over the input:

$$( I * K)(x,y) = \sum_m \sum_n I(m,n) \cdot K(x-m,\ y-n)$$

**ReLU** — introduces non-linearity, kills negative activations:

$$\text{ReLU}(x) = \max(0,\ x)$$

**Softmax** — converts raw logits to a probability distribution over 10 classes:

$$P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Cross-Entropy Loss** — penalizes confident wrong predictions most heavily:

$$\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$$

---

## Dataset

**[MNIST Handwritten Digit Dataset](http://yann.lecun.com/exdb/mnist/)** — Yann LeCun et al.

| Property | Value |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28×28 grayscale |
| Classes | 10 (digits 0–9) |
| Source | `tensorflow.keras.datasets.mnist` |

---

## Project Structure

```
CodeAlpha-HandwrittenCharacterRecognition/
│
├── dataset/                   # MNIST data (auto-downloaded)
│
├── train.py                   # Model training + evaluation
├── predict.py                 # CLI inference on single image
├── app.py                     # Streamlit web app
├── model.keras                # Saved trained model
├── digit.png                  # Sample test image
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1 · Clone the Repository

```bash
git clone https://github.com/Mohit-1307/CodeAlpha-HandwrittenCharacterRecognition
cd CodeAlpha-HandwrittenCharacterRecognition
```

### 2 · Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model

```bash
python train.py
```

Trains the CNN on MNIST, evaluates on the test set, plots accuracy/loss curves, and saves the model to `model.keras`.

### CLI Prediction

Place your handwritten digit image as `digit.png` in the project root, then:

```bash
python predict.py
```

```text
Prediction Results
──────────────────
Predicted Digit : 7
Confidence      : 0.9981
```

### Streamlit Web App

```bash
streamlit run app.py
# → http://localhost:8501
```

Upload any handwritten digit image and get real-time prediction with confidence score.

---

## Results

| Metric | Score |
|---|---|
| Training Accuracy | ~98% |
| Validation Accuracy | ~99% |
| Test Accuracy | ~99% |

The CNN achieves near-human performance on MNIST due to the dataset's relatively constrained domain (single centered digits, uniform stroke style). Real-world generalization depends heavily on the preprocessing pipeline's ability to isolate and normalize the digit from varied inputs.

---

## Future Improvements

- [ ] Extend to full alphabet recognition (A–Z, a–z)
- [ ] Multi-digit detection in a single image
- [ ] Real-time webcam recognition
- [ ] Transformer-based OCR (TrOCR)
- [ ] Multi-language handwriting support
- [ ] FastAPI backend + Docker containerization

---

## Tech Stack

| Technology | Purpose |
|---|---|
| [Python 3.12](https://python.org) | Core application |
| [TensorFlow / Keras](https://tensorflow.org) | CNN model training & inference |
| [OpenCV](https://opencv.org) | Image preprocessing |
| [NumPy](https://numpy.org) | Numerical operations |
| [Matplotlib](https://matplotlib.org) | Training curve visualization |
| [Streamlit](https://streamlit.io) | Web interface |

---

## Author

**Mohit Singh Rajput** — AI / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghdausa@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>