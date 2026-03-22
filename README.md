# 🌿 Plant Disease Detection System

An end-to-end AI-powered web application that detects potato leaf diseases using a Convolutional Neural Network,
served via a REST API and a modern animated frontend.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)


---

## 🎯 Problem Statement
Farmers lose significant crop yield every year due to undetected plant diseases. Early detection can save up to 
40% of crop loss. This system provides an accessible, AI-powered tool to detect potato leaf diseases instantly 
from a photo.

---

## 🏗️ Architecture
```
Image Upload (Frontend)
        ↓
FastAPI REST API
        ↓
Image Preprocessing (PIL, NumPy)
        ↓
CNN Model (TensorFlow/Keras)
        ↓
Prediction + Confidence Score
        ↓
Result Display (Frontend)
```

---

## ✨ Features

- 🔍 Detects 3 conditions: **Early Blight**, **Late Blight**, **Healthy**
- 📊 Real-time **confidence scoring** with animated progress bar
- 🖼️ **Drag & drop** image upload with live preview
- ⚠️ **Unknown image rejection** — rejects non-leaf images below confidence threshold
- 🌐 **REST API** built with FastAPI
- 💊 **Disease descriptions** with cause and symptoms
- 🎨 Modern animated UI with glassmorphism design

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | TensorFlow, Keras, NumPy |
| Backend | FastAPI, Python, Uvicorn |
| Frontend | HTML5, CSS3, JavaScript |
| Image Processing | PIL, NumPy |
| Deployment | Google Cloud Run (WIP) |

---

## 📁 Project Structure
```
plant-disease-detection/
├── api/
│   ├── main.py          # FastAPI app + prediction endpoint
│   └── requirements.txt
├── frontend/
│   ├── index.html       # Animated frontend UI
│   └── FARM.jpg         # Background image
├── training/
│   └── training.ipynb   # Model training notebook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- TensorFlow 2.x
- FastAPI

### Installation
```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/plant-disease-detection.git
cd plant-disease-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r api/requirements.txt

# Run the server
cd api
uvicorn main:app --reload --port 8000
```

### Usage
1. Open `http://127.0.0.1:8000` in your browser
2. Upload a potato leaf image
3. Click **Analyse Plant**
4. View disease prediction with confidence score

---

## 📊 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 256x256 RGB images
- **Classes:** Early Blight, Late Blight, Healthy
- **Dataset:** PlantVillage Dataset

- **Data Augmentation:** ImageDataGenerator (rotation, flip, zoom, shift)[yet to be done]

---

## ⚠️ Current Limitations

- Trained only on potato leaves — other plants not supported
- Requires clear, close-up leaf images for accurate results
- Low confidence predictions (< 90%) are rejected as unknown
- No mobile app support yet

---

## 🔮 Future Improvements
- [ ]- **Data Augmentation:** ImageDataGenerator (rotation, flip, zoom, shift)
- [ ] Multi-crop support (Tomato, Corn, Wheat)
- [ ] Multilingual farmer chatbot with voice input/output
- [ ] React Native mobile app with live camera capture
- [ ] Retrain with "Unknown" class for better rejection
- [ ] Google Cloud Run deployment
- [ ] User authentication and prediction history

---
