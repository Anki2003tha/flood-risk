# 🌧️🌍 Flood Risk Prediction Dashboard
### *AI-Powered Satellite Flood Detection • Grad-CAM Explainability • Interactive Streamlit UI*


---



## 📛 Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30-red)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-black)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)


---

# 🎯 Overview

The **Flood Risk Prediction Dashboard** is an intelligent system that analyzes satellite or uploaded images to estimate flood risk levels (Low / Medium / High).

It includes:

- 🛰️ Satellite Image Upload Prediction  
- 🌈 Grad-CAM Explainability  
- 🗺️ Interactive Folium Map  
- 🤖 Smart Chatbot for contextual explanations  
- ⚡ Heuristic fallback prediction when the model is missing  
- 🎛️ Beautiful Streamlit UI  

This makes it suited for **research**, **education**, **disaster management**, and **environmental analysis**.


---

# ✨ Features

### 🛰️ Flood Risk Prediction
Accepts real satellite images and outputs a risk score.

### 🌈 Grad-CAM Heatmaps  
Explains which areas of the image contributed to the prediction.

### 🗺️ Interactive Flood Map  
Click anywhere → synthetic sample → predicted flood risk.

### 🤖 Chatbot Assistant  
Understands previous predictions and answers accordingly.

### ⚡ Lightweight & Fast  
Runs locally on CPU.


---

# 🧱 System Architecture

You can download the architecture PNG from the generated diagram and use:

```md
![Architecture Diagram](architecture.png)
```
---
# 📂 Project Structure
```
project/
│── smart_dashboard.py
│── model.py
│── model_demo.h5 (optional)
│── demo_data/
│── utils/
│     ├── gradcam.py
│     ├── preprocessing.py
│     └── heuristics.py
│── requirements.txt
│── README.md
```
---
#⚙️ Installation
1️⃣ Create virtual environment
python -m venv .venv
```
.\.venv\Scripts\activate
```

2️⃣ Install dependencies
```
pip install -r requirements.txt
```

3️⃣ Run the dashboard
```
streamlit run smart_dashboard.py
```
# 🧠 Model Workflow

- Normalize & resize image

- Predict via CNN (model_demo.h5)

- If no model → use blue-channel heuristic

- Generate Grad-CAM (model mode)

- Display results in dashboard

- Chatbot responds using prediction history

