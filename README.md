# 🐾 Animal Image Classifier using TensorFlow & Keras

This project is an **Animal Image Classification model** built using **TensorFlow** and **Keras**.  
It classifies animal images into multiple categories using **deep learning and transfer learning**.

The dataset is stored in the **`Animals/`** folder and test images are available in the **`Testing/`** folder.  
Simply clone the repository, install dependencies, and run the script.

---

## 🔥 Badges

![License](https://img.shields.io/github/license/Ritikraj11/ImageClassifier)
![Stars](https://img.shields.io/github/stars/Ritikraj11/ImageClassifier)
![Issues](https://img.shields.io/github/issues/Ritikraj11/ImageClassifier)
![Forks](https://img.shields.io/github/forks/Ritikraj11/ImageClassifier)
![Last Commit](https://img.shields.io/github/last-commit/Ritikraj11/ImageClassifier)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Made With](https://img.shields.io/badge/made%20with-Python-blue)

---

## 📂 Table of Contents
- [About](#about)
- [Uses](#uses)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## 📌 About

This project is a **machine learning–based animal image classifier** trained using **TensorFlow and Keras**.  
It uses **MobileNetV2 (transfer learning)** to efficiently classify animal images with good accuracy.

The model supports **multiple animal classes** and can be retrained easily with new data.

---

## 🌍 Uses

1. **Agriculture**
   - Detect animals entering farms
   - Prevent crop damage

2. **Zoos & Wildlife Sanctuaries**
   - Automatic animal identification
   - Monitoring animal behavior

3. **Education**
   - Learning tool for students
   - Animal recognition apps

4. **Research & Conservation**
   - Wildlife monitoring
   - Species identification

---

## ✨ Features

- 🚀 Multi-class animal image classification  
- 🧠 Uses pretrained **MobileNetV2**  
- 📊 Displays training & validation metrics  
- 📁 Clean and modular code  
- 🔧 Easy to modify and retrain  
- 💻 Runs on CPU (no GPU required)

---

## 🧰 Tech Stack

- **Language:** Python 3.10+
- **Framework:** TensorFlow, Keras
- **Libraries:** NumPy, Matplotlib, scikit-learn
- **Model:** MobileNetV2
- **Environment:** Virtualenv / Conda

---

## 📁 Dataset Structure

ImageClassifier/
│
├── Animals/
│ ├── antelope/
│ ├── bear/
│ ├── cat/
│ └── ...
│
├── Testing/
│ ├── test1.jpg
│ ├── test2.jpg
│
├── Tensor.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Ritikraj11/ImageClassifier.git

# Navigate to the project directory
cd ImageClassifier

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
