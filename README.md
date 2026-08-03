# 💵 Banknote Authentication using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![License](https://img.shields.io/badge/License-MIT-purple)

A Machine Learning project that classifies banknotes as **Genuine** or **Counterfeit** using statistical image features. The project compares multiple Machine Learning algorithms, evaluates their performance, and automatically selects the best-performing model.

---

# 📌 Table of Contents

- Project Overview
- Dataset
- Features Used
- Machine Learning Workflow
- Exploratory Data Analysis
- Models Compared
- Results
- Project Structure
- Installation
- How to Run
- Technologies Used
- Future Improvements
- Author

---

# 📖 Project Overview

Counterfeit currency is a major problem across the world. Detecting fake banknotes accurately is important for banks, businesses, and financial institutions.

This project uses Machine Learning classification algorithms to predict whether a banknote is **genuine** or **counterfeit** based on four statistical image features extracted from banknote images.

Instead of relying on a single algorithm, this project compares multiple Machine Learning models and automatically selects the best-performing one.

---

# 🎯 Objective

The objective of this project is to:

- Understand the complete Machine Learning workflow.
- Perform Exploratory Data Analysis (EDA).
- Compare different classification algorithms.
- Select the best-performing model.
- Save the trained model for future predictions.

---

# 📂 Dataset

**Dataset Name**

Banknote Authentication Dataset

**Source**

UCI Machine Learning Repository

The dataset contains **1372 banknote samples**.

Each sample contains four numerical features extracted from banknote images using image processing techniques.

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| Variance | Variance of Wavelet Transformed Image |
| Skewness | Skewness of Wavelet Transformed Image |
| Curtosis | Curtosis of Wavelet Transformed Image |
| Entropy | Entropy of Image |
| Class | 0 = Genuine, 1 = Counterfeit |

---

# 🧠 Machine Learning Workflow

```text
Dataset
    │
    ▼
Load Data
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Selection
    │
    ▼
Train-Test Split
    │
    ▼
Train Multiple ML Models
    │
    ▼
Compare Performance
    │
    ▼
Select Best Model
    │
    ▼
Save Trained Model
```

---

# 📊 Exploratory Data Analysis

The dataset was analyzed before training the models.

The following analyses were performed:

- Dataset Information
- Missing Value Detection
- Statistical Summary
- Class Distribution
- Histograms
- Correlation Heatmap
- Pair Plot

## Class Distribution

<p align="center">
<img src="screenshots/class_distribution.png" width="650">
</p>

---

## Feature Distribution

<p align="center">
<img src="screenshots/feature_distribution.png" width="700">
</p>

---

## Correlation Heatmap

<p align="center">
<img src="screenshots/heatmap.png" width="650">
</p>

---

# 🤖 Models Compared

The following Machine Learning algorithms were trained and evaluated.

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Each model was evaluated using:

- Accuracy
- Cross Validation
- Confusion Matrix
- Classification Report

---

# 📈 Model Comparison

| Model | Accuracy |
|---------|----------|
| Logistic Regression | 98.55% |
| Decision Tree | 98.18% |
| Random Forest | 98.91% |
| K-Nearest Neighbors (KNN) | **100.00%** |
| Support Vector Machine (SVM) | **100.00%** |

---

# 📉 Accuracy Comparison

<p align="center">
<img src="screenshots/accuracy_comparison.png" width="700">
</p>

---

# 📌 Best Performing Model

After comparing all algorithms,

**K-Nearest Neighbors (KNN)** achieved the highest accuracy on the test dataset.

The trained model was saved as

```
models/currency_model.pkl
```

for future use.

---

# 📊 Confusion Matrix

<p align="center">
<img src="screenshots/confusion_matrix.png" width="600">
</p>

The confusion matrix shows that the selected model correctly classified almost every banknote in the testing dataset.

---

# 📁 Project Structure

```
Banknote-Authentication-ML
│
├── dataset/
│   └── dataset.txt
│
├── models/
│   └── currency_model.pkl
│
├── screenshots/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── feature_distribution.png
│   └── heatmap.png
│
├── compare_models.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/banknote-authentication-ml.git
```

Move into the project folder.

```bash
cd banknote-authentication-ml
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train the Model

```bash
python train_model.py
```

---

## Compare All Models

```bash
python compare_models.py
```

This script will

- Train multiple Machine Learning models
- Compare their performance
- Display the ranking
- Save the best model
- Generate comparison graphs

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

---

# 📚 Learning Outcomes

Through this project I learned:

- Data preprocessing
- Exploratory Data Analysis
- Feature Selection
- Train-Test Splitting
- Supervised Machine Learning
- Model Evaluation
- Cross Validation
- Saving trained models using Joblib
- Comparing multiple Machine Learning algorithms

---

# 🚀 Future Improvements

Some possible improvements for this project are:

- Build an image-based banknote authentication system.
- Train a Convolutional Neural Network (CNN) on banknote images.
- Create a web application for predictions.
- Support multiple currencies.
- Deploy the model using Streamlit or Flask.

---

# 👨‍💻 Author

**Satvik**

If you found this project useful, feel free to ⭐ this repository.
