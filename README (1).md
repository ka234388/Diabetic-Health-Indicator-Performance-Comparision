# Diabetic Health Indicator System (CAP6545 Final Project)

This repository contains a machine learning project aimed at predicting diabetes status (Non-Diabetic, Pre-Diabetic, Diabetic) using structured health data from the BRFSS 2015 dataset. The project implements and compares three deep learning models:

- Multi-layer Perceptron (MLP)
- TabTransformer
- Deep Neural Decision Forest (DNDF)

---

## 📁 Dataset

- Source: [BRFSS 2015 Health Indicators Dataset](https://www.cdc.gov/brfss/)
- Preprocessing:
  - Train-test split
  - Class balancing using SMOTE
  - Standardization using StandardScaler

---

## 🧠 Models Implemented

### 1. MLP (Multi-layer Perceptron)
- Built using PyTorch
- Four hidden layers with ReLU and Dropout
- Optimizer: Adam
- Loss: CrossEntropyLoss with class weights
- Learning rate scheduler: ReduceLROnPlateau

### 2. TabTransformer
- Attention-based model built using PyTorch
- Embedding + Transformer Encoder + Classifier
- Label smoothing applied
- Optimizer: AdamW
- Dropout and weight decay for regularization

### 3. DNDF (Deep Neural Decision Forest)
- Implemented using TensorFlow/Keras
- Dense layers with LeakyReLU
- Softmax trees averaged as ensemble layer
- Optimizer: Adam
- Regularized with Dropout and EarlyStopping

---

## 📊 Evaluation Metrics

- Accuracy
- Precision / Recall / F1-score (per class)
- Confusion Matrix
- Training & Validation Loss / Accuracy plots

---

## 📈 Results Summary

| Model           | Val Accuracy | Macro F1 | Strengths                        |
|----------------|--------------|----------|----------------------------------|
| MLP            | 89.99%       | 0.90     | Best performance overall         |
| TabTransformer | 81.77%       | 0.43     | Good Non-Diabetic prediction     |
| DNDF           | 65.56%       | 0.42     | Stronger Diabetic recall         |

---

## 🔧 How to Run

```bash
# Clone this repo
git clone https://github.com/yourusername/diabetes-prediction-cap6545.git
cd diabetes-prediction-cap6545

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook cap6545-mlp-tabtransformerfinalproject.ipynb
```

---

## 📌 Authors

- [Your Name]
- CAP6545 - Deep Learning
- University of Central Florida

---

## 📜 License

This project is for academic use only.