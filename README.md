# 📊 Twitter Sentiment Analysis  
### End-to-End NLP Pipeline using Machine Learning & Transformer Models

---

## 🚀 Project Overview

This project implements a complete end-to-end Natural Language Processing (NLP) pipeline for sentiment classification on Twitter data.

The system compares:

- Classical Machine Learning models  
- Deep Learning architectures  
- Transformer-based models (DistilBERT)

The objective is to evaluate performance differences and build a clean, modular, and reproducible NLP workflow similar to real-world ML projects.

---

## 🎯 Objectives

- Perform sentiment classification on Twitter text data  
- Compare ML, Deep Learning, and Transformer models  
- Fine-tune DistilBERT for contextual understanding  
- Evaluate models using proper classification metrics  
- Maintain structured and reusable code  

---

## 🧠 Models Implemented

### 🔹 Classical ML
- Logistic Regression (TF-IDF features)
- Random Forest
- XGBoost
- Naive Bayes
- Linear SVM 

### 🔹 Deep Learning

- CNN  
- LSTM  
- BiLSTM  

### 🔹 Transformer Model
- DistilBERT (Fine-tuned)

---

## 🔄 NLP Pipeline

1. Data Cleaning  
   - URL removal  
   - Special character removal  
   - Lowercasing  
   - Stopword removal  

2. Tokenization  
   - Custom tokenizer for ML models  
   - HuggingFace tokenizer for DistilBERT  

3. Feature Engineering  
   - TF-IDF vectorization  
   - Sequence padding  

4. Model Training  
   - Train-validation split  
   - Supervised learning  

5. Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix  

---

## 📈 Model Performance

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 79%      |
| CNN                  | 64%      |
| LSTM                 | 64%      |
| BiLSTM               | 71%      |
| DistilBERT           | **XX%**  |

> Replace XX% with your actual results before publishing.

---

## 🏗 Project Structure
twitter_analysis/
│
├── data/ # Raw and processed datasets
├── models/ # Saved trained models (ignored in Git)
├── notebooks/ # EDA and experimentation notebooks
├── scripts/ # Training and preprocessing scripts
│ ├── data_cleaning.py
│ ├── visualization.py
│ ├── train_logistic.py
│ ├── train_lstm.py
│ ├── train_cnn.py
│ └── train_distilbert.py
│
├── requirements.txt
├── setup.py
└── README.md


> Note: Large trained model files are excluded due to GitHub size limitations. Models can be regenerated using the training scripts.

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- PyTorch  
- HuggingFace Transformers  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  

---
## ▶️ How to Run
python notebooks/model_training.ipynb


---

## 📊 Exploratory Data Analysis

EDA includes:
- Sentiment distribution visualization  
- Text length analysis  
- Word frequency analysis  

Notebooks are available in the `/notebooks` directory.

---

## 🔮 Future Improvements

- Hyperparameter tuning  
- Model deployment using FastAPI  
- Docker containerization  
- Real-time inference API  

---

## 👤 Author

**Sameer Tripathi**  
Aspiring AI/ML Engineer | Data Science Enthusiast
