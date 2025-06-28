# 📈 Candlestick Pattern Recognition with Sentiment-Aware Trading Signal Prediction

## 📘 Project Description

This project combines candlestick pattern recognition with sentiment analysis to predict stock trading signals (Buy, Sell, Hold). It uses historical OHLC data and financial news to train machine learning models for signal classification.

---

## 🎯 Objectives

- Detect candlestick patterns from OHLC data.
- Analyze financial news using FinBERT, VADER, and TextBlob.
- Generate trading signals based on pattern and sentiment logic.
- Train, tune, and evaluate ML classifiers.
- Visualize performance through confusion matrices and metrics.

---

## 📁 Dataset Sources

- **Stock Data**: TSLA (Tesla) OHLC via `yfinance`
- **News Sentiment**: Queried via NewsAPI and scored using:
  - FinBERT (transformers)
  - VADER
  - TextBlob

---

## 📌 Features

- Open, High, Low, Close
- sentimentfinbert, sentimentvader, sentimenttextblob
- Signal label (Buy/Sell/Hold)

---

## 🔁 Signal Adjustment Rules

| Candlestick Signal | Sentiment | Final Signal |
|--------------------|-----------|--------------|
| Buy                | Negative  | Hold         |
| Sell               | Positive  | Hold         |
| Hold               | Positive  | Buy          |
| Hold               | Negative  | Sell         |

---

## 🧹 Preprocessing

- Removed duplicates, NaNs
- Categorical encoding (LabelEncoder)
- Scaling (StandardScaler)
- Class balancing using SMOTETomek

---

## 🤖 Models Trained

- Random Forest
- XGBoost
- Gaussian Naive Bayes
- Support Vector Classifier
- Stochastic Gradient Descent
- Logistic Regression
- Decision Tree
- **Stacked Ensemble (Meta-model: Random Forest)**

> ❌ KNN was tested but **excluded** from the final comparison.

---

## ✅ Model Results Summary

| Model                        | Accuracy | F1 Score | Train Accuracy | Test Accuracy |
|-----------------------------|----------|----------|----------------|---------------|
| Random Forest               | 0.75     | 0.7316   | 0.9964         | 0.7460        |
| XGBoost                     | 0.73     | 0.7195   | 0.9445         | 0.7317        |
| Gaussian Naive Bayes        | 0.33     | 0.2590   | 0.3799         | 0.3309        |
| Support Vector Classifier   | 0.66     | 0.5625   | 0.6957         | 0.6601        |
| Stochastic Gradient Descent | 0.67     | 0.5435   | 0.7101         | 0.6655        |
| Logistic Regression         | 0.72     | 0.6971   | 0.7579         | 0.7209        |
| Decision Tree               | 0.43     | 0.3772   | 0.5092         | 0.4347        |
| **Stacked Model**           | **0.75** | **0.7473**| 0.8492         | **0.7549**    |

---

## 📊 Visualizations

### 🔻 Confusion Matrix (Stacked Model)
![Confusion Matrix - Stacked Model](images/confusion_matrix_stacked.png)

---

### 📈 F1 Score Comparison
![F1 Score Comparison](images/f1_score_comparison.png)

---

### 📉 Accuracy Score Comparison
![Accuracy Score Comparison](images/accuracy_score_comparison.png)

---

## 🧪 Evaluation Metrics Used

- Accuracy
- F1 Score (Macro)
- ROC-AUC (Multiclass)
- Confusion Matrix

---

## 🔧 Hyperparameter Tuning

Tuned with `RandomizedSearchCV` for:
- Random Forest
- SVC
- GaussianNB
- Logistic Regression
- SGD
- Decision Tree

---

## 🛠️ Tools Used

- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn`, `XGBoost`, `imbalanced-learn`
- `HuggingFace Transformers`, `TextBlob`, `VADER`

---


