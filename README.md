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

- **Stock Data**:`yfinance`
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

---

## 🧠 Best Performing Model

While multiple machine learning models were trained and evaluated, **the stacked ensemble model consistently achieved the most balanced and accurate results** across different evaluation metrics.

> ⚠️ **Note**: Since stock market data and news sentiment are time-sensitive and change frequently, **the performance metrics (accuracy, F1-score, ROC-AUC) can vary between runs**. Thus, static performance tables or visualizations are not emphasized in this project.

Instead, this project focuses on providing a **flexible, reusable, and modular pipeline** that adapts to updated data for real-time or retrained forecasting.

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


