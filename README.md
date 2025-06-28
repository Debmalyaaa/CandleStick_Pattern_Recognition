# 📈 Candlestick Pattern Recognition with Sentiment-Aware Trading Signal Prediction

## 📘 Project Description

This project aims to predict short-term trading signals by recognizing candlestick patterns and combining them with sentiment analysis from financial news. It leverages classical machine learning models trained on engineered features including price data and textual sentiment scores.

---

## 🧠 Key Objectives

- Detect various candlestick patterns from OHLC data.
- Use FinBERT, TextBlob, and VADER to generate sentiment scores from financial news.
- Build models to classify the market signal as **Buy**, **Sell**, or **Hold**.
- Compare performance across multiple ML classifiers.
- Improve model generalizability using SMOTETomek oversampling.

---

## 🗃️ Datasets Used

- **Stock Data**: Tesla (TSLA) intraday OHLC data via `yfinance`.
- **News Articles**: Queried using `NewsAPI` for sentiment analysis.
- **Sentiment Extractors**:
  - `FinBERT` (HuggingFace model)
  - `VADER` (Lexicon-based)
  - `TextBlob` (Naive Bayes-based)

---

## 📋 Features Extracted

- **Price Features**: Open, High, Low, Close
- **Sentiment Features**: Scores from FinBERT, VADER, and TextBlob
- **Candlestick Pattern Labels**: 70+ patterns categorized as Bullish/Bearish/Neutral
- **Final Signal**: Buy, Sell, or Hold (based on pattern + sentiment rules)

---

## 🔁 Signal Adjustment Logic

| Pattern Signal | Sentiment Polarity | Adjusted Signal |
|----------------|--------------------|-----------------|
| Buy            | Strong Negative    | Hold            |
| Sell           | Strong Positive    | Hold            |
| Hold           | Strong Positive    | Buy             |
| Hold           | Strong Negative    | Sell            |

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution & boxplots of OHLC values
- Skewness measured to understand bias
- Barplot of candlestick pattern distribution

---

## ⚙️ Preprocessing Steps

- Dropped duplicates and invalid 'False' entries
- Null value removal
- Label Encoding on signal
- Feature Scaling with `StandardScaler`
- Class Balancing with `SMOTETomek`

---

## 🏋️‍♂️ Model Training & Evaluation

| Model                        | Accuracy | F1 Score | Train Acc | Test Acc |
|-----------------------------|----------|----------|-----------|----------|
| Random Forest               | 0.75     | 0.7316   | 0.9964    | 0.7460   |
| XGBoost                     | 0.73     | 0.7195   | 0.9445    | 0.7317   |
| Gaussian Naive Bayes        | 0.33     | 0.2590   | 0.3799    | 0.3309   |
| Support Vector Classifier   | 0.66     | 0.5625   | 0.6957    | 0.6601   |
| Stochastic Gradient Descent | 0.67     | 0.5435   | 0.7101    | 0.6655   |
| Logistic Regression         | 0.72     | 0.6971   | 0.7579    | 0.7209   |
| Decision Tree               | 0.43     | 0.3772   | 0.5092    | 0.4347   |
| **Stacked Model** (RFR meta)| **0.75** | **0.7473**| 0.8492    | **0.7549** |


---

## 📊 Visualization

- **Confusion Matrices** for all models
- **ROC Curves** (multi-class) for each model
- **Bar Charts** comparing model F1 and Accuracy Scores

---

## 🔍 Hyperparameter Tuning

- Grid/RandomizedSearchCV used for:
  - `Random Forest`
  - `Naive Bayes`
  - `SVC`
  - `SGD`
  - `Logistic Regression`
  - `Decision Tree`

---

## 🛠️ Tools & Libraries

- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn`, `XGBoost`, `imbalanced-learn`
- `HuggingFace Transformers` (for FinBERT)
- `TextBlob`, `VADER` for sentiment scoring

---

## 💡 Potential Applications

- Smart trading bots using pattern + sentiment cues
- Educational platform for candlestick pattern detection
- Visualization dashboard for financial signal intelligence



---

## 🧪 Setup & Installation

```bash
pip install -r requirements.txt
