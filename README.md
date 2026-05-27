# 🚨 Real-Time Financial Fraud Detection System

**Author:** Manjunath Mohith
**Deployment:** Streamlit Web Application  
**Core Technologies:** Python, Pandas, XGBoost, Scikit-Learn  

## 📊 Executive Summary
For early-stage SaaS and fintech startups, digital fraud—such as stolen credit card usage and synthetic identity attacks—poses a critical threat to survival. The core business challenge is the **False Positive vs. False Negative tradeoff**: blocking too many legitimate transactions ruins the user experience and stifles growth, while allowing fraudulent transactions drains vital capital. 

This project implements an end-to-end machine learning pipeline using **XGBoost** to predict fraudulent transactions in real-time. By applying algorithmic class-weight penalization to highly imbalanced financial logs, the model successfully catches **87% of fraud** while maintaining an **87% precision rate**, minimizing friction for legitimate customers.

---

## 📂 Repository Structure
```text
├── data/
│   ├──cleaned_dataset_4_training.csv    # Preprocessed data used for model training
│   └── paysim_sample.csv                # raw data, a random sample data from(https://www.kaggle.com/datasets/ealaxi/paysim1)
├── deployment/ 
│   ├── app.py                           # Streamlit frontend application script
│   ├── requirements.txt                 # Dependencies for deployment
│   └── xgboost_fraud_model.joblib       # Serialized XGBoost model
├── reports/
│   └── capstone_project.pdf             # Exported technical documentation
├── presentation/
│   └── capstone_presentation.pdf        # Business slide deck
├── README.md                            # Project overview
└── capstone_project.ipynb               # Main exploratory data analysis and modeling notebook
```

---

## 📈 The Dataset & Exploratory Data Analysis

This project utilizes the **PaySim Synthetic Dataset**, which accurately simulates 30 days of mobile money transaction logs.

**Initial Imbalance:** The raw dataset contains 318,131 transactions with a massive class imbalance common in fraud detection:

| Class | Count | Percentage |
|---|---|---|
| Legitimate | 317,735 | 99.87% |
| Fraudulent | 396 | 0.12% |

### Key EDA Insights

1. **Fraud Isolation:** 100% of the fraudulent activity occurred exclusively within two transaction types: `CASH_OUT` and `TRANSFER`. The dataset was filtered to only include these types to optimize model training.
2. **Financial Impact:** The average legitimate transaction was ~$177k, while the average fraudulent transaction was significantly higher at ~$1.47M, capping at exactly $10M.

---

## 🛠️ Methodology & Machine Learning Strategy

### 1. Data Preprocessing
- Filtered unneeded transaction types (`PAYMENT`, `DEBIT`, `CASH_IN`).
- Dropped non-predictive string identifiers (`nameOrig`, `nameDest`).
- Binary encoded the transaction `type` (`CASH_OUT`: 0, `TRANSFER`: 1).
- Applied a **Stratified Train-Test Split (80/20)** to ensure the 0.12% fraud ratio remained perfectly balanced across both sets.

### 2. Model Selection & Tuning
- **Algorithm:** `XGBoost Classifier`
- **Handling Imbalance:** To prevent the *Accuracy Paradox* (where a model blindly predicts "Not Fraud" for 99% accuracy), the `scale_pos_weight` hyperparameter was mathematically calculated to **348.43**. This forced the algorithm to penalize missed fraud cases 348 times harder than missed normal transactions.
- **Evaluation Metric:** Area Under the Precision-Recall Curve (`aucpr`).

---

## 🚀 Model Performance & Business Impact

Because "Accuracy" is a misleading metric for highly imbalanced data, the model's success was strictly evaluated using **F1-Score**, **Precision**, and **Recall**.

### Final Test Set Results
*(79 Actual Fraud Cases out of 27,692 total)*

| Metric | Score |
|---|---|
| F1-Score | 0.87 |
| Recall (True Positive Rate) | 0.87 |
| Precision | 0.87 |

### The Confusion Matrix Translated

| Result | Count | Meaning |
|---|---|---|
| ✅ True Positives | 69 | 69 out of 79 fraudsters were successfully caught and blocked. |
| ⚠️ False Positives | 10 | Out of 27,613 legitimate transactions, only 10 innocent customers were falsely flagged. |
| ❌ False Negatives | 10 | Only 10 fraudulent transactions slipped through the system. |

---

## 💻 How to Run the Application Locally

The project includes a fully functional, real-time web dashboard built with **Streamlit**.

**1. Clone the repository and navigate to the project directory:**
```bash
git clone https://github.com/Mohith-151/Week-12.git
cd Week-12
```

**2. Install the required dependencies:**
```bash
pip install -r deployment/requirements.txt
```

**3. Launch the Streamlit application:**
```bash
streamlit run deployment/app.py
```
```
*The web interface will automatically open in your default browser at `localhost:8501`. Enter transaction details to receive an immediate **Risk Score** (Safe vs. High-Risk).*
```