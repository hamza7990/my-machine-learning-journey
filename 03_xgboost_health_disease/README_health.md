# 🏥 Health Disease Prediction — XGBoost Multi-Label Classification

> **Project 03** of my Machine Learning Journey
>
> A complete end-to-end machine learning pipeline that predicts the presence of **10 diseases simultaneously** from a patient's health profile using **XGBoost — one model per disease**.

---

## 📁 Project Structure

```
📦 03_xgboost_health_disease/
├── 📓 health_disease_prediction_xgboost.ipynb   ← Main notebook
├── 📄 health_disease_pridiction.csv             ← Raw dataset
└── 📄 README.md                                 ← You are here
```

---

## 📊 Dataset Overview

The dataset contains **1,000 patient records** with health indicators and disease diagnoses.

### 🔵 Input Features

| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| `Age` | Numeric | 18 – 100 | Patient's age in years |
| `Gender` | Categorical | Male / Female | Biological sex of the patient |
| `Blood Pressure` | Categorical | Normal / High / Low | Blood pressure level |
| `Cholesterol` | Categorical | Normal / High / Low | Cholesterol level in blood |
| `Glucose` | Categorical | Normal / High / Low | Blood sugar level |
| `Smoking` | Categorical | Yes / No | Whether the patient smokes |
| `Alcohol Consumption` | Categorical | Yes / No | Whether the patient drinks alcohol |
| `Exercise` | Categorical | Yes / No | Whether the patient exercises regularly |
| `BMI` | Numeric | 18.5 – 50+ | Body Mass Index — weight relative to height |
| `Family History` | Categorical | Yes / No | Family history of diseases |

### 🔴 Target Variables — 10 Diseases

| Disease | Prevalence | Description |
|---------|-----------|-------------|
| `Heart Disease` | ~25% | Cardiovascular conditions affecting the heart |
| `Diabetes` | ~19% | Inability to regulate blood sugar levels |
| `Liver Disease` | ~16% | Liver dysfunction or damage |
| `Kidney Disease` | ~14% | Chronic or acute kidney dysfunction |
| `Alzheimer's Disease` | ~13% | Progressive brain disorder affecting memory |
| `Stroke` | ~13% | Interruption of blood supply to the brain |
| `COPD` | ~10% | Chronic Obstructive Pulmonary Disease |
| `Cancer` | ~10% | Abnormal cell growth in the body |
| `Parkinson's Disease` | ~8% | Nervous system disorder affecting movement |
| `Tuberculosis` | ~4% | Bacterial infection primarily affecting the lungs |

> ⚠️ **Multi-Label Problem** — a patient can have **more than one disease** at the same time

---

## 🔄 Pipeline Walkthrough

### 1️⃣ Exploratory Data Analysis (EDA)
- **Disease prevalence** bar chart across all 10 conditions
- **Patient demographics** — Age, Gender, BMI distributions
- **Risk factor distributions** — Blood Pressure, Cholesterol, Smoking, Exercise
- **Smoking impact** on each disease separately
- **Age & BMI** distribution by disease presence
- **Disease co-occurrence heatmap** — which diseases appear together

### 2️⃣ Preprocessing
- Check and remove duplicate rows
- **LabelEncoder** on all 8 categorical features
- No missing values in this dataset ✅

### 3️⃣ Model Strategy — Binary Relevance
Since this is a **Multi-Label** problem, we train one **XGBoost classifier per disease**:

```
Heart Disease    → XGBoost Model 1
Diabetes         → XGBoost Model 2
Stroke           → XGBoost Model 3
...              → ...
Tuberculosis     → XGBoost Model 10
```

Each model independently learns the relationship between features and its target disease.

---

## 🧠 Model — XGBoost

XGBoost (Extreme Gradient Boosting) builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous one.

### Hyperparameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 4 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `subsample` | 0.8 | Fraction of samples per tree |
| `colsample_bytree` | 0.8 | Fraction of features per tree |
| `eval_metric` | logloss | Evaluation during training |

### Model API

```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions per disease |
| **ROC-AUC** | Model's ability to separate sick vs healthy |
| **Confusion Matrix** | True/False Positives and Negatives |
| **ROC Curve** | Trade-off between sensitivity and specificity |
| **Feature Importance** | Which features matter most per disease |

---

## 🔍 Key Findings

- **`Age`**, **`BMI`**, and **`Family History`** are the strongest predictors across almost all diseases
- **Heart Disease** has the highest prevalence (~25%) and the best model performance
- **Tuberculosis** is the rarest (~4%) making it the hardest to predict
- Diseases like **Diabetes** and **Heart Disease** show high co-occurrence
- **Smoking** has a measurable impact on respiratory diseases (COPD, Tuberculosis)

---

## 🤖 Live Patient Risk Assessment

The notebook includes a live prediction cell — input any patient profile and get a risk score for all 10 diseases:

```python
new_patient = {
    "Age": 55,
    "Gender": "Male",
    "Blood Pressure": "High",
    "Cholesterol": "High",
    "Glucose": "High",
    "Smoking": "Yes",
    "Alcohol Consumption": "No",
    "Exercise": "No",
    "BMI": 32.5,
    "Family History": "Yes"
}
```

Output:
```
Heart Disease          → 68.3%  🔴 HIGH
Diabetes               → 45.1%  🟡 MEDIUM
Stroke                 → 31.2%  🟡 MEDIUM
Kidney Disease         → 28.4%  🟡 MEDIUM
...
Tuberculosis           → 5.2%   🟢 LOW
```

---

## ▶️ How to Run

**1. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

**2. Place the dataset in the project folder**
```
health_disease_pridiction.csv  ← same directory as the notebook
```

**3. Open and run the notebook**
```bash
jupyter notebook health_disease_prediction_xgboost.ipynb
```

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data loading & manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Preprocessing, metrics, train/test split |
| `xgboost` | Gradient boosting classifier |

---

## 🗺️ ML Journey Progress

| # | Algorithm | Dataset | Status |
|---|-----------|---------|--------|
| 01 | Linear Regression | Laptop Price Prediction | ✅ Done |
| 02 | Logistic Regression | Loan Approval Prediction | ✅ Done |
| **03** | **XGBoost** | **Health Disease Prediction** | ✅ **Done** |
| 04 | Random Forest | Coming soon… | 🔜 |
| 05 | Neural Network | Coming soon… | 🔜 |

---

*Built to understand how gradient boosting handles real-world multi-label medical classification problems.*
