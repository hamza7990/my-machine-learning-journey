# 🏦 Loan Approval Prediction — Logistic Regression from Scratch

> **Project 02** of my Machine Learning Journey
>
> A complete end-to-end machine learning pipeline that predicts whether a loan will be **Approved or Rejected** using a custom **Logistic Regression model trained with Batch Gradient Descent**, built entirely from scratch with NumPy.

---

## 📁 Project Structure

```
📦 02_logistic_regression/
├── 📓 loan_approval_logistic_regression.ipynb   ← Main notebook
├── 📄 loan_approval_dataset.csv                 ← Raw dataset
└── 📄 README.md                                 ← You are here
```

---

## 📊 Dataset Overview

The dataset contains **4,269 loan applications** with financial and personal information about each applicant.

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | ID | Unique loan identifier (dropped before training) |
| `no_of_dependents` | Numeric | Number of dependents of the applicant |
| `education` | Categorical | Graduate / Not Graduate |
| `self_employed` | Categorical | Yes / No |
| `income_annum` | Numeric | Annual income of the applicant |
| `loan_amount` | Numeric | Requested loan amount |
| `loan_term` | Numeric | Loan repayment term (in years) |
| `cibil_score` | Numeric | Credit score — **strongest predictor** |
| `residential_assets_value` | Numeric | Value of residential assets |
| `commercial_assets_value` | Numeric | Value of commercial assets |
| `luxury_assets_value` | Numeric | Value of luxury assets |
| `bank_asset_value` | Numeric | Value of bank assets |
| `loan_status` | **Target** | `Approved` → 1 / `Rejected` → 0 |

---

## 🔄 Pipeline Walkthrough

### 1️⃣ Exploratory Data Analysis (EDA)
- **Target distribution** — count of Approved vs Rejected
- **Box plots** of every numeric feature split by loan status
- **Bar charts** — approval rate by education & self-employment
- **Correlation heatmap** across all numerical features
- **CIBIL score histogram** — visualizing its strong separation power

### 2️⃣ Data Cleaning
- Strip leading/trailing whitespace from all column names
- Check and remove **duplicate rows**
- Drop `loan_id` (not a predictive feature)

### 3️⃣ Encoding
| Feature | Encoding |
|---------|---------|
| `loan_status` | `Approved` → 1, `Rejected` → 0 |
| `education` | `Graduate` → 1, `Not Graduate` → 0 |
| `self_employed` | `Yes` → 1, `No` → 0 |

### 4️⃣ Train / Test Split & Scaling
```
80% Training  |  20% Testing   (random_state=42, stratified)
```
- **StandardScaler** applied on all features
- `stratify=y` ensures same class ratio in both splits

---

## 🧠 Model — Logistic Regression via Gradient Descent

The model is implemented **from scratch** using NumPy only — no sklearn for the model itself.

### Math

**Sigmoid Activation:**

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \hat{y} = \sigma(Xw + b)$$

**Binary Cross-Entropy Loss:**

$$\mathcal{L} = -\frac{1}{n} \sum \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]$$

**Gradients:**

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T (\hat{y} - y) \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)$$

**Weight Update:**

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w} \qquad b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

### Class API

```python
model = LogisticRegressionGD(
    learning_rate=0.1,    # Step size α
    n_iterations=1000,    # Gradient descent steps
    threshold=0.5,        # Decision boundary
    verbose=True,
    log_every=100
)
model.fit(X_train, y_train)

y_pred       = model.predict(X_test)        # Binary predictions (0 or 1)
y_pred_proba = model.predict_proba(X_test)  # Raw probabilities
```

### Key Implementation Details
- **Numerically stable sigmoid** — handles large positive and negative values without overflow
- **Clipped log** — prevents `log(0)` errors in the loss function
- Tracks both **loss** and **accuracy** at every iteration

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **ROC-AUC** | Model's ability to distinguish Approved vs Rejected |
| **Confusion Matrix** | True/False Positives and Negatives |
| **Precision** | Of predicted Approved — how many were actually approved |
| **Recall** | Of actual Approved — how many did the model catch |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Probability Distribution** | How confidently the model separates the two classes |

---

## 🔍 Key Findings

- **`cibil_score`** is by far the strongest predictor of loan approval
- Applicants with higher **income** and more **assets** are more likely to get approved
- **Education** level has a noticeable effect on approval rates
- The model converges well with `learning_rate=0.1` over 1000 iterations

### Learning Rate Comparison

| Learning Rate | Behavior |
|:---:|---------|
| 0.001 | Very slow convergence |
| 0.01 | Moderate — needs more iterations |
| **0.1** | ✅ Best balance of speed and stability |
| 0.5 | Fast but may overshoot |

---

## ▶️ How to Run

**1. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**2. Place the dataset in the project folder**
```
loan_approval_dataset.csv  ← must be in the same directory as the notebook
```

**3. Open and run the notebook**
```bash
jupyter notebook loan_approval_logistic_regression.ipynb
```

Run all cells top to bottom — each section is labeled and self-contained.

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Sigmoid, gradients, matrix math |
| `pandas` | Data loading & manipulation |
| `matplotlib` | Plotting curves and charts |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Preprocessing, metrics, train/test split only |

---

## 🗺️ ML Journey Progress

| # | Algorithm | Dataset | Status |
|---|-----------|---------|--------|
| 01 | Linear Regression | Laptop Price Prediction | ✅ Done |
| **02** | **Logistic Regression** | **Loan Approval Prediction** | ✅ **Done** |
| 03 | Decision Tree | Coming soon… | 🔜 |
| 04 | Random Forest | Coming soon… | 🔜 |
| 05 | Neural Network | Coming soon… | 🔜 |

---

*Built from scratch to understand the internals of Logistic Regression through a real-world binary classification problem.*
