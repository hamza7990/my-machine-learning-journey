# 💻 Laptop Price Prediction — Linear Regression with Gradient Descent

> A full end-to-end machine learning pipeline that predicts laptop prices using a **custom Linear Regression model trained with Batch Gradient Descent**, built entirely from scratch with NumPy.

---

## 📁 Project Structure

```
📦 laptop-price-prediction/
├── 📓 laptop_price_prediction.ipynb   ← Main notebook (pipeline + model)
├── 📄 laptop_data.csv                 ← Raw dataset
└── 📄 README.md                       ← You are here
```

---

## 📊 Dataset Overview

The dataset (`laptop_data.csv`) contains **specifications and prices of laptops** scraped from e-commerce platforms.

| Column | Type | Description |
|--------|------|-------------|
| `Company` | Categorical | Laptop brand (Dell, Apple, Lenovo…) |
| `TypeName` | Categorical | Laptop type (Notebook, Gaming, Ultrabook…) |
| `Inches` | Numeric | Screen size in inches |
| `ScreenResolution` | Text | Full resolution string (e.g. `IPS 1920x1080 Touchscreen`) |
| `Cpu` | Text | Full CPU string (e.g. `Intel Core i7 2.8GHz`) |
| `Ram` | Text → Numeric | RAM size (e.g. `8GB` → `8`) |
| `Memory` | Text | Storage config (e.g. `256GB SSD + 1TB HDD`) |
| `Gpu` | Text | GPU description |
| `OpSys` | Categorical | Operating system |
| `Weight` | Text → Numeric | Laptop weight (e.g. `1.37kg` → `1.37`) |
| `Price` | Numeric | **Target variable** — price in local currency |

---

## 🔄 Pipeline Walkthrough

### 1️⃣ Exploratory Data Analysis (EDA)
- Bar charts of **average price** grouped by Company, TypeName, RAM, CPU, Memory, GPU
- **Price distribution** (raw + log-transformed) to understand skewness

### 2️⃣ Data Cleaning
- Drop the unnamed index column (`Unnamed: 0`)
- Remove **duplicate rows**
- Convert `Ram` from string `"8GB"` → float `8.0`
- Convert `Weight` from string `"1.37kg"` → float `1.37`

### 3️⃣ Feature Engineering

| New Feature | Source | Description |
|-------------|--------|-------------|
| `Cpu_Brand` | `Cpu` | First word: Intel / AMD / Samsung |
| `Cpu_Speed` | `Cpu` | Last token (GHz) as float |
| `resolution_width` | `ScreenResolution` | Extracted pixel width |
| `resolution_height` | `ScreenResolution` | Extracted pixel height |
| `is_touchscreen` | `ScreenResolution` | 1 if touchscreen, else 0 |
| `is_ips` | `ScreenResolution` | 1 if IPS panel, else 0 |
| `ppi` | Resolution + Inches | Pixel density = √(w²+h²) / inches |
| `SSD` | `Memory` | SSD storage in GB |
| `HDD` | `Memory` | HDD storage in GB |
| `Flash` | `Memory` | Flash storage in GB |
| `Hybrid` | `Memory` | Hybrid storage in GB |
| `Gpu_Brand` | `Gpu` | First word: Nvidia / Intel / AMD |

### 4️⃣ Encoding & Scaling
- **One-Hot Encoding** on: `Company`, `TypeName`, `Cpu_Brand`, `Gpu_Brand`, `OpSys`
- **StandardScaler** on all numeric columns: Inches, Ram, Weight, Cpu_Speed, Resolution, PPI, Storage columns

### 5️⃣ Train / Test Split
```
80% Training  |  20% Testing   (random_state=42)
```

---

## 🧠 Model — Linear Regression via Gradient Descent

The model is implemented **from scratch** using NumPy — no `sklearn` for the model itself.

### Math

$$\hat{y} = X \cdot w + b$$

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

$$\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n} X^T (\hat{y} - y) \qquad \frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum (\hat{y} - y)$$

$$w \leftarrow w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w} \qquad b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$$

### Class API

```python
model = LinearRegressionGD(
    learning_rate=0.01,   # Step size α
    n_iterations=1000,    # Number of gradient steps
    verbose=True,         # Print loss every log_every steps
    log_every=100
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Hyperparameter Tuning — Learning Rate Comparison

| Learning Rate | Test R² |
|:---:|:---:|
| 0.001 | lower / slower convergence |
| 0.005 | moderate |
| **0.01** | **best balance** |
| 0.05 | may overshoot |

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R² Score** | Proportion of variance explained (closer to 1 = better) |
| **RMSE** | Root Mean Squared Error — same unit as Price |
| **Residuals Plot** | Distribution of (Actual − Predicted) — should be centered at 0 |
| **Predicted vs Actual** | Scatter plot — points close to diagonal = good fit |

---

## ▶️ How to Run

**1. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**2. Place the dataset in the project folder**
```
laptop_data.csv   ← must be in the same directory as the notebook
```

**3. Open and run the notebook**
```bash
jupyter notebook laptop_price_prediction.ipynb
```

Run all cells top to bottom — each section is self-contained and labeled.

---

## 🛠️ Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Gradient descent math |
| `pandas` | ≥ 2.0 | Data loading & manipulation |
| `matplotlib` | ≥ 3.7 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical visualizations |
| `scikit-learn` | ≥ 1.3 | Preprocessing, metrics, train/test split |

---

## 💡 Key Takeaways

- **RAM, SSD, and CPU Brand** are among the strongest price predictors
- **Log-transforming** the target could further improve model fit (optional extension)
- The custom **Gradient Descent implementation** matches sklearn's LinearRegression in behavior, making it a great learning exercise
- Learning rate `0.01` with `1000` iterations gives stable convergence on this dataset

---

*Built as a learning project to understand linear regression internals through a practical real-world dataset.*
