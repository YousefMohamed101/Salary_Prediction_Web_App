# 💰 IT Salary Predictor

A machine learning web app that predicts a yearly IT salary (in USD) from a person's experience, country, education level, company size, and their programming languages / frameworks. Built end-to-end: data preprocessing and model experimentation in a notebook, deployed as an interactive **Streamlit** app.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**🔗 Live app:** [Salary_Prediction_Web_App](https://itsalaryprediction.streamlit.app/)

---

## 📖 Overview

This project predicts an IT professional's expected yearly salary based on:

- Years of experience
- Country
- Education level
- Company size
- Programming languages known
- Frameworks known

Four regression models (**XGBoost, XGBoost Random Forest, CatBoost, and LightGBM**) were trained and compared on the same feature set. The best-performing, most reliable model was exported and wired up to a **Streamlit** front end so anyone can get a live salary estimate from a simple form.

---

## ✨ Features

- 🎛️ Interactive Streamlit UI — slider + dropdowns + multi-select, no coding required to use it
- 🌍 Supports 10 countries, 5 education levels, 6 company-size brackets
- 🧑‍💻 Multi-select for 10 programming languages and 10 frameworks
- ⚡ Instant prediction powered by a pre-trained, serialized CatBoost model
- 📓 Full model training / evaluation notebook included for transparency

---

## 🧠 The Machine Learning Pipeline

### 1. Data

- `train.csv` (40,000 rows) / `test.csv` — one row per IT professional
- Target: `salary_usd`, ranging roughly from **$12K to $278K**
- Raw features: `experience`, `country`, `education`, `company_size`, `languages`, `frameworks`

### 2. Preprocessing & feature engineering

| Feature | Technique |
|---|---|
| `education` | Ordinal mapping: `High School → PhD` (0–4) |
| `company_size` | Ordinal mapping: `1-10 → 5000+` (0–5) |
| `country` | Label-mapped to an integer per country |
| `languages` | Multi-label binarized → one binary column per language (10 cols) |
| `frameworks` | Multi-label binarized → one binary column per framework (10 cols) |
| `salary_usd` (target) | `log1p` transform during training to stabilize the right-skewed salary distribution, then `expm1` on prediction to convert back to USD |

After encoding, each record becomes a **24-feature vector**: `experience`, `country`, `education`, `company_size` + 10 one-hot language flags + 10 one-hot framework flags.

Data was split 75/25 into training and validation sets (on top of the separate held-out `test.csv`) for early stopping and unbiased evaluation.

### 3. Model comparison

All models were evaluated on the same held-out test set using R², Mean Absolute Percentage Error (MAPE), and an "accuracy" score defined as `(1 - MAPE) × 100`.

| Model | Key hyperparameters | R² Score | MAPE | Accuracy |
|---|---|:---:|:---:|:---:|
| **XGBoost Regressor** | `n_estimators=300`, `lr=0.03`, early stopping | 0.88 | 10.16% | 89.84% |
| **CatBoost Regressor** ⭐ | `iterations=500`, `lr=0.1`, early stopping (best iter. 128) | 0.88 | 10.16% | 89.84% |
| **LightGBM Regressor** | `n_estimators=5000`, `lr=0.01`, `num_leaves=31`, early stopping | 0.88 | 10.16% | 89.84% |
| XGBoost Random Forest (baseline) | `n_estimators=100`, `max_depth=5` | 0.85 | — | — |

**Takeaways:**
- The three **gradient-boosted** models (XGBoost, CatBoost, LightGBM) converged to essentially the same performance (R² ≈ 0.88, ~10% average error), suggesting the engineered feature set captures most of the signal available in the data, regardless of which boosting implementation is used.
- The **bagged Random Forest baseline** (XGBRFRegressor) trailed behind at R² = 0.85 — expected, since bagging tends to underperform sequential boosting on structured/tabular regression tasks like this one.
- **CatBoost** was selected as the production model. With comparable accuracy to XGBoost and LightGBM, it needed the fewest boosting iterations to converge (stopped automatically at iteration 128 via its overfitting detector) and required the least manual tuning to reach peak performance — making it the most practical choice to ship.

> 📓 Full training code, logs, and metrics are in [`SalaryPredictor.ipynb`](./SalaryPredictor.ipynb).

### 4. Deployment

The trained CatBoost model is serialized with `joblib` and loaded directly by the Streamlit app (`App.py`). At inference time, the app:
1. Collects the user's inputs through the UI
2. Encodes them into the same 24-feature layout the model was trained on
3. Predicts the log-salary and applies `expm1` to return a dollar figure

---

## 🛠️ Tech Stack

- **Language:** Python
- **Modeling:** XGBoost, CatBoost, LightGBM, scikit-learn
- **Web app:** Streamlit
- **Utilities:** pandas, NumPy, joblib

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/YousefMohamed101/Salary_Prediction_Web_App.git
cd Salary_Prediction_Web_App

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run App.py
```

The app will open in your browser (usually at `http://localhost:8501`).

### How to use it
1. Set your years of experience with the slider
2. Pick your country, education level, and company size
3. Select the programming languages and frameworks you know
4. Hit **Predict** to see your estimated yearly salary in USD

## 📄 License

This project is available for personal and educational use. Add a license of your choice (e.g. MIT) if you plan to open it up for reuse.
