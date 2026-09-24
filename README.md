# Netflix Customer Churn Prediction and Data Analysis

**IBM SkillsBuild Data Analytics with AI — Academic Internship**  
**Submitted under:** BharatCares & AICTE  
**Student:** Telukuntla Yashwanth

---

## Project Overview

This project builds a complete end-to-end data analytics and machine learning pipeline to predict Netflix customer churn. It uses a structured dataset of 5,000 customer records to identify at-risk subscribers and derives actionable business recommendations to reduce churn.

**Target Variable:** `churned` — `0` = Active customer, `1` = Churned customer

---

## Project Files

| File | Description |
|------|-------------|
| `netflix_customer_churn.csv` | Source dataset (5,000 rows × 14 columns) |
| `TELUKUNTLA-YASHWANTH_NetflixCustomerChurn.ipynb` | Jupyter Notebook — interactive analysis |
| `TELUKUNTLA-YASHWANTH_NetflixCustomerChurn.py` | Python script — end-to-end pipeline |
| `TELUKUNTLA-YASHWANTH_ProjectReport.docx` | Full academic project report |
| `requirements.txt` | Python package dependencies |
| `README.md` | This file |
| `outputs/` | Saved chart PNGs and `model_results.csv` |

---

## Dataset

**File:** `netflix_customer_churn.csv`  
**Rows:** 5,000 | **Columns:** 14  
**Missing values:** None | **Duplicates:** None

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique identifier (dropped before ML) |
| `age` | int | Customer age (18–70) |
| `gender` | string | Male / Female / Other |
| `subscription_type` | string | Basic / Standard / Premium |
| `watch_hours` | float | Total hours watched |
| `last_login_days` | int | Days since last login |
| `region` | string | Geographic region (6 regions) |
| `device` | string | TV / Mobile / Laptop / Desktop / Tablet |
| `monthly_fee` | float | Monthly subscription fee (USD) |
| `payment_method` | string | Credit Card / Debit Card / PayPal / Crypto / Gift Card |
| `number_of_profiles` | int | Profiles on account (1–5) |
| `avg_watch_time_per_day` | float | Average daily watch time (hrs) |
| `favorite_genre` | string | 7 content genres |
| `churned` | int | **Target: 0 = Active, 1 = Churned** |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 2. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd "IBM PROJECT"

# Or simply download and extract the project folder
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`, `notebook`

### 4. Verify Dataset

Ensure `netflix_customer_churn.csv` is present in the project root directory (same folder as the notebook and script).

---

## How to Run

### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Then open `TELUKUNTLA-YASHWANTH_NetflixCustomerChurn.ipynb` in your browser and run all cells:

**Kernel → Restart & Run All**

### Option B — Python Script

```bash
python TELUKUNTLA-YASHWANTH_NetflixCustomerChurn.py
```

All charts will be saved to the `outputs/` directory automatically. A `model_results.csv` will also be saved there.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load and inspect dataset |
| 2 | Auto-detect target column (`churned`) |
| 3 | Check missing values and duplicates |
| 4 | Exploratory Data Analysis (12 charts) |
| 5 | Label Encoding (categorical features) |
| 6 | StandardScaler (numerical features) |
| 7 | 80/20 stratified train-test split |
| 8 | Train Logistic Regression, Decision Tree, Random Forest |
| 9 | Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| 10 | 5-fold cross-validation |
| 11 | Feature importance (Random Forest) |
| 12 | Business insights and conclusion |

---

## Results Summary

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 88.50% | 0.8876 | 0.9588 |
| Decision Tree | 93.50% | 0.9332 | 0.9841 |
| **Random Forest** | **95.70%** | **0.9570** | **0.9934** |

**Best Model:** Random Forest  
**Top Features:** `avg_watch_time_per_day`, `watch_hours`, `last_login_days`

---

## Output Files (generated in `outputs/`)

```
01_churn_distribution.png
02_churn_gender.png
03_churn_subscription.png
04_churn_age.png
05_churn_watch_hours.png
06_churn_last_login.png
07_churn_region.png
08_churn_device.png
09_churn_payment.png
10_churn_genre_profiles.png
11_churn_monthly_fee.png
12_correlation_heatmap.png
13_confusion_matrices.png
14_roc_curves.png
15_model_comparison.png
16_feature_importance.png
17_cv_scores.png
model_results.csv
```

---

## Academic Information

| Field | Details |
|-------|---------|
| **Program** | IBM SkillsBuild Data Analytics with AI |
| **Initiative** | BharatCares & AICTE Academic Internship |
| **Student** | Telukuntla Yashwanth |
| **Language** | Python 3 |
| **Random State** | 42 (all models and splits) |
| **Purpose** | Academic project — not for commercial use |

---

## License

This project is submitted strictly for academic purposes under the IBM SkillsBuild Data Analytics with AI internship programme (BharatCares & AICTE). The dataset used is a structured demonstration dataset. No real personal data belonging to actual Netflix subscribers was used or processed.
