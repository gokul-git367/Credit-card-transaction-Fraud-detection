# Credit Card Transaction Fraud Detection (ADP Project)

An **Advanced Data Analytics & Processing (ADP)** machine learning solution for detecting fraudulent online credit card transactions, benchmarked on the **IEEE-CIS Fraud Detection** dataset targeting an **80%+ evaluation/confidence threshold** (ROC-AUC / Precision-Recall).

---

## 📌 Project Overview

Online transaction fraud leads to billions of dollars in losses annually. This project builds an end-to-end, high-throughput fraud detection model designed to identify anomalous transaction behavior across complex financial features, categorical identifiers, and network attributes.

### Key Objectives & Highlights
- **IEEE-CIS Fraud Detection Benchmark**: Utilizing transaction (`train_transaction.csv`) and identity (`train_identity.csv`) data tables containing tabular, spatial, device, and network attributes.
- **80% Performance Threshold Target**: Optimized for $\ge 0.80$ ROC-AUC score and high precision-recall trade-offs to minimize false positives in financial decisioning.
- **Feature Engineering Pipeline**: Includes time-delta features, card frequency encoding, transaction amount aggregations, device/browser fingerprinting, and interaction terms.
- **Class Imbalance Management**: Handling extreme positive class imbalance ($\sim 3.5\%$ fraud rate) using SMOTE, focal loss, and scale-pos-weight tuning in ensemble models (XGBoost, LightGBM, CatBoost).
- **Model Explainability**: SHAP (SHapley Additive exPlanations) integration to provide transparent, interpretable fraud decision rationale for auditing.

---

## 📁 Repository Structure

```
.
├── .gitignore              # Ignores large IEEE-CIS CSVs, model weights, and outputs
├── README.md               # Main project documentation
├── data/                   # Local dataset directory (ignored by Git)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── notebooks/              # Exploratory data analysis & feature experimentation
├── src/                    # Production source code
│   ├── data_preprocessing.py # Data merging, missing value imputation & encoding
│   ├── feature_engineering.py# Temporal, aggregation & interaction feature generation
│   ├── train.py            # Model training & hyperparameter tuning
│   └── evaluate.py         # IEEE-CIS 80% threshold validation & SHAP analysis
└── outputs/                # Evaluation reports, confusion matrices & submission files (ignored)
```

---

## ⚙️ Environment Setup & Dependencies

Install required scientific, machine learning, and visualization libraries:

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost shap matplotlib seaborn
```

---

## 🚀 Workflow & Execution Guide

### 1. Data Preprocessing & Feature Engineering
Clean incoming transaction and identity records, handle missing values, and construct aggregated features:
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
```

### 2. Model Training & Hyperparameter Tuning
Train ensemble models (LightGBM / XGBoost) with class-rebalancing:
```bash
python src/train.py --model lightgbm --threshold 0.80
```

### 3. Evaluation & Threshold Verification
Validate against the **IEEE-CIS 80% benchmark target**:
```bash
python src/evaluate.py --threshold 0.80
```

---

## 🔒 Data Privacy & Git Policy (`.gitignore`)

In compliance with IEEE-CIS data policies and GitHub repository size limits, **all raw CSV datasets (`*.csv`), model checkpoints (`*.model`, `*.pth`, `*.joblib`), and generated output logs are strictly excluded from Git tracking**.

To run the pipeline locally:
1. Download the IEEE-CIS Fraud Detection dataset.
2. Place the CSV files (`train_transaction.csv`, `train_identity.csv`, etc.) inside the `data/` folder.
3. Outputs will automatically save to `outputs/` without polluting your Git commits.
