# Credit Card Transaction Fraud Detection 

An **Advanced Data Analytics & Processing** project for detecting fraudulent
credit card transactions. The current work compares LightGBM, a bidirectional GRU,
and an FT-Transformer, combines their predictions with a logistic-regression
stacking ensemble, and uses SHAP to explain the strongest model.

## Project Status

The weekly progress report :

1. Clean transaction data and handle missing values.
2. Build time-based, statistical, and card-behaviour features.
3. Encode categorical columns and trim the feature set using SHAP.
4. Apply SMOTE-ENN to address the heavy fraud-class imbalance.
5. Train LightGBM, a bidirectional GRU, and an FT-Transformer independently.
6. Combine out-of-fold predictions with a logistic-regression stacking ensemble.
7. Use SHAP to explain LightGBM predictions and trace an example fraud decision.

The report notes that deep-learning training was performed on Data Science Lab
CPU hardware. Training times are approximate rather than measured timings.

## Reported Results

| Model | Approximate training time | AUC-ROC |
| --- | ---: | ---: |
| LightGBM | ~5 hours | 0.9768 |
| Bidirectional GRU | ~3 hours | 0.7667 |
| FT-Transformer | ~4 hours | 0.8511 |
| Stacking ensemble | Not recorded | 0.9625 |

LightGBM currently performs best as a standalone model. The stacking ensemble
does not yet improve on it because the weaker GRU and FT-Transformer predictions
reduce the combined score.

SHAP identified **C13, D1, C5, D4, and C14** as the most influential features in
the reported analysis. A sample fraud transaction was also inspected with a
waterfall plot to show which features increased its fraud score.

## Current Repository Structure

The repository currently contains only the following tracked project files. The
structure described in the progress report is treated as experimental context;
it is not added here as an assumed directory layout.

```
.
├── .gitignore  # Excludes datasets, model artifacts, and generated outputs
└── README.md   # Project documentation and progress summary
```

Raw data, trained models, notebooks, and executable training code are not
currently tracked in this repository.

## Next Steps

- Reweight or remove weaker base models so the ensemble can exceed LightGBM.
- Perform hyperparameter searches for the GRU and FT-Transformer.
- Test alternative resampling settings to improve the deep-learning models.
- Record training times during execution instead of estimating them afterward.

## Data and Artifact Policy

The `.gitignore` excludes raw tabular datasets, generated outputs, model
checkpoints, archives, and local Python environments. This keeps sensitive or
large experiment artifacts out of Git. If the pipeline is added later, datasets
and generated artifacts should remain local unless explicitly approved for
version control.
