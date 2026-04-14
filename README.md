# ML-Based Biomarker Identification for Early Cardiovascular Disease Detection

> **Topic:** Machine Learning–Based Biomarker Identification for Early Disease Detection (Cardiovascular)  
> **Dataset:** UCI Heart Disease Dataset — Cleveland (Kaggle / ketangangal)

---

## Project Structure

```
cvd_project/
├── cardiovascular_biomarker_ml.py   ← Complete pipeline (12 sections)
├── Heart_Disease_UCI.csv            ← Dataset (303 patients, 14 features)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
└── results/                         ← Auto-generated on first run
    ├── statistical_analysis.csv
    ├── lasso_coefficients.csv
    ├── feature_importance.csv
    ├── model_results.csv
    └── figures/
        ├── 01_overview.png
        ├── 02_correlation_heatmap.png
        ├── 03_violin_plots.png
        ├── 04_categorical_distributions.png
        ├── 05_pairplot.png
        ├── 06_statistical_significance.png
        ├── 07_lasso_analysis.png              ← LASSO path + CV error + coefficients
        ├── 08_feature_importance.png          ← Multi-method heatmap + composite bar
        ├── 09_roc_curves.png                  ← All models
        ├── 10_model_comparison.png            ← 4-metric bar comparison
        ├── 11_best_model_detail.png           ← Confusion + ROC + PR + threshold
        ├── 12_cv_stability.png                ← 10-fold CV box plots
        ├── 13_learning_curves.png             ← Accuracy & AUC learning curves
        └── 14_shap_analysis.png               ← Beeswarm + bar + waterfall + dependence
```

---

## Dataset Features

| Feature    | Type        | Description                              |
|------------|-------------|------------------------------------------|
| `age`      | Continuous  | Age in years                             |
| `sex`      | Binary      | 1 = Male, 0 = Female                     |
| `cp`       | Categorical | Chest pain type (0–3)                    |
| `trestbps` | Continuous  | Resting blood pressure (mmHg)            |
| `chol`     | Continuous  | Serum cholesterol (mg/dL)                |
| `fbs`      | Binary      | Fasting blood sugar > 120 mg/dL          |
| `restecg`  | Categorical | Resting ECG results (0–2)                |
| `thalach`  | Continuous  | Maximum heart rate achieved              |
| `exang`    | Binary      | Exercise-induced angina (1=Yes)          |
| `oldpeak`  | Continuous  | ST depression (exercise vs rest)         |
| `slope`    | Categorical | Slope of peak exercise ST segment (0–2)  |
| `ca`       | Ordinal     | Number of major vessels (0–3)            |
| `thal`     | Categorical | Thalassemia type (1–3)                   |
| **`target`** | **Binary** | **0 = No Disease, 1 = Disease**         |

---

## Pipeline Overview (12 Sections)

| # | Section | Description |
|---|---------|-------------|
| 0 | Imports | All libraries with graceful XGBoost/SHAP fallback |
| 1 | Data Loading | CSV / UCI auto-fetch; informative errors |
| 2 | Preprocessing | Median imputation, label encoding, stratified 80/20 split, StandardScaler |
| 3 | EDA | 5 figures: class balance, distributions, violin plots, categoricals, pairplot |
| 4 | Statistical Analysis | Mann-Whitney U (continuous), Chi² (categorical), Point-Biserial r |
| 5 | **LASSO Selection** | LassoCV optimal alpha, regularisation path, CV error, coefficient bar |
| 6 | **Composite Importance** | 7-method ranking: MI + RF + GBM + XGBoost + LASSO + LR + Permutation |
| 7 | Models | 9–10 classifiers defined |
| 8 | Training & Evaluation | 5-fold CV (Acc/AUC/F1/MCC) + hold-out test scores |
| 9 | Hyperparameter Tuning | GridSearchCV on best model |
| 10 | Visualisations | ROC, model comparison, confusion matrix, PR curve, threshold, CV box, learning curves |
| 11 | **SHAP** | Beeswarm, bar, waterfall, dependence plots |
| 12 | Final Report | Ranked biomarkers, LASSO table, stats, leaderboard, clinical interpretation |

### Models Included

| Model | Regularisation | Notes |
|-------|---------------|-------|
| Logistic Regression (L1) | **LASSO** | Sparse biomarker selection |
| Logistic Regression (L2) | Ridge | Stable baseline |
| Random Forest | Implicit | Ensemble, low variance |
| Gradient Boosting | Implicit | Sequential boosting |
| **XGBoost** ⭐ | L1 + L2 | Best in class for tabular data |
| SVM (RBF) | C | Non-linear decision boundary |
| K-Nearest Neighbors | — | Non-parametric |
| Decision Tree | — | Interpretable |
| Naïve Bayes | — | Probabilistic baseline |
| AdaBoost | — | Adaptive boosting |

