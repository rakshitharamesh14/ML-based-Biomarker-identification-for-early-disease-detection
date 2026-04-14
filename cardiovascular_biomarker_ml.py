"""
================================================================================
  ML-BASED BIOMARKER IDENTIFICATION FOR EARLY CARDIOVASCULAR DISEASE DETECTION
  ─────────────────────────────────────────────────────────────────────────────
  Dataset  : UCI Heart Disease Dataset  (Cleveland / Kaggle – ketangangal)
  Topic    : Machine Learning–Based Biomarker Identification
  Models   : Logistic Regression (LASSO/Ridge), Random Forest, Gradient
             Boosting, XGBoost, SVM, KNN, Naïve Bayes, AdaBoost, Decision
             Tree, Voting Ensemble
  Key Extras: LASSO regularisation path, SHAP explainability, composite
             multi-method biomarker ranking, full statistical testing
  Output   : 12+ publication-quality figures + 4 CSVs + console report
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0 – IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings, sys, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from itertools import cycle

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_val_score, GridSearchCV,
                                       learning_curve)
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, roc_auc_score,
                                       roc_curve, precision_recall_curve,
                                       average_precision_score, f1_score,
                                       matthews_corrcoef)
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection        import permutation_importance
from sklearn.linear_model      import (LogisticRegression, LassoCV, Lasso)
from sklearn.ensemble          import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        AdaBoostClassifier,
                                        VotingClassifier)
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.calibration       import CalibratedClassifierCV

# ── XGBoost (optional) ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("[OK] XGBoost detected.")
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed  →  pip install xgboost")

# ── SHAP (optional) ───────────────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
    print("[OK] SHAP detected.")
except ImportError:
    HAS_SHAP = False
    print("[WARN] SHAP not installed  →  pip install shap")

# ── scipy ─────────────────────────────────────────────────────────────────────
from scipy.stats import mannwhitneyu, chi2_contingency, pointbiserialr

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5
TEST_SIZE    = 0.20
np.random.seed(RANDOM_STATE)

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"
FIGS_DIR   = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})

BLUE   = "#3A86FF"
RED    = "#E63946"
GREEN  = "#2DC653"
ORANGE = "#FB8500"
PURPLE = "#7209B7"
GRAY   = "#ADB5BD"

# ── domain metadata ───────────────────────────────────────────────────────────
FEATURE_META = {
    "age":      "Age (years)",
    "sex":      "Sex (0=Female, 1=Male)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar > 120",
    "restecg":  "Resting ECG (0–2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise)",
    "slope":    "ST Slope (0–2)",
    "ca":       "Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia (1–3)",
}

CATEGORICAL_COLS = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}

CLINICAL_NOTES = {
    "cp":      "Atypical chest pain patterns carry strong diagnostic signal",
    "thalach": "Reduced peak HR reflects impaired cardiac reserve",
    "oldpeak": "ST-depression magnitude indicates ischaemic burden",
    "ca":      "Vessel occlusion count from fluoroscopy – structural marker",
    "thal":    "Thalassemia type links hereditary defects to cardiac risk",
    "exang":   "Exercise angina = positive stress-test biomarker",
    "age":     "Primary non-modifiable cardiovascular risk factor",
    "chol":    "Modifiable lipid biomarker – targets therapy",
    "sex":     "Male sex confers independent elevated risk",
    "slope":   "Downsloping ST pattern associated with ischaemia",
    "trestbps":"Resting hypertension – chronic pressure load on heart",
    "fbs":     "Hyperglycaemia reflects metabolic–cardiovascular overlap",
    "restecg": "Resting ECG abnormality – baseline cardiac pathology",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 – DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath=None):
    """Load dataset from file, UCI repository, or raise informative error."""

    COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
               "thalach","exang","oldpeak","slope","ca","thal","target"]

    # 1a. Provided file
    if filepath and Path(filepath).exists():
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        print(f"[DATA] Loaded from file  →  shape={df.shape}")

    # 1b. Script-local CSV (Heart_Disease_UCI.csv in same folder)
    elif (BASE_DIR / "Heart_Disease_UCI.csv").exists():
        df = pd.read_csv(BASE_DIR / "Heart_Disease_UCI.csv")
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        print(f"[DATA] Loaded Heart_Disease_UCI.csv  →  shape={df.shape}")

    # 1c. UCI repository fallback
    else:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        try:
            df = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")
            print(f"[DATA] Fetched from UCI  →  shape={df.shape}")
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot load data.\n  File not found locally.\n"
                f"  UCI fetch failed: {e}\n"
                f"  → Place Heart_Disease_UCI.csv in the project folder."
            )

    # Ensure binary target
    if "target" not in df.columns:
        raise ValueError("Dataset must have a 'target' column.")
    if df["target"].nunique() > 2:
        df["target"] = (df["target"] > 0).astype(int)

    print(f"[DATA] Shape: {df.shape}  |  "
          f"Disease: {df.target.sum()} ({df.target.mean():.1%})  |  "
          f"Healthy: {(df.target==0).sum()} ({(df.target==0).mean():.1%})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 – PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    print("\n[PREP] Preprocessing …")

    df = df.copy()

    # Handle missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"[PREP] Missing values detected:\n{missing[missing>0]}")
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    Xs_tr   = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_names,
                            index=X_tr.index)
    Xs_te   = pd.DataFrame(scaler.transform(X_te),     columns=feature_names,
                            index=X_te.index)

    print(f"[PREP] Train: {X_tr.shape}  |  Test: {X_te.shape}")
    print(f"[PREP] Train class balance → "
          f"0: {(y_tr==0).sum()}  1: {(y_tr==1).sum()}")

    return X, y, X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te, scaler, feature_names


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 – EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df, feature_names):
    print("\n[EDA] Running EDA …")

    # ── Figure 1 : Overview  (class balance / age / thalach) ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Dataset Overview — UCI Heart Disease", fontsize=13, y=1.01)

    cnt = df["target"].value_counts().sort_index()
    bars = axes[0].bar(["No Disease","Disease"], cnt.values,
                       color=[BLUE, RED], width=0.45, edgecolor="white", lw=1.5)
    axes[0].set_title("Class Distribution"); axes[0].set_ylabel("Count")
    for b in bars:
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+2,
                     str(int(b.get_height())), ha="center", fontweight="bold")

    for t, lbl, c in [(0,"No Disease", BLUE),(1,"Disease", RED)]:
        axes[1].hist(df[df.target==t]["age"], bins=14, alpha=0.65,
                     label=lbl, color=c, edgecolor="white")
    axes[1].set_title("Age Distribution by Class")
    axes[1].set_xlabel("Age"); axes[1].legend()

    for t, lbl, c in [(0,"No Disease", BLUE),(1,"Disease", RED)]:
        axes[2].hist(df[df.target==t]["thalach"], bins=14, alpha=0.65,
                     label=lbl, color=c, edgecolor="white")
    axes[2].set_title("Max Heart Rate by Class")
    axes[2].set_xlabel("thalach"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"01_overview.png", bbox_inches="tight"); plt.close()

    # ── Figure 2 : Correlation heatmap ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, square=True,
                vmin=-1, vmax=1, annot_kws={"size":8})
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"02_correlation_heatmap.png", bbox_inches="tight"); plt.close()

    # ── Figure 3 : Violin plots (continuous biomarkers) ───────────────────────
    cont  = [c for c in ["age","trestbps","chol","thalach","oldpeak"]
             if c in df.columns]
    fig, axes = plt.subplots(1, len(cont), figsize=(4*len(cont), 5))
    for ax, col in zip(axes, cont):
        g0 = df[df.target==0][col].dropna()
        g1 = df[df.target==1][col].dropna()
        parts = ax.violinplot([g0, g1], positions=[0,1], showmedians=True,
                               showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([BLUE, RED][i]); pc.set_alpha(0.72)
        for part in ("cbars","cmins","cmaxes","cmedians"):
            if part in parts:
                parts[part].set_color("black"); parts[part].set_lw(1.2)
        ax.set_xticks([0,1])
        ax.set_xticklabels(["No Dis.","Disease"], fontsize=8)
        ax.set_title(FEATURE_META.get(col, col), fontsize=9)
    fig.suptitle("Continuous Biomarker Distributions by Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"03_violin_plots.png", bbox_inches="tight"); plt.close()

    # ── Figure 4 : Categorical biomarkers – grouped bar charts ────────────────
    cats  = [c for c in ["cp","exang","slope","ca","thal","sex"]
             if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(cats)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()
    for i, col in enumerate(cats):
        ct = (df.groupby([col,"target"]).size()
                .unstack(fill_value=0)
                .rename(columns={0:"No Disease",1:"Disease"}))
        ct.plot(kind="bar", ax=axes[i], color=[BLUE, RED],
                edgecolor="white", width=0.65, rot=0)
        axes[i].set_title(FEATURE_META.get(col, col), fontsize=9)
        axes[i].set_xlabel(""); axes[i].legend(fontsize=7)
    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Categorical Biomarker Distributions by Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"04_categorical_distributions.png", bbox_inches="tight"); plt.close()

    # ── Figure 5 : Pair plot (top continuous features) ────────────────────────
    top5 = [c for c in ["age","thalach","oldpeak","chol","trestbps"]
            if c in df.columns][:5] + ["target"]
    g = sns.pairplot(df[top5], hue="target",
                     palette={0: BLUE, 1: RED},
                     plot_kws={"alpha":0.45, "s":18},
                     diag_kind="kde")
    g.fig.suptitle("Pairplot — Key Continuous Biomarkers", y=1.01, fontsize=12)
    g.fig.savefig(FIGS_DIR/"05_pairplot.png", bbox_inches="tight"); plt.close()

    print(f"[EDA] 5 figures saved → {FIGS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 – STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def statistical_analysis(df, feature_names):
    print("\n[STAT] Running statistical significance tests …")

    rows = []
    for feat in feature_names:
        if feat not in df.columns:
            continue
        g0 = df[df.target==0][feat].dropna()
        g1 = df[df.target==1][feat].dropna()

        if feat in CATEGORICAL_COLS:
            ct = pd.crosstab(df[feat], df["target"])
            stat, p, _, _ = chi2_contingency(ct)
            test = "Chi²"
        else:
            res  = mannwhitneyu(g0, g1, alternative="two-sided")
            stat, p = res.statistic, res.pvalue
            test = "Mann-Whitney U"

        r, _ = pointbiserialr(
            df["target"].dropna(),
            df[feat].dropna().reindex(df["target"].dropna().index)
        )
        rows.append({
            "Feature":    feat,
            "Test":       test,
            "Statistic":  round(float(stat), 4),
            "P-Value":    round(p, 6),
            "Sig(p<.05)": "✓" if p < 0.05 else "✗",
            "r_PBC":      round(r, 4),
            "Mean_Neg":   round(g0.mean(), 3),
            "Mean_Pos":   round(g1.mean(), 3),
            "Effect_Dir": "↑" if g1.mean() > g0.mean() else "↓",
        })

    stat_df = pd.DataFrame(rows).sort_values("P-Value")
    print(stat_df.to_string(index=False))
    stat_df.to_csv(OUTPUT_DIR/"statistical_analysis.csv", index=False)

    # ── Figure 6 : Significance volcano-style bar ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Biomarker Statistical Significance", fontsize=13)

    log_p  = -np.log10(stat_df["P-Value"] + 1e-10)
    colors = [RED if p < 0.05 else GRAY for p in stat_df["P-Value"]]
    bars   = axes[0].barh(stat_df["Feature"], log_p, color=colors, edgecolor="white")
    axes[0].axvline(-np.log10(0.05), color="black", ls="--", lw=1.5,
                    label="p = 0.05")
    axes[0].set_xlabel("-log₁₀(p-value)")
    axes[0].set_title("Statistical Significance  (red = p < 0.05)")
    axes[0].legend()

    r_vals  = stat_df.sort_values("r_PBC", key=abs, ascending=False)
    r_cols  = [RED if v > 0 else BLUE for v in r_vals["r_PBC"]]
    axes[1].barh(r_vals["Feature"], r_vals["r_PBC"], color=r_cols, edgecolor="white")
    axes[1].axvline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("Point-Biserial Correlation (r)")
    axes[1].set_title("Effect Size — r_PBC  (red=risk↑, blue=risk↓)")

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"06_statistical_significance.png", bbox_inches="tight")
    plt.close()

    return stat_df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 – LASSO BIOMARKER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def lasso_selection(Xs_tr, y_tr, Xs_te, y_te, feature_names):
    print("\n[LASSO] LASSO regularisation for biomarker selection …")

    # 5a. Cross-validate to find optimal alpha
    alphas    = np.logspace(-4, 1, 100)
    lasso_cv  = LassoCV(alphas=alphas, cv=CV_FOLDS,
                        random_state=RANDOM_STATE, max_iter=20000)
    lasso_cv.fit(Xs_tr, y_tr)
    best_alpha = lasso_cv.alpha_
    print(f"[LASSO] Optimal α = {best_alpha:.6f}")

    # 5b. Final LASSO with best alpha
    lasso = Lasso(alpha=best_alpha, max_iter=20000, random_state=RANDOM_STATE)
    lasso.fit(Xs_tr, y_tr)

    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": lasso.coef_,
        "Abs_Coef":    np.abs(lasso.coef_),
        "Selected":    np.abs(lasso.coef_) > 0,
        "Direction":   ["↑ Risk" if c > 0 else ("↓ Risk" if c < 0 else "Zeroed")
                        for c in lasso.coef_],
    }).sort_values("Abs_Coef", ascending=False)

    selected = coef_df[coef_df["Selected"]]["Feature"].tolist()
    zeroed   = coef_df[~coef_df["Selected"]]["Feature"].tolist()
    print(f"[LASSO] Selected ({len(selected)}): {selected}")
    print(f"[LASSO] Zeroed   ({len(zeroed)}):  {zeroed}")
    coef_df.to_csv(OUTPUT_DIR/"lasso_coefficients.csv", index=False)

    # 5c. Build regularisation path
    path_coefs = []
    for a in alphas:
        m = Lasso(alpha=a, max_iter=20000)
        m.fit(Xs_tr, y_tr)
        path_coefs.append(m.coef_.copy())
    path_coefs = np.array(path_coefs)   # shape (n_alphas, n_features)

    # ── Figure 7 : LASSO regularisation path + selected coefficients ──────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("LASSO Regularisation — Biomarker Selection", fontsize=13)

    # Path
    palette = plt.cm.tab20(np.linspace(0, 1, len(feature_names)))
    for i, feat in enumerate(feature_names):
        axes[0].plot(np.log10(alphas), path_coefs[:, i],
                     label=feat, color=palette[i], lw=1.6, alpha=0.9)
    axes[0].axvline(np.log10(best_alpha), color="black", ls="--", lw=2,
                    label=f"Best α={best_alpha:.4f}")
    axes[0].set_xlabel("log₁₀(α)")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Regularisation Path")
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")

    # LassoCV MSE path
    mse_path = lasso_cv.mse_path_.mean(axis=1)
    axes[1].plot(np.log10(lasso_cv.alphas_), mse_path, color=BLUE, lw=2)
    axes[1].axvline(np.log10(best_alpha), color=RED, ls="--", lw=2,
                    label=f"Best α={best_alpha:.4f}")
    axes[1].set_xlabel("log₁₀(α)")
    axes[1].set_ylabel("Mean CV MSE")
    axes[1].set_title("Cross-Validation Error vs Alpha")
    axes[1].legend()

    # Selected coefficients
    nz   = coef_df[coef_df["Selected"]].sort_values("Coefficient")
    c_cl = [RED if v > 0 else BLUE for v in nz["Coefficient"]]
    axes[2].barh(nz["Feature"], nz["Coefficient"], color=c_cl, edgecolor="white")
    axes[2].axvline(0, color="grey", lw=0.8)
    axes[2].set_title("Selected Biomarkers & Coefficients\n(red=risk↑ blue=risk↓)")
    axes[2].set_xlabel("LASSO Coefficient")
    for i, (v, n) in enumerate(zip(nz["Coefficient"], nz["Feature"])):
        axes[2].text(v + (0.003 if v >= 0 else -0.003), i,
                     f"{v:+.3f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"07_lasso_analysis.png", bbox_inches="tight"); plt.close()

    return coef_df, selected, best_alpha


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 – COMPOSITE FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_analysis(X_tr, y_tr, X_te, y_te, feature_names,
                                  Xs_tr, Xs_te, lasso_coef_df):
    print("\n[FEAT] Multi-method composite importance …")

    scores = {}

    # Mutual Information
    mi = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE)
    scores["Mutual Info"] = dict(zip(feature_names, mi))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    scores["Random Forest"] = dict(zip(feature_names, rf.feature_importances_))

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
    gb.fit(X_tr, y_tr)
    scores["Grad Boosting"] = dict(zip(feature_names, gb.feature_importances_))

    # XGBoost (if available)
    if HAS_XGB:
        xgb_m = XGBClassifier(n_estimators=300, learning_rate=0.05,
                               max_depth=6, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0)
        xgb_m.fit(X_tr, y_tr)
        scores["XGBoost"] = dict(zip(feature_names, xgb_m.feature_importances_))

    # LASSO |coef|
    lasso_imp = lasso_coef_df.set_index("Feature")["Abs_Coef"].to_dict()
    scores["LASSO |coef|"] = {f: lasso_imp.get(f, 0) for f in feature_names}

    # Logistic Regression |coef|
    lr = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    lr.fit(Xs_tr, y_tr)
    scores["LR |coef|"] = dict(zip(feature_names, np.abs(lr.coef_[0])))

    # Permutation importance
    perm = permutation_importance(rf, X_te, y_te, n_repeats=30,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    scores["Permutation"] = dict(
        zip(feature_names, np.clip(perm.importances_mean, 0, None))
    )

    imp_df = pd.DataFrame(scores).fillna(0)
    normed = imp_df.apply(
        lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9)
    )
    imp_df["Composite"] = normed.mean(axis=1)
    imp_df = imp_df.sort_values("Composite", ascending=False)
    imp_df.to_csv(OUTPUT_DIR/"feature_importance.csv")

    print("\n[FEAT] Composite Biomarker Rankings:")
    print(imp_df["Composite"].round(4).to_string())

    # ── Figure 8 : Importance heatmap + composite bar ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Multi-Method Biomarker Importance Analysis", fontsize=13)

    method_cols = [c for c in normed.columns if c != "Composite"]
    heat_data   = normed[method_cols].loc[imp_df.index]
    sns.heatmap(heat_data.T, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.4, ax=axes[0], vmin=0, vmax=1,
                annot_kws={"size": 8})
    axes[0].set_title("Normalised Importance — All Methods")
    axes[0].set_xlabel("Feature")
    axes[0].tick_params(axis="x", rotation=45)

    comp    = imp_df["Composite"].sort_values(ascending=True)
    c_ramp  = plt.cm.RdYlGn(np.linspace(0.15, 0.90, len(comp)))
    axes[1].barh(comp.index, comp.values, color=c_ramp, edgecolor="white")
    axes[1].set_title("Composite Biomarker Importance Score")
    axes[1].set_xlabel("Score (0–1)")
    for i, v in enumerate(comp.values):
        axes[1].text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"08_feature_importance.png", bbox_inches="tight"); plt.close()

    top_features = imp_df.head(8).index.tolist()
    print(f"\n[FEAT] Top-8 biomarkers: {top_features}")
    return imp_df, top_features


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 – MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_models():
    models = {
        "LR — LASSO (L1)":   LogisticRegression(penalty="l1", solver="liblinear",
                                                  C=0.5, max_iter=5000,
                                                  random_state=RANDOM_STATE),
        "LR — Ridge (L2)":   LogisticRegression(penalty="l2", C=1.0,
                                                  max_iter=5000,
                                                  random_state=RANDOM_STATE),
        "Random Forest":      RandomForestClassifier(n_estimators=300,
                                                      random_state=RANDOM_STATE),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200,
                                                           learning_rate=0.1,
                                                           max_depth=4,
                                                           random_state=RANDOM_STATE),
        "SVM (RBF)":          SVC(kernel="rbf", probability=True, C=1.0,
                                   random_state=RANDOM_STATE),
        "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=7,
                                                    weights="distance"),
        "Decision Tree":      DecisionTreeClassifier(max_depth=6,
                                                      random_state=RANDOM_STATE),
        "Naïve Bayes":        GaussianNB(),
        "AdaBoost":           AdaBoostClassifier(n_estimators=200,
                                                  learning_rate=0.5,
                                                  random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", use_label_encoder=False,
            random_state=RANDOM_STATE, verbosity=0,
        )
    return models


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 – MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(models, Xs_tr, y_tr, Xs_te, y_te):
    print("\n[MODEL] Training and evaluating classifiers …")
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    rows = {}
    trained = {}

    for name, clf in models.items():
        cv_acc = cross_val_score(clf, Xs_tr, y_tr, cv=cv,
                                  scoring="accuracy", n_jobs=-1)
        cv_auc = cross_val_score(clf, Xs_tr, y_tr, cv=cv,
                                  scoring="roc_auc",  n_jobs=-1)
        cv_f1  = cross_val_score(clf, Xs_tr, y_tr, cv=cv,
                                  scoring="f1",       n_jobs=-1)

        clf.fit(Xs_tr, y_tr)
        trained[name] = clf

        y_pred = clf.predict(Xs_te)
        y_prob = (clf.predict_proba(Xs_te)[:,1]
                  if hasattr(clf,"predict_proba") else None)

        rows[name] = {
            "Model":      name,
            "CV_Acc":     round(cv_acc.mean(), 4),
            "CV_Acc_std": round(cv_acc.std(),  4),
            "CV_AUC":     round(cv_auc.mean(), 4),
            "CV_AUC_std": round(cv_auc.std(),  4),
            "CV_F1":      round(cv_f1.mean(),  4),
            "Test_Acc":   round(accuracy_score(y_te, y_pred), 4),
            "Test_AUC":   round(roc_auc_score(y_te, y_prob), 4) if y_prob is not None else np.nan,
            "Test_F1":    round(f1_score(y_te, y_pred), 4),
            "Test_MCC":   round(matthews_corrcoef(y_te, y_pred), 4),
        }
        print(f"  {name:<26} | "
              f"Acc={rows[name]['Test_Acc']:.3f}  "
              f"AUC={rows[name]['Test_AUC']:.3f}  "
              f"F1={rows[name]['Test_F1']:.3f}  "
              f"MCC={rows[name]['Test_MCC']:.3f}")

    res_df = (pd.DataFrame(rows)
                .T
                .sort_values("Test_AUC", ascending=False)
                .reset_index(drop=True))
    res_df.to_csv(OUTPUT_DIR/"model_results.csv", index=False)
    return res_df, trained


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 – HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_best_model(trained, res_df, Xs_tr, y_tr, Xs_te, y_te):
    best_name = res_df.iloc[0]["Model"]
    print(f"\n[TUNE] Grid-search tuning: {best_name}")

    param_grids = {
        "XGBoost": {
            "n_estimators":  [200, 300, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth":     [4, 6],
            "subsample":     [0.7, 0.8, 1.0],
        },
        "Gradient Boosting": {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 4, 5],
        },
        "Random Forest": {
            "n_estimators":       [200, 300, 500],
            "max_depth":          [None, 10, 20],
            "min_samples_split":  [2, 5],
        },
        "SVM (RBF)": {
            "C":     [0.1, 1, 5, 10],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
        "LR — LASSO (L1)": {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        "LR — Ridge (L2)": {"C": [0.01, 0.1, 1.0, 5.0, 10.0]},
    }

    clf  = trained[best_name]
    grid = param_grids.get(best_name)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if grid:
        gs = GridSearchCV(clf, grid, cv=cv, scoring="roc_auc",
                          n_jobs=-1, verbose=0)
        gs.fit(Xs_tr, y_tr)
        best_clf = gs.best_estimator_
        print(f"[TUNE] Best params: {gs.best_params_}")
    else:
        best_clf = clf
        best_clf.fit(Xs_tr, y_tr)

    y_pred = best_clf.predict(Xs_te)
    y_prob = (best_clf.predict_proba(Xs_te)[:,1]
              if hasattr(best_clf,"predict_proba") else None)

    print(f"[TUNE] Final  →  "
          f"Acc={accuracy_score(y_te,y_pred):.4f}  "
          f"AUC={roc_auc_score(y_te,y_prob):.4f}  "
          f"F1={f1_score(y_te,y_pred):.4f}  "
          f"MCC={matthews_corrcoef(y_te,y_pred):.4f}")

    return best_name, best_clf, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10 – VISUALISATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(trained, res_df, Xs_tr, y_tr, Xs_te, y_te,
                 best_name, best_clf, y_pred, y_prob):
    print("\n[PLOT] Generating result figures …")

    # ── Figure 9 : ROC curves — all models ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    pal     = plt.cm.tab10(np.linspace(0, 1, len(trained)))
    for (name, clf), c in zip(trained.items(), pal):
        if hasattr(clf, "predict_proba"):
            prob  = clf.predict_proba(Xs_te)[:,1]
            fpr, tpr, _ = roc_curve(y_te, prob)
            auc   = roc_auc_score(y_te, prob)
            lw    = 2.8 if name == best_name else 1.3
            alpha = 1.0 if name == best_name else 0.75
            ax.plot(fpr, tpr, color=c, lw=lw, alpha=alpha,
                    label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — All Models  (bold = {best_name})")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"09_roc_curves.png", bbox_inches="tight"); plt.close()

    # ── Figure 10 : Model comparison — 4 metrics ─────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    fig.suptitle("Model Performance Comparison  (red = best model)", fontsize=13)
    for ax, metric, title in zip(
        axes,
        ["Test_Acc","Test_AUC","Test_F1","Test_MCC"],
        ["Accuracy","ROC-AUC","F1 Score","MCC"]
    ):
        sd = res_df.sort_values(metric, ascending=True)
        sd[metric] = pd.to_numeric(sd[metric], errors="coerce")
        c_list = [RED if n==best_name else "#AED6F1" for n in sd["Model"]]
        ax.barh(sd["Model"], sd[metric].fillna(0), color=c_list, edgecolor="white")
        lo = max(0.0, float(sd[metric].dropna().min()) - 0.15)
        ax.set_xlim(lo, 1.0)
        ax.set_title(title)
        for i, (v, n) in enumerate(zip(sd[metric], sd["Model"])):
            if pd.notna(v):
                ax.text(float(v)+0.003, i, f"{float(v):.3f}",
                        va="center", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"10_model_comparison.png", bbox_inches="tight"); plt.close()

    # ── Figure 11 : Best model — confusion + ROC + PR + calibration ──────────
    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig)
    fig.suptitle(f"Best Model — {best_name}", fontsize=13)

    # Confusion
    ax0  = fig.add_subplot(gs[0])
    cm   = confusion_matrix(y_te, y_pred)
    # Add % annotations
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annots = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
                        for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annots, fmt="", cmap="Blues", ax=ax0,
                xticklabels=["No Dis.","Disease"],
                yticklabels=["No Dis.","Disease"],
                linewidths=1.5, linecolor="white",
                annot_kws={"size":11, "weight":"bold"})
    ax0.set_title("Confusion Matrix"); ax0.set_ylabel("True"); ax0.set_xlabel("Predicted")

    if y_prob is not None:
        ax1 = fig.add_subplot(gs[1])
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ax1.plot(fpr, tpr, color=RED, lw=2.5, label=f"AUC={auc:.3f}")
        ax1.fill_between(fpr, tpr, alpha=0.12, color=RED)
        ax1.plot([0,1],[0,1],"k--",lw=1)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(alpha=0.2)

        ax2 = fig.add_subplot(gs[2])
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)
        ax2.step(rec, prec, color=BLUE, lw=2.5, where="post",
                 label=f"AP={ap:.3f}")
        ax2.fill_between(rec, prec, alpha=0.12, color=BLUE, step="post")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve"); ax2.legend(); ax2.grid(alpha=0.2)

        ax3 = fig.add_subplot(gs[3])
        thresh = np.linspace(0, 1, 50)
        accs   = [accuracy_score(y_te, (y_prob >= t).astype(int)) for t in thresh]
        f1s    = [f1_score(y_te, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresh]
        ax3.plot(thresh, accs, color=GREEN,  lw=2, label="Accuracy")
        ax3.plot(thresh, f1s,  color=ORANGE, lw=2, label="F1 Score")
        ax3.axvline(0.5, color="grey", ls="--", lw=1)
        ax3.set_xlabel("Decision Threshold"); ax3.set_ylabel("Score")
        ax3.set_title("Threshold Sensitivity"); ax3.legend(); ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"11_best_model_detail.png", bbox_inches="tight"); plt.close()

    # ── Figure 12 : CV stability — box plots ─────────────────────────────────
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_d = {n: cross_val_score(c, Xs_te, y_te, cv=cv10, scoring="roc_auc")
            for n, c in trained.items()}
    srt  = sorted(cv_d, key=lambda n: np.median(cv_d[n]), reverse=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot([cv_d[n] for n in srt], patch_artist=True, notch=True,
                    medianprops={"color":"black","lw":2})
    for patch, name in zip(bp["boxes"], srt):
        patch.set_facecolor(RED if name==best_name else "#AED6F1")
        patch.set_alpha(0.78)
    ax.set_xticklabels(srt, rotation=32, ha="right", fontsize=8.5)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model Stability — 10-Fold CV AUC Distribution  (red = best)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"12_cv_stability.png", bbox_inches="tight"); plt.close()

    # ── Figure 13 : Learning curves (best model) ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Learning Curves — {best_name}", fontsize=13)

    for ax, scoring, title, c1, c2 in [
        (axes[0], "accuracy", "Accuracy",  GREEN,  BLUE),
        (axes[1], "roc_auc",  "ROC-AUC",   ORANGE, RED),
    ]:
        tr_sz, tr_sc, cv_sc = learning_curve(
            best_clf, Xs_tr, y_tr,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring=scoring, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
        cv_m, cv_s = cv_sc.mean(1), cv_sc.std(1)
        ax.plot(tr_sz, tr_m, "o-", color=c1, lw=2, label="Train")
        ax.fill_between(tr_sz, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color=c1)
        ax.plot(tr_sz, cv_m, "s-", color=c2, lw=2, label="CV Valid.")
        ax.fill_between(tr_sz, cv_m-cv_s, cv_m+cv_s, alpha=0.15, color=c2)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(title)
        ax.set_title(f"Learning Curve — {title}")
        ax.legend(); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"13_learning_curves.png", bbox_inches="tight"); plt.close()

    print(f"[PLOT] 5 result figures saved → {FIGS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 11 – SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
def shap_analysis(best_name, best_clf, Xs_te, feature_names):
    if not HAS_SHAP:
        print("\n[SHAP] Skipped — install shap:  pip install shap")
        return

    print("\n[SHAP] Computing SHAP values …")
    try:
        Xs_te_arr = np.array(Xs_te)
        tree_kws  = ("Forest","Boost","XGB","Ada","Tree")

        if any(k in best_name for k in tree_kws):
            explainer   = shap.TreeExplainer(best_clf)
            shap_vals   = explainer(Xs_te_arr)
        else:
            background  = shap.sample(Xs_te_arr, min(50, len(Xs_te_arr)),
                                       random_state=RANDOM_STATE)
            explainer   = shap.KernelExplainer(best_clf.predict_proba, background)
            raw         = explainer.shap_values(Xs_te_arr)
            # Convert to Explanation object
            base = (explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value)
            shap_vals = shap.Explanation(
                values      = raw[1] if isinstance(raw, list) else raw,
                base_values = np.full(len(Xs_te_arr), base),
                data        = Xs_te_arr,
                feature_names = feature_names,
            )

        # Handle multi-class shape (n_samples, n_features, n_classes)
        if hasattr(shap_vals, "values") and shap_vals.values.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        # ── Figure 14 : SHAP — beeswarm, bar, waterfall, dependence ──────────
        fig = plt.figure(figsize=(22, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(f"SHAP Explainability — {best_name}", fontsize=14,
                     fontweight="bold")

        # Beeswarm
        ax0 = fig.add_subplot(gs[0, 0])
        plt.sca(ax0)
        shap.plots.beeswarm(shap_vals, max_display=13, show=False)
        ax0.set_title("Beeswarm — Impact Distribution")

        # Bar (mean |SHAP|)
        ax1 = fig.add_subplot(gs[0, 1])
        plt.sca(ax1)
        shap.plots.bar(shap_vals, max_display=13, show=False)
        ax1.set_title("Global Feature Importance (mean |SHAP|)")

        # Waterfall — first positive sample
        pos_idx = np.where(np.array(shap_vals.base_values +
                                     shap_vals.values.sum(1)) > 0.5)[0]
        sample_idx = int(pos_idx[0]) if len(pos_idx) > 0 else 0
        ax2 = fig.add_subplot(gs[0, 2])
        plt.sca(ax2)
        shap.plots.waterfall(shap_vals[sample_idx], show=False)
        ax2.set_title(f"Waterfall — Sample #{sample_idx} (disease case)")

        # Dependence plots — top 3 features
        sv_arr  = shap_vals.values
        top3    = np.argsort(np.abs(sv_arr).mean(0))[::-1][:3]
        for k, feat_idx in enumerate(top3):
            ax = fig.add_subplot(gs[1, k])
            feat_name = feature_names[feat_idx]
            shap.dependence_plot(
                feat_idx, sv_arr, Xs_te_arr,
                feature_names=feature_names,
                ax=ax, show=False, dot_size=20, alpha=0.7
            )
            ax.set_title(f"Dependence — {feat_name}")

        plt.savefig(FIGS_DIR/"14_shap_analysis.png", bbox_inches="tight")
        plt.close()
        print("[SHAP] Figure 14 saved.")

    except Exception as e:
        print(f"[SHAP] Error: {e} — skipping SHAP figure.")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 12 – FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_final_report(res_df, imp_df, stat_df, lasso_coef_df,
                        best_name, best_clf, y_pred, y_prob, y_te,
                        feature_names):
    SEP  = "─" * 70
    DSEP = "═" * 70

    print(f"\n\n{DSEP}")
    print("  RESEARCH REPORT")
    print("  Topic : ML-Based Biomarker Identification — Cardiovascular Disease")
    print(f"{DSEP}")

    # A – Composite importance
    print(f"\n{SEP}")
    print("  A · COMPOSITE BIOMARKER RANKINGS  (multi-method average)")
    print(SEP)
    print(f"  {'Rank':<6} {'Feature':<14} {'Composite':>10}  Clinical Note")
    print(f"  {'─'*4}   {'─'*12}   {'─'*10}  {'─'*30}")
    for rank, (feat, row) in enumerate(
            imp_df["Composite"].items(), start=1):
        note = CLINICAL_NOTES.get(feat, "—")
        print(f"  {rank:<6} {feat:<14} {row:>10.4f}  {note}")

    # B – LASSO
    print(f"\n{SEP}")
    print("  B · LASSO SELECTED BIOMARKERS")
    print(SEP)
    nz = lasso_coef_df[lasso_coef_df["Selected"]].copy()
    nz = nz.sort_values("Abs_Coef", ascending=False)
    print(f"  {'Feature':<14} {'Coef':>10}  Direction")
    for _, r in nz.iterrows():
        print(f"  {r.Feature:<14} {r.Coefficient:>10.4f}  {r.Direction}")
    z_feats = lasso_coef_df[~lasso_coef_df["Selected"]]["Feature"].tolist()
    print(f"\n  Zeroed out (uninformative): {z_feats}")

    # C – Statistical tests
    print(f"\n{SEP}")
    print("  C · STATISTICAL SIGNIFICANCE  (p < 0.05 = significant)")
    print(SEP)
    print(stat_df[["Feature","Test","P-Value","Sig(p<.05)","r_PBC",
                   "Effect_Dir"]].to_string(index=False))

    # D – Model leaderboard
    print(f"\n{SEP}")
    print("  D · MODEL LEADERBOARD  (sorted by Test AUC)")
    print(SEP)
    print(res_df[["Model","CV_AUC","CV_AUC_std","Test_Acc",
                  "Test_AUC","Test_F1","Test_MCC"]].to_string(index=False))

    # E – Best model report
    print(f"\n{SEP}")
    print(f"  E · BEST MODEL CLASSIFICATION REPORT  →  {best_name}")
    print(SEP)
    print(classification_report(y_te, y_pred,
                                  target_names=["No Disease","Disease"],
                                  digits=4))
    if y_prob is not None:
        print(f"  ROC-AUC            : {roc_auc_score(y_te, y_prob):.4f}")
        print(f"  Average Precision  : {average_precision_score(y_te, y_prob):.4f}")
        print(f"  Matthews Corr (MCC): {matthews_corrcoef(y_te, y_pred):.4f}")

    # F – Clinical interpretation
    print(f"\n{SEP}")
    print("  F · CLINICAL INTERPRETATION")
    print(SEP)
    for rank, feat in enumerate(imp_df["Composite"].head(8).index, 1):
        lasso_row  = lasso_coef_df[lasso_coef_df["Feature"]==feat]
        lasso_note = (f"  LASSO={lasso_row.iloc[0].Coefficient:+.3f}"
                      if len(lasso_row) > 0 else "")
        stat_row   = stat_df[stat_df["Feature"]==feat]
        p_note     = (f"  p={stat_row.iloc[0]['P-Value']:.4f}"
                      if len(stat_row) > 0 else "")
        print(f"  {rank}. {feat:<12}  {CLINICAL_NOTES.get(feat,'—')}")
        print(f"     └─ {lasso_note}  {p_note}")

    # G – Output files
    print(f"\n{SEP}")
    print("  G · OUTPUT FILES")
    print(SEP)
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  {str(f.relative_to(BASE_DIR)):<55} {size:>7.1f} KB")

    print(f"\n{DSEP}")
    print(f"  All outputs saved to: {OUTPUT_DIR.resolve()}")
    print(DSEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def main(filepath=None):
    # ── 1. Data ───────────────────────────────────────────────────────────────
    df = load_data(filepath)

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    (X, y, X_tr, X_te, y_tr, y_te,
     Xs_tr, Xs_te, scaler, features) = preprocess(df)

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    run_eda(df, features)

    # ── 4. Statistical analysis ───────────────────────────────────────────────
    stat_df = statistical_analysis(df, features)

    # ── 5. LASSO ──────────────────────────────────────────────────────────────
    lasso_coef_df, selected, best_alpha = lasso_selection(
        Xs_tr, y_tr, Xs_te, y_te, features
    )

    # ── 6. Composite importance ───────────────────────────────────────────────
    imp_df, top_feat = feature_importance_analysis(
        X_tr, y_tr, X_te, y_te, features, Xs_tr, Xs_te, lasso_coef_df
    )

    # ── 7. Train all models ───────────────────────────────────────────────────
    models          = build_models()
    res_df, trained = evaluate_models(models, Xs_tr, y_tr, Xs_te, y_te)

    # ── 8. Tune best model ────────────────────────────────────────────────────
    best_name, best_clf, y_pred, y_prob = tune_best_model(
        trained, res_df, Xs_tr, y_tr, Xs_te, y_te
    )

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_results(trained, res_df, Xs_tr, y_tr, Xs_te, y_te,
                 best_name, best_clf, y_pred, y_prob)

    # ── 10. SHAP ──────────────────────────────────────────────────────────────
    shap_analysis(best_name, best_clf, Xs_te, features)

    # ── 11. Final report ──────────────────────────────────────────────────────
    print_final_report(res_df, imp_df, stat_df, lasso_coef_df,
                       best_name, best_clf, y_pred, y_prob, y_te, features)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv)
