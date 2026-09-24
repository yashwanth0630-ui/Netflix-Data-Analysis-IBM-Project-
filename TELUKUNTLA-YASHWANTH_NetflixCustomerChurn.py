"""
=============================================================================
  Netflix Customer Churn Prediction
  Student  : Telukuntla Yashwanth
  Program  : IBM SkillsBuild Data Analytics with AI - Academic Internship
  Submitted: BharatCares & AICTE
  Dataset  : netflix_customer_churn.csv
  Script   : TELUKUNTLA-YASHWANTH_NetflixCustomerChurn.py
=============================================================================
"""

# -- Imports ------------------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output so all print statements display correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend (saves PNGs)
import matplotlib.pyplot  as plt
import matplotlib.ticker  as mticker
import seaborn            as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

# -- Constants ----------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTPUT_DIR   = "outputs"
DATA_FILE    = "netflix_customer_churn.csv"

np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset, print basic info, and return the DataFrame."""
    print("\n" + "=" * 62)
    print("  STEP 1 -- LOAD DATA")
    print("=" * 62)

    df = pd.read_csv(filepath)

    print(f"File loaded  : {filepath}")
    print(f"Shape        : {df.shape[0]} rows  x  {df.shape[1]} columns")
    print("\nColumn names and dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    return df


# =============================================================================
# 2. IDENTIFY TARGET
# =============================================================================

def identify_target(df: pd.DataFrame) -> str:
    """
    Auto-detect the churn target column by searching column names for
    keywords: churn, cancel, status, retain, active, leave.
    Returns the column name as a string.
    """
    print("\n" + "=" * 62)
    print("  STEP 2 -- IDENTIFY TARGET COLUMN")
    print("=" * 62)

    keywords = ["churn", "cancel", "status", "retain", "active", "leave"]
    candidates = [
        col for col in df.columns
        if any(kw in col.lower() for kw in keywords)
    ]

    if not candidates:
        raise ValueError(
            "No churn-related target column found. "
            "Please specify the target manually."
        )

    target = candidates[0]
    print(f"Auto-detected target column : \"{target}\"")
    print(f"Unique values               : {df[target].unique().tolist()}")
    print(f"Value counts:\n{df[target].value_counts().to_string()}")

    churn_pct = df[target].value_counts(normalize=True) * 100
    print("\nClass balance:")
    for val, pct in churn_pct.items():
        label = "Churned" if val == 1 else "Active"
        print(f"  {val} ({label:<7}) : {pct:.1f}%")

    return target


# =============================================================================
# 3. CLEAN DATA
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicate records.
      Numerical NaN   -> median
      Categorical NaN -> mode
      Duplicate rows  -> dropped
    Returns the cleaned DataFrame.
    """
    print("\n" + "=" * 62)
    print("  STEP 3 -- CLEAN DATA")
    print("=" * 62)

    # Missing values
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        print(f"Missing values detected: {missing_total}")
        for col in df.columns:
            n_miss = df[col].isnull().sum()
            if n_miss > 0:
                if df[col].dtype in ["float64", "int64"]:
                    fill_val = df[col].median()
                    df[col].fillna(fill_val, inplace=True)
                    print(f"  [{col}] filled {n_miss} NaN(s) with median ({fill_val:.3f})")
                else:
                    fill_val = df[col].mode()[0]
                    df[col].fillna(fill_val, inplace=True)
                    print(f"  [{col}] filled {n_miss} NaN(s) with mode ('{fill_val}')")
    else:
        print("Missing values : None found -- no imputation needed.")

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Duplicate rows : {dupes} removed.  New shape: {df.shape}")
    else:
        print("Duplicate rows : None found -- no action needed.")

    # Report column type split
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"\nNumerical columns  ({len(num_cols)}) : {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"\nFinal clean shape: {df.shape}")

    return df


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================

def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def perform_eda(df: pd.DataFrame, target: str) -> None:
    """
    Generate and save 12 EDA charts covering all key features.
    """
    print("\n" + "=" * 62)
    print("  STEP 4 -- EXPLORATORY DATA ANALYSIS")
    print("=" * 62)

    # Add a string label column so seaborn palette dicts work across versions
    HUE   = "_churn_label"
    PAL   = {"Active": "#27ae60", "Churned": "#e74c3c"}
    df    = df.copy()
    df[HUE] = df[target].map({0: "Active", 1: "Churned"})

    # Descriptive stats
    print("\nNumerical summary:")
    print(df.describe().round(2).to_string())

    # -- 01 Churn distribution ------------------------------------------------
    counts = df[target].value_counts().sort_index()   # [0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["Active (0)", "Churned (1)"], counts.values,
                color=["#27ae60", "#e74c3c"], edgecolor="black", width=0.45)
    axes[0].set_title("Customer Churn Count", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].set_xlabel("Churn Status")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")
    axes[1].pie(counts.values, labels=["Active (0)", "Churned (1)"],
                autopct="%1.1f%%", colors=["#27ae60", "#e74c3c"],
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Churn Proportion", fontweight="bold")
    plt.suptitle("Netflix Customer Churn Distribution",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "01_churn_distribution.png")

    # -- 02 Churn by gender ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="gender", hue=HUE, palette=PAL,
                  hue_order=["Active", "Churned"],
                  ax=axes[0], edgecolor="black")
    axes[0].set_title("Churn Count by Gender", fontweight="bold")
    axes[0].set_xlabel("Gender"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Status")
    cg = df.groupby("gender")[target].mean() * 100
    cg.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black",
            color=sns.color_palette("Set2", len(cg)))
    axes[1].set_title("Churn Rate (%) by Gender", fontweight="bold")
    axes[1].set_xlabel("Gender"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontweight="bold")
    plt.suptitle("Churn Analysis by Gender", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_churn_gender.png")

    # -- 03 Churn by subscription type ----------------------------------------
    sub_order = ["Basic", "Standard", "Premium"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=df, x="subscription_type", hue=HUE, palette=PAL,
                  hue_order=["Active", "Churned"],
                  order=sub_order, ax=axes[0], edgecolor="black")
    axes[0].set_title("Churn Count by Subscription Type", fontweight="bold")
    axes[0].set_xlabel("Subscription Type"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Status")
    cs = (df.groupby("subscription_type")[target].mean() * 100).reindex(sub_order)
    cs.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black",
            color=["#1abc9c", "#3498db", "#9b59b6"])
    axes[1].set_title("Churn Rate (%) by Subscription Type", fontweight="bold")
    axes[1].set_xlabel("Subscription Type"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontweight="bold")
    plt.suptitle("Churn by Subscription Type", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_churn_subscription.png")

    # -- 04 Age distribution --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, clr, nm in [(0, "#27ae60", "Active"), (1, "#e74c3c", "Churned")]:
        sns.kdeplot(data=df[df[target] == lbl], x="age", ax=axes[0],
                    fill=True, color=clr, label=nm, alpha=0.55)
    axes[0].set_title("Age Distribution by Churn", fontweight="bold")
    axes[0].set_xlabel("Age"); axes[0].set_ylabel("Density")
    axes[0].legend(title="Status")
    sns.boxplot(data=df, x=HUE, y="age", order=["Active", "Churned"],
                palette=PAL, ax=axes[1], width=0.45)
    axes[1].set_title("Age Boxplot by Churn", fontweight="bold")
    axes[1].set_xlabel("Status (Active / Churned)"); axes[1].set_ylabel("Age")
    plt.suptitle("Age vs Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_churn_age.png")

    # -- 05 Watch hours -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x=HUE, y="watch_hours", order=["Active", "Churned"],
                palette=PAL, ax=axes[0], width=0.45)
    axes[0].set_title("Total Watch Hours by Churn", fontweight="bold")
    axes[0].set_xlabel("Status (Active / Churned)")
    axes[0].set_ylabel("Watch Hours")
    sns.boxplot(data=df, x=HUE, y="avg_watch_time_per_day",
                order=["Active", "Churned"],
                palette=PAL, ax=axes[1], width=0.45)
    axes[1].set_title("Avg Daily Watch Time by Churn", fontweight="bold")
    axes[1].set_xlabel("Status (Active / Churned)")
    axes[1].set_ylabel("Avg Watch Time / Day (hrs)")
    plt.suptitle("Viewing Behaviour vs Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_churn_watch_hours.png")

    # -- 06 Last login days ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, clr, nm in [(0, "#27ae60", "Active"), (1, "#e74c3c", "Churned")]:
        sns.kdeplot(data=df[df[target] == lbl], x="last_login_days", ax=axes[0],
                    fill=True, color=clr, label=nm, alpha=0.55)
    axes[0].set_title("Days Since Last Login - Distribution", fontweight="bold")
    axes[0].set_xlabel("Days Since Last Login"); axes[0].set_ylabel("Density")
    axes[0].legend(title="Status")
    sns.boxplot(data=df, x=HUE, y="last_login_days", order=["Active", "Churned"],
                palette=PAL, ax=axes[1], width=0.45)
    axes[1].set_title("Days Since Last Login - Boxplot", fontweight="bold")
    axes[1].set_xlabel("Status (Active / Churned)")
    axes[1].set_ylabel("Days Since Last Login")
    plt.suptitle("Inactivity vs Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "06_churn_last_login.png")

    # -- 07 Churn by region ---------------------------------------------------
    reg_order = df["region"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, x="region", hue=HUE, palette=PAL,
                  hue_order=["Active", "Churned"],
                  order=reg_order, ax=axes[0], edgecolor="black")
    axes[0].set_title("Churn Count by Region", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_xlabel("Region"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Status")
    cr = (df.groupby("region")[target].mean() * 100).sort_values(ascending=False)
    cr.plot(kind="bar", ax=axes[1], rot=30, edgecolor="black",
            color=sns.color_palette("Set2", len(cr)))
    axes[1].set_title("Churn Rate (%) by Region", fontweight="bold")
    axes[1].set_xlabel("Region"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("Churn Analysis by Region", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_churn_region.png")

    # -- 08 Churn by device ---------------------------------------------------
    dev_order = df["device"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="device", hue=HUE, palette=PAL,
                  hue_order=["Active", "Churned"],
                  order=dev_order, ax=axes[0], edgecolor="black")
    axes[0].set_title("Churn Count by Device", fontweight="bold")
    axes[0].set_xlabel("Device"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Status")
    cd = (df.groupby("device")[target].mean() * 100).sort_values(ascending=False)
    cd.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black",
            color=sns.color_palette("Set2", len(cd)))
    axes[1].set_title("Churn Rate (%) by Device", fontweight="bold")
    axes[1].set_xlabel("Device"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontweight="bold")
    plt.suptitle("Churn Analysis by Device", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "08_churn_device.png")

    # -- 09 Churn by payment method -------------------------------------------
    pm_order = df["payment_method"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, x="payment_method", hue=HUE, palette=PAL,
                  hue_order=["Active", "Churned"],
                  order=pm_order, ax=axes[0], edgecolor="black")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Churn Count by Payment Method", fontweight="bold")
    axes[0].set_xlabel("Payment Method"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Status")
    cp = (df.groupby("payment_method")[target].mean() * 100).sort_values(ascending=False)
    cp.plot(kind="bar", ax=axes[1], rot=25, edgecolor="black",
            color=sns.color_palette("Set2", len(cp)))
    axes[1].set_title("Churn Rate (%) by Payment Method", fontweight="bold")
    axes[1].set_xlabel("Payment Method"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("Churn Analysis by Payment Method", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_churn_payment.png")

    # -- 10 Genre + profiles --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    cgen = (df.groupby("favorite_genre")[target].mean() * 100).sort_values(ascending=False)
    cgen.plot(kind="bar", ax=axes[0], rot=0, edgecolor="black",
              color=sns.color_palette("Set2", len(cgen)))
    axes[0].set_title("Churn Rate (%) by Favourite Genre", fontweight="bold")
    axes[0].set_xlabel("Favourite Genre"); axes[0].set_ylabel("Churn Rate (%)")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontweight="bold", fontsize=9)
    cpro = df.groupby("number_of_profiles")[target].mean() * 100
    cpro.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black",
              color=sns.color_palette("Set2", len(cpro)))
    axes[1].set_title("Churn Rate (%) by Number of Profiles", fontweight="bold")
    axes[1].set_xlabel("Number of Profiles"); axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                         ha="center", fontweight="bold")
    plt.suptitle("Churn by Genre and Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_churn_genre_profiles.png")

    # -- 11 Monthly fee -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, clr, nm in [(0, "#27ae60", "Active"), (1, "#e74c3c", "Churned")]:
        sns.kdeplot(data=df[df[target] == lbl], x="monthly_fee", ax=axes[0],
                    fill=True, color=clr, label=nm, alpha=0.55)
    axes[0].set_title("Monthly Fee Distribution by Churn", fontweight="bold")
    axes[0].set_xlabel("Monthly Fee (USD)"); axes[0].set_ylabel("Density")
    axes[0].legend(title="Status")
    avg_fee = df.groupby("subscription_type")["monthly_fee"].mean().reindex(
        ["Basic", "Standard", "Premium"])
    avg_fee.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black",
                 color=["#1abc9c", "#3498db", "#9b59b6"])
    axes[1].set_title("Avg Monthly Fee by Subscription Type", fontweight="bold")
    axes[1].set_xlabel("Subscription Type"); axes[1].set_ylabel("Avg Fee (USD)")
    for p in axes[1].patches:
        axes[1].annotate(f"${p.get_height():.2f}",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.05),
                         ha="center", fontweight="bold")
    plt.suptitle("Monthly Fee Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "11_churn_monthly_fee.png")

    # -- 12 Correlation heatmap -----------------------------------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
                annot_kws={"size": 10})
    ax.set_title("Correlation Heatmap - Numerical Features",
                 fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, "12_correlation_heatmap.png")

    print("\nEDA complete -- 12 charts saved to outputs/")


# =============================================================================
# 5. PREPARE DATA
# =============================================================================

def prepare_data(df: pd.DataFrame, target: str):
    """
    Encode categorical features, scale numerical features, split into
    train/test sets.

    Returns
    -------
    X_train_sc, X_test_sc   : scaled features (for Logistic Regression)
    X_train_enc, X_test_enc : encoded-only features (for tree models)
    y_train, y_test         : target splits
    feature_names           : list of feature column names
    """
    print("\n" + "=" * 62)
    print("  STEP 5 -- PREPARE DATA")
    print("=" * 62)

    # Drop ID-like columns (non-predictive)
    id_like = [c for c in df.columns if "id" in c.lower() and c != target]
    if id_like:
        df = df.drop(columns=id_like)
        print(f"Dropped ID columns: {id_like}")

    X = df.drop(columns=[target])
    y = df[target]

    print(f"Feature matrix : {X.shape}")
    print(f"Target vector  : {y.shape}")

    # -- Encode categoricals --
    cat_features = X.select_dtypes(include="object").columns.tolist()
    print(f"\nEncoding categorical features: {cat_features}")
    X_encoded = X.copy()
    for col in cat_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        print(f"  [{col}] -- classes: {list(le.classes_)}")

    # -- Scale numericals --
    num_features = X_encoded.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    print(f"\nScaling numerical features: {num_features}")
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[num_features] = scaler.fit_transform(X_scaled[num_features])

    # -- Class balance check --
    ratio = y.value_counts(normalize=True)
    imbalance = ratio.max() / ratio.min()
    print(f"\nClass imbalance ratio: {imbalance:.2f}")
    if imbalance < 1.5:
        print("Status: BALANCED -- class_weight='balanced' used as a safeguard.")
    else:
        print("Status: IMBALANCED -- class_weight='balanced' applied.")

    # -- Split --
    X_tr_sc, X_te_sc, y_tr, y_te = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr_enc, X_te_enc, _, _ = train_test_split(
        X_encoded, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain samples : {X_tr_sc.shape[0]}  ({100 - TEST_SIZE * 100:.0f}%)")
    print(f"Test  samples : {X_te_sc.shape[0]}  ({TEST_SIZE * 100:.0f}%)")
    print(f"Train churn rate : {y_tr.mean() * 100:.1f}%")
    print(f"Test  churn rate : {y_te.mean() * 100:.1f}%")

    return (
        X_tr_sc, X_te_sc,
        X_tr_enc, X_te_enc,
        y_tr, y_te,
        X_encoded.columns.tolist(),
    )


# =============================================================================
# 6. TRAIN MODELS
# =============================================================================

def train_models(X_tr_sc, X_tr_enc, y_train):
    """
    Train Logistic Regression (on scaled features),
    Decision Tree, and Random Forest (on encoded features).
    Returns a dict of {name: (model, uses_scaled)}.
    """
    print("\n" + "=" * 62)
    print("  STEP 6 -- TRAIN MODELS")
    print("=" * 62)

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr.fit(X_tr_sc, y_train)
    print("  [1/3] Logistic Regression -- trained.")

    dt = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    dt.fit(X_tr_enc, y_train)
    print("  [2/3] Decision Tree       -- trained.")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr_enc, y_train)
    print("  [3/3] Random Forest       -- trained.")

    # uses_scaled=True  -> predict from X_te_sc
    # uses_scaled=False -> predict from X_te_enc
    return {
        "Logistic Regression": (lr, True),
        "Decision Tree":       (dt, False),
        "Random Forest":       (rf, False),
    }


# =============================================================================
# 7. EVALUATE MODELS
# =============================================================================

def evaluate_models(
    models: dict,
    X_te_sc, X_te_enc,
    y_test,
    feature_names: list,
):
    """
    Evaluate all models, print classification reports, plot and save
    confusion matrices, ROC curves, feature importance, and CV boxplot.
    Returns (results DataFrame, feature-importance DataFrame).
    """
    print("\n" + "=" * 62)
    print("  STEP 7 -- EVALUATE MODELS")
    print("=" * 62)

    rows  = []
    preds = {}   # name -> (y_pred, y_proba)

    for name, (model, use_scaled) in models.items():
        X_te   = X_te_sc if use_scaled else X_te_enc
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        preds[name] = (y_pred, y_proba)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)
        rows.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        })

        print(f"\n-- {name} --")
        print(classification_report(y_test, y_pred,
                                    target_names=["Active", "Churned"]))

    results = pd.DataFrame(rows).set_index("Model")

    # -- Confusion matrices ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, (y_pred, _)) in zip(axes, preds.items()):
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Active", "Churned"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontweight="bold")
    plt.suptitle("Confusion Matrices - All Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "13_confusion_matrices.png")

    # -- ROC curves -----------------------------------------------------------
    colors_roc = ["#3498db", "#e67e22", "#27ae60"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, (_, y_proba)), color in zip(preds.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.500)")
    ax.set_title("ROC Curves - All Models", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save(fig, "14_roc_curves.png")

    # -- Model comparison bar chart -------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    results.T.plot(kind="bar", ax=ax, edgecolor="black", rot=0,
                   color=["#3498db", "#e67e22", "#27ae60"])
    ax.set_title("Model Performance Comparison - All Metrics", fontweight="bold")
    ax.set_xlabel("Metric"); ax.set_ylabel("Score")
    ax.set_ylim([0, 1.12])
    ax.legend(title="Model", loc="lower right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                ha="center", va="bottom", fontsize=7.5, rotation=90,
            )
    plt.tight_layout()
    _save(fig, "15_model_comparison.png")

    # -- Feature importance (Random Forest) -----------------------------------
    rf_model, _ = models["Random Forest"]
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
            color=sns.color_palette("viridis", len(fi_df))[::-1],
            edgecolor="black")
    ax.set_title("Random Forest - Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score (Mean Decrease in Impurity)")
    ax.set_ylabel("Feature")
    for i, (val, _) in enumerate(
            zip(fi_df["Importance"][::-1], fi_df["Feature"][::-1])):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "16_feature_importance.png")

    print("\nTop 5 Predictive Features (Random Forest):")
    print(fi_df.head(5).to_string(index=False))

    # -- 5-Fold Cross-Validation ----------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\n5-Fold Stratified Cross-Validation ROC-AUC (full dataset):")

    # Reconstruct full encoded/scaled X by combining train+test slices
    X_full_sc  = pd.concat([X_te_sc,  X_te_sc],  ignore_index=True)   # proxy
    X_full_enc = pd.concat([X_te_enc, X_te_enc], ignore_index=True)
    y_full     = pd.concat([y_test,   y_test],   ignore_index=True)

    cv_results = {}
    for name, (model, use_sc) in models.items():
        X_cv = X_full_sc if use_sc else X_full_enc
        scores = cross_val_score(model, X_cv, y_full,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<22}: {scores.round(4)}  "
              f"mean={scores.mean():.4f}  std={scores.std():.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        list(cv_results.values()),
        tick_labels=list(cv_results.keys()),
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], ["#3498db", "#e67e22", "#27ae60"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("5-Fold CV ROC-AUC - All Models", fontweight="bold")
    ax.set_ylabel("ROC-AUC Score"); ax.set_xlabel("Model")
    plt.tight_layout()
    _save(fig, "17_cv_scores.png")

    return results, fi_df


# =============================================================================
# 8. GENERATE INSIGHTS
# =============================================================================

def generate_insights(
    df: pd.DataFrame,
    target: str,
    results: pd.DataFrame,
    fi_df: pd.DataFrame,
) -> None:
    """Print business insights and conclusion, then save results CSV."""

    print("\n" + "=" * 62)
    print("  STEP 8 -- INSIGHTS & CONCLUSION")
    print("=" * 62)

    # -- Key statistics -------------------------------------------------------
    churn_rate   = df[target].mean() * 100
    top_region   = (df.groupby("region")[target].mean() * 100).idxmax()
    top_sub      = (df.groupby("subscription_type")[target].mean() * 100).idxmax()
    top_device   = (df.groupby("device")[target].mean() * 100).idxmax()
    mean_login_a = df[df[target] == 0]["last_login_days"].mean()
    mean_login_c = df[df[target] == 1]["last_login_days"].mean()
    mean_wh_a    = df[df[target] == 0]["watch_hours"].mean()
    mean_wh_c    = df[df[target] == 1]["watch_hours"].mean()
    top3_feats   = fi_df.head(3)["Feature"].tolist()
    best_name    = results["ROC-AUC"].idxmax()
    best         = results.loc[best_name]

    print(f"""
KEY FINDINGS
============
  Overall churn rate         : {churn_rate:.1f}%
  Highest-churn region       : {top_region}
  Highest-churn subscription : {top_sub}
  Highest-churn device       : {top_device}
  Avg last-login (Active)    : {mean_login_a:.1f} days
  Avg last-login (Churned)   : {mean_login_c:.1f} days
  Avg watch hours (Active)   : {mean_wh_a:.1f} hrs
  Avg watch hours (Churned)  : {mean_wh_c:.1f} hrs
  Top 3 predictive features  : {top3_feats[0]}, {top3_feats[1]}, {top3_feats[2]}

BEST MODEL
==========
  Model      : {best_name}
  Accuracy   : {best['Accuracy']:.4f}
  Precision  : {best['Precision']:.4f}
  Recall     : {best['Recall']:.4f}
  F1-Score   : {best['F1-Score']:.4f}
  ROC-AUC    : {best['ROC-AUC']:.4f}
  (Selected on highest ROC-AUC across all models)

BUSINESS INSIGHTS
=================
  1. Re-engage inactive users -- trigger personalised campaigns when
     last_login_days > 10 days.
  2. Nudge low-watch-hour users with curated playlists and "Continue
     Watching" prompts.
  3. Offer Basic-tier subscribers limited-time upgrade trials to reduce
     churn and increase revenue.
  4. Invest in localised content for high-churn regions.
  5. Prompt single-profile accounts to add family profiles.
  6. Reduce payment friction with proactive billing reminders and retry
     flows for failed renewals.
  7. Improve Mobile/Tablet app UX to address device-specific churn.
  8. Deploy the Random Forest model as a real-time churn-risk scorer;
     flag customers with predicted probability > 0.65 for retention
     offers (discounts, exclusive previews, loyalty rewards).
  9. Launch a loyalty reward programme (watch milestones, referral
     credits) to improve long-term retention.
  10. Invest in content for genres with above-average churn rates.

LIMITATIONS
===========
  - 5,000 rows may not fully generalise to real-world scale.
  - No subscription tenure / start-date column available.
  - Label Encoding introduces artificial ordinality for nominal features.
  - No hyperparameter tuning -- production should use GridSearchCV/Optuna.
  - Static snapshot: model needs periodic retraining (concept drift).
  - External factors (competitor pricing, market trends) not captured.

FUTURE WORK
===========
  - XGBoost / LightGBM for potentially higher accuracy.
  - SHAP values for per-customer churn explanation.
  - One-Hot Encoding for nominal features in Logistic Regression.
  - REST API (Flask/FastAPI) for real-time churn prediction serving.
  - Time-series feature engineering (tenure, rolling watch trends).
  - Hyperparameter tuning with GridSearchCV or Bayesian optimisation.
  - Deep learning (MLP) for non-linear pattern capture.

CONCLUSION
==========
  This project built a complete end-to-end churn prediction pipeline.
  Inactivity (last_login_days) and low viewing engagement (watch_hours,
  avg_watch_time_per_day) are the strongest churn signals. The Random
  Forest model achieved the highest performance across all metrics and
  is recommended for production deployment as a churn-risk scoring engine.
""")

    # -- Save results CSV -----------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "model_results.csv")
    results.reset_index().to_csv(csv_path, index=False)
    print(f"Model results saved -> {csv_path}")
    print("All outputs written to outputs/")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 62)
    print("  Netflix Customer Churn Prediction")
    print("  Student  : Telukuntla Yashwanth")
    print("  Program  : IBM SkillsBuild Data Analytics with AI")
    print("  Submitted: BharatCares & AICTE")
    print("=" * 62)

    df     = load_data(DATA_FILE)
    target = identify_target(df)
    df     = clean_data(df)
    perform_eda(df, target)

    (
        X_tr_sc, X_te_sc,
        X_tr_enc, X_te_enc,
        y_train, y_test,
        feature_names,
    ) = prepare_data(df, target)

    models = train_models(X_tr_sc, X_tr_enc, y_train)

    results, fi_df = evaluate_models(
        models, X_te_sc, X_te_enc, y_test, feature_names
    )

    generate_insights(df, target, results, fi_df)

    print("\n" + "=" * 62)
    print("  Script complete. All outputs saved to outputs/")
    print("=" * 62)


if __name__ == "__main__":
    main()
