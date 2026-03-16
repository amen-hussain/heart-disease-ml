"""
eda.py
======
Exploratory Data Analysis — generates and saves plots to reports/figures/.

Analyses
--------
1.  Target class distribution
2.  Numeric feature distributions by target class
3.  Correlation heatmap
4.  Categorical feature vs target bar charts
5.  Age distribution by sex and target
6.  Key clinical feature pair plots
7.  Summary statistics to console
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

FIGURES_DIR = "reports/figures"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[eda] Saved → {path}")


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease", "Heart Disease"]

    axes[0].bar(labels, counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.2)
    axes[0].set_title("Target Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Balance", fontweight="bold")

    fig.suptitle("Heart Disease Prevalence in Dataset", fontsize=14, fontweight="bold")
    _save(fig, "01_target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for cls, color, lbl in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
            subset = df[df["target"] == cls][col].dropna()
            axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=lbl, edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=9)

    # Hide unused subplot
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Feature Distributions by Target Class",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "02_numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cat_labels = {
        "sex":     {0: "Female", 1: "Male"},
        "cp":      {0: "Asymptomatic", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Typical Angina"},
        "fbs":     {0: "FBS ≤ 120", 1: "FBS > 120"},
        "restecg": {0: "Normal", 1: "ST-T Abnorm.", 2: "LV Hypertrophy"},
        "exang":   {0: "No", 1: "Yes"},
        "slope":   {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
        "ca":      {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
        "thal":    {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"},
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
        ct.columns = ["No Disease", "Heart Disease"]
        ct.index = [cat_labels.get(col, {}).get(v, str(v)) for v in ct.index]
        ct.plot(kind="bar", ax=axes[i], color=["#2ecc71", "#e74c3c"],
                edgecolor="white", linewidth=0.8)
        axes[i].set_title(col.upper(), fontweight="bold")
        axes[i].set_ylabel("% within group")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].legend(fontsize=8)

    axes[-1].set_visible(False)
    fig.suptitle("Disease Rate by Categorical Feature",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "04_categorical_vs_target.png")


def plot_age_sex_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age distribution
    for cls, color, lbl in [(0, "#2ecc71", "No Disease"), (1, "#e74c3c", "Disease")]:
        axes[0].hist(df[df["target"] == cls]["age"], bins=15,
                     alpha=0.65, color=color, label=lbl, edgecolor="white")
    axes[0].set_title("Age Distribution by Disease Status", fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Sex × target grouped bar
    sex_target = df.groupby(["sex", "target"]).size().unstack()
    sex_target.index = ["Female", "Male"]
    sex_target.columns = ["No Disease", "Heart Disease"]
    sex_target.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"],
                    edgecolor="white", linewidth=0.8)
    axes[1].set_title("Disease Count by Sex", fontweight="bold")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    _save(fig, "05_age_sex_target.png")


def plot_key_pairs(df: pd.DataFrame) -> None:
    """Scatter plots for the most clinically informative feature pairs."""
    pairs = [
        ("age", "thalach", "Age vs Max Heart Rate"),
        ("trestbps", "chol", "Blood Pressure vs Cholesterol"),
        ("age", "oldpeak", "Age vs ST Depression"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = {0: "#2ecc71", 1: "#e74c3c"}

    for ax, (x, y, title) in zip(axes, pairs):
        for cls in [0, 1]:
            sub = df[df["target"] == cls]
            ax.scatter(sub[x], sub[y], c=colors[cls], alpha=0.5, s=30,
                       label=["No Disease", "Heart Disease"][cls], edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "06_key_feature_pairs.png")


def print_summary_statistics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(" SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts().to_string()}")

    print("\n── Numeric feature stats by target ──")
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in numeric_cols:
        grp = df.groupby("target")[col].agg(["mean", "std"]).round(2)
        grp.index = ["No Disease", "Disease"]
        print(f"\n{col}:\n{grp.to_string()}")

    print("\n── Disease rate by sex ──")
    print(df.groupby("sex")["target"].mean().round(3).rename({0: "Female", 1: "Male"}).to_string())

    print("\n── Disease rate by chest pain type ──")
    print(df.groupby("cp")["target"].mean().round(3).to_string())
    print("=" * 60 + "\n")


# ── Main EDA runner ───────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Run all EDA steps."""
    print("[eda] Running exploratory data analysis …")
    print_summary_statistics(df)
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_vs_target(df)
    plot_age_sex_target(df)
    plot_key_pairs(df)
    print(f"[eda] All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.data_cleaning import clean
    df = clean(load_data())
    run_eda(df)
