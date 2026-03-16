"""
evaluate.py
===========
Comprehensive evaluation of trained models on the held-out test set.

Generates
---------
- Per-model classification report
- Confusion matrices  (plot)
- ROC curves          (plot)
- Precision-Recall curves (plot)
- Feature importance  (plot — for tree-based models)
- Model comparison summary CSV → reports/model_comparison.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

FIGURES_DIR = "reports/figures"
REPORTS_DIR = "reports"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {path}")


# ── Per-model metrics ─────────────────────────────────────────────────────────

def evaluate_all(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate every model and return a summary DataFrame.

    Parameters
    ----------
    models : {name: fitted_pipeline}
    """
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        row = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        }
        rows.append(row)

        print(f"\n{'─'*50}")
        print(f" {name}")
        print(f"{'─'*50}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Disease", "Disease"]))

    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n[evaluate] Model comparison saved → {csv_path}")
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    linewidths=0.5, ax=ax, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_confusion_matrices.png")


def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # type: ignore

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, "08_roc_curves.png")


def plot_precision_recall_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # type: ignore
    baseline = y_test.mean()

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name}  (AP = {ap:.3f})")

    ax.axhline(baseline, color="k", linestyle="--", lw=1.5,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    _save(fig, "09_precision_recall_curves.png")


def plot_feature_importance(
    models: dict,
    feature_names: list,
    top_n: int = 15,
) -> None:
    """Plot feature importances for tree-based models."""
    tree_models = {
        name: m for name, m in models.items()
        if hasattr(m.named_steps["clf"], "feature_importances_")
    }
    if not tree_models:
        return

    for name, model in tree_models.items():
        importances = model.named_steps["clf"].feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in idx]
        top_values   = importances[idx]

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))[::-1]  # type: ignore
        bars = ax.barh(top_features[::-1], top_values[::-1],
                       color=colors, edgecolor="white")
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances\n{name}",
                     fontsize=13, fontweight="bold")
        ax.invert_xaxis()
        for bar, val in zip(bars, top_values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        safe = name.lower().replace(" ", "_")
        _save(fig, f"10_feature_importance_{safe}.png")


def plot_model_comparison_bar(summary: pd.DataFrame) -> None:
    metrics = ["accuracy", "roc_auc", "f1", "avg_precision"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        data = summary.sort_values(metric, ascending=False)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(data)))[::-1]  # type: ignore
        bars = ax.bar(data["model"], data[metric], color=colors, edgecolor="white")
        ax.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        ax.set_ylim(max(0, data[metric].min() - 0.05), min(1.05, data[metric].max() + 0.05))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Model Performance Comparison — Test Set",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "11_model_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    print("\n[evaluate] Evaluating models on test set …")
    summary = evaluate_all(models, X_test, y_test)

    print("\n[evaluate] Generating evaluation plots …")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_feature_importance(models, list(X_test.columns))
    plot_model_comparison_bar(summary)

    print("\n[evaluate] Summary:\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    from src.data_loader import load_data
    from src.data_cleaning import clean, get_feature_target_split

    df = clean(load_data())
    X, y = get_feature_target_split(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Load saved models
    import glob, os
    model_files = glob.glob("models/*.joblib")
    models = {}
    for path in model_files:
        name = os.path.basename(path).replace(".joblib", "").replace("_", " ").title()
        if "best" not in name.lower():
            models[name] = joblib.load(path)
    run_evaluation(models, X_test, y_test)
