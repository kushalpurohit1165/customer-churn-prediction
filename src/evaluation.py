"""
evaluation.py
-------------
Metric calculation, best-model selection and model-related charts.

Positive class = churn (Churn = Yes = 1). Precision / Recall / F1 are reported for that class,
because finding the customers who WILL leave is the whole purpose of the project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

PALETTE = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444", "#0EA5E9"]


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate one fitted pipeline on the untouched test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def build_metrics_table(trained: dict, X_test, y_test):
    """Evaluate all models. Returns (metrics DataFrame, {model: {"y_pred", "y_proba"}})."""
    rows, details = [], {}
    for name, info in trained.items():
        m = evaluate_model(info["model"], X_test, y_test)
        details[name] = {"y_pred": m.pop("y_pred"), "y_proba": m.pop("y_proba")}
        rows.append({"Model": name, **m, "CV F1 (train)": info["cv_f1"]})
    return pd.DataFrame(rows), details


def select_best_model(metrics: pd.DataFrame) -> str:
    """
    Best model = highest F1-score for the churn class (balance of precision and recall),
    ROC-AUC is the tie-breaker. Accuracy alone is NOT used: with 29% churners a model that
    predicts 'No' for everybody already gets ~71% accuracy.
    """
    ranked = metrics.sort_values(["F1-Score", "ROC-AUC"], ascending=False)
    return str(ranked.iloc[0]["Model"])


def get_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["Stay (No)", "Churn (Yes)"], zero_division=0)


def get_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=["Stay (No)", "Churn (Yes)"],
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(3)


# ----------------------------------------------------------------------------
# Charts (all return a matplotlib Figure so they work in files AND in Streamlit)
# ----------------------------------------------------------------------------
def plot_accuracy_comparison(metrics: pd.DataFrame, best: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = metrics.sort_values("Accuracy")
    colors = ["#F97316" if m == best else "#93C5FD" for m in data["Model"]]
    bars = ax.barh(data["Model"], data["Accuracy"], color=colors)
    for b, v in zip(bars, data["Accuracy"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy Comparison (best model by F1 in orange)", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_prf_comparison(metrics: pd.DataFrame):
    long = metrics.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score"],
                        var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=long, x="Model", y="Score", hue="Metric", palette=PALETTE[:3], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision, Recall and F1-Score (churn class)", fontweight="bold")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper right", ncol=3, frameon=False)
    sns.despine()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14},
                xticklabels=["Stay (No)", "Churn (Yes)"], yticklabels=["Stay (No)", "Churn (Yes)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true, probas: dict, highlight: str = None):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, (name, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        ax.plot(fpr, tpr, lw=3 if name == highlight else 1.6, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves - all models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    sns.despine()
    fig.tight_layout()
    return fig


def get_feature_importance(rf_pipeline, top_n: int = 15) -> pd.DataFrame:
    """Impurity-based importance from the Random Forest (for the one-hot encoded features)."""
    names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = rf_pipeline.named_steps["model"].feature_importances_
    df = pd.DataFrame({"Feature": names, "Importance": imp})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = importance_df.sort_values("Importance")
    ax.barh(data["Feature"], data["Importance"], color="#2563EB")
    ax.set_xlabel("Importance")
    ax.set_title("Top Features - Random Forest", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig
