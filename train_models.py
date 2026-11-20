"""
train_models.py
---------------
Run this ONCE before starting the dashboard:    python train_models.py

Steps: load -> inspect -> clean -> stratified split -> tune + train 6 models ->
evaluate on the test set -> pick the best model -> save models, metrics and graphs.
Nothing here is hard-coded: every number comes from the real CSV and the real models.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw charts into files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd

from src.eda import save_all_eda_plots
from src.evaluation import (build_metrics_table, get_classification_report, get_feature_importance,
                            plot_accuracy_comparison, plot_confusion_matrix, plot_feature_importance,
                            plot_prf_comparison, plot_roc_curves, select_best_model)
from src.model_training import save_artifact, train_all_models
from src.preprocessing import clean_data, inspect_data, load_data, split_data

BASE = Path(__file__).parent
CSV_PATH = BASE / "telco_customer_churn_synthetic.csv"
MODELS_DIR = BASE / "models"
GRAPHS_DIR = BASE / "outputs" / "graphs"
REPORTS_DIR = BASE / "outputs" / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    start = time.time()
    for d in (MODELS_DIR, GRAPHS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("1. Loading and inspecting data")
    raw = load_data(CSV_PATH)
    info = inspect_data(raw)
    print(f"   shape = {raw.shape}, missing = {info['total_missing']}, "
          f"duplicate rows = {info['duplicate_rows']}, classes = {info['class_counts']}")

    print("2. Cleaning data")
    df, clean_report = clean_data(raw)
    print(f"   rows before = {clean_report['rows_before']}, after = {clean_report['rows_after']}")

    print("3. Stratified train/test split (customerID excluded)")
    X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)
    print(f"   train = {len(X_train)}, test = {len(X_test)}, "
          f"churn share train = {y_train.mean():.3f}, test = {y_test.mean():.3f}")

    print("4. Training models (GridSearchCV on training data only)")
    trained = train_all_models(X_train, y_train, random_state=RANDOM_STATE)

    print("5. Evaluating on the test set")
    metrics, details = build_metrics_table(trained, X_test, y_test)
    best = select_best_model(metrics)
    metrics["Best"] = metrics["Model"] == best
    metrics = metrics.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    print(metrics.round(3).to_string(index=False))
    print(f"   BEST MODEL (highest F1, ROC-AUC tie-break): {best}")

    print("6. Saving models, metrics and graphs")
    save_artifact(trained[best]["model"], MODELS_DIR / "best_model.joblib")
    save_artifact({n: t["model"] for n, t in trained.items()}, MODELS_DIR / "all_models.joblib")

    metrics.round(4).to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    reports = {n: get_classification_report(y_test, d["y_pred"]) for n, d in details.items()}
    with open(REPORTS_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for n, rep in reports.items():
            f.write(f"=== {n} ===\n{rep}\n")

    # Everything the dashboard needs, saved so the app never re-trains or invents numbers
    save_artifact({
        "y_test": y_test.to_numpy(),
        "y_pred": {n: d["y_pred"] for n, d in details.items()},
        "y_proba": {n: d["y_proba"] for n, d in details.items()},
        "test_index": X_test.index.to_numpy(),
    }, MODELS_DIR / "test_results.joblib")

    importance = get_feature_importance(trained["Random Forest"]["model"], top_n=15)
    importance.to_csv(REPORTS_DIR / "feature_importance_random_forest.csv", index=False)

    summary = {
        "dataset": {"file": CSV_PATH.name, "synthetic": True, **info},
        "cleaning": clean_report,
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True,
                  "train_rows": len(X_train), "test_rows": len(X_test)},
        "best_model": best,
        "best_params": {n: t["best_params"] for n, t in trained.items()},
        "selection_rule": "Highest F1-score for the churn class; ROC-AUC used as tie-breaker.",
        "training_seconds": round(time.time() - start, 1),
    }
    with open(REPORTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_artifact(summary, MODELS_DIR / "run_summary.joblib")

    save_all_eda_plots(df, GRAPHS_DIR)
    figs = {
        "11_accuracy_comparison": plot_accuracy_comparison(metrics, best),
        "12_precision_recall_f1_comparison": plot_prf_comparison(metrics),
        "13_confusion_matrix_best_model": plot_confusion_matrix(y_test, details[best]["y_pred"],
                                                                f"Confusion Matrix - {best}"),
        "14_roc_curves": plot_roc_curves(y_test, {n: d["y_proba"] for n, d in details.items()}, best),
        "15_feature_importance": plot_feature_importance(importance),
    }
    for name, fig in figs.items():
        fig.savefig(GRAPHS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nDone in {time.time() - start:.1f} s. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
