from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.utils import ensure_directories, save_dataframe, save_json


def compute_classification_metrics(y_true: pd.Series, y_prob: Any, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }


def select_best_row(frame: pd.DataFrame) -> pd.Series:
    sorted_frame = frame.sort_values(
        by=[
            config.SELECTION_PRIMARY_METRIC,
            config.SELECTION_SECONDARY_METRIC,
            "accuracy",
            "precision",
            "recall",
        ],
        ascending=False,
    ).reset_index(drop=True)
    return sorted_frame.iloc[0]


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_final_metrics(final_metrics: pd.DataFrame) -> None:
    metric_columns = ["accuracy", "precision", "recall", "f1", "auc_pr"]
    tidy = final_metrics.melt(id_vars="model", value_vars=metric_columns, var_name="metric", value_name="value")

    figure, axis = plt.subplots(figsize=(10, 6))
    for model_name, subset in tidy.groupby("model"):
        axis.plot(subset["metric"], subset["value"], marker="o", linewidth=2, label=model_name)

    axis.set_title("Final Test Metrics Comparison")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.FINAL_METRICS_FIGURE, dpi=300)
    plt.close(figure)


def build_final_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()

    xgb_summary = _load_summary(config.XGB_SUMMARY_PATH)
    mlp_summary = _load_summary(config.MLP_SUMMARY_PATH)

    final_metrics = pd.DataFrame(
        [
            {"model": "XGBoost", **xgb_summary["final_test_metrics"]},
            {"model": "MLP", **mlp_summary["final_test_metrics"]},
        ]
    )
    training_cost = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "best_config": json.dumps(xgb_summary["best_config"], ensure_ascii=False),
                "train_time_seconds": xgb_summary["final_train_time_seconds"],
                "notes": xgb_summary["notes"],
            },
            {
                "model": "MLP",
                "best_config": json.dumps(mlp_summary["best_config"], ensure_ascii=False),
                "train_time_seconds": mlp_summary["final_train_time_seconds"],
                "notes": mlp_summary["notes"],
            },
        ]
    )

    save_dataframe(final_metrics, config.FINAL_TEST_METRICS_PATH)
    save_dataframe(training_cost, config.TRAINING_TIME_SUMMARY_PATH)
    save_json(
        config.BEST_CONFIGS_PATH,
        {
            "xgboost": xgb_summary["best_config"],
            "mlp": mlp_summary["best_config"],
        },
    )
    _plot_final_metrics(final_metrics)
    return final_metrics, training_cost


def main() -> None:
    final_metrics, training_cost = build_final_reports()
    print(final_metrics.to_string(index=False))
    print()
    print(training_cost.to_string(index=False))


if __name__ == "__main__":
    main()
