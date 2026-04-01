from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_prep import load_splits
from src.evaluate import compute_classification_metrics, select_best_row
from src.utils import (
    as_float32,
    ensure_directories,
    make_one_hot_encoder,
    parameter_string,
    save_dataframe,
    save_json,
    timed_call,
)


def build_mlp_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        sparse_threshold=0.0,
    )


def _plot_loss_curve(loss_curve: list[float], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(loss_curve, linewidth=2)
    axis.set_title("MLP Training Loss Curve")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_architecture_comparison(results: pd.DataFrame, output_path: Path) -> None:
    plot_frame = results.copy()
    plot_frame["architecture_label"] = plot_frame["hidden_layer_sizes"].astype(str)
    x_positions = range(len(plot_frame))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        [x - 0.2 for x in x_positions],
        plot_frame["f1"],
        width=0.4,
        label="Validation F1",
    )
    axis.bar(
        [x + 0.2 for x in x_positions],
        plot_frame["auc_pr"],
        width=0.4,
        label="Validation AUC-PR",
    )
    axis.set_title("MLP Architecture Comparison")
    axis.set_xlabel("Hidden Layer Sizes")
    axis.set_ylabel("Validation Score")
    axis.set_xticks(list(x_positions), plot_frame["architecture_label"])
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _make_run_name(stage: str, params: dict[str, Any]) -> str:
    pieces = [stage]
    for key in sorted(params):
        pieces.append(f"{key}-{str(params[key]).replace(' ', '')}")
    return "__".join(pieces).replace("/", "_")


def _train_candidate_from_preprocessed(
    X_train_transformed: Any,
    y_train: pd.Series,
    X_val_transformed: Any,
    y_val: pd.Series,
    params: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_name = _make_run_name(stage, params)
    model = MLPClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model, train_time = timed_call(model.fit, X_train_transformed, y_train)

    validation_probabilities = model.predict_proba(X_val_transformed)[:, 1]
    metrics = compute_classification_metrics(y_val, validation_probabilities)
    history_payload = {
        "run_name": run_name,
        "stage": stage,
        "params": params,
        "loss_curve": model.loss_curve_,
        "n_iter": model.n_iter_,
    }
    save_json(config.MLP_HISTORY_DIR / f"{run_name}.json", history_payload)

    record = {
        "model": "MLP",
        "stage": stage,
        "run_name": run_name,
        **params,
        **metrics,
        "train_time_seconds": train_time,
        "n_iter": model.n_iter_,
        "loss_curve_length": len(model.loss_curve_),
    }
    artifacts = {
        "model": model,
        "params": params,
        "loss_curve": model.loss_curve_,
        "record": record,
    }
    return record, artifacts


def train_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preprocessor = build_mlp_preprocessor(numeric_columns, categorical_columns)
    X_train_transformed = as_float32(preprocessor.fit_transform(X_train))
    X_val_transformed = as_float32(preprocessor.transform(X_val))
    record, artifacts = _train_candidate_from_preprocessed(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        params,
        stage,
    )
    artifacts["preprocessor"] = preprocessor
    return record, artifacts


def run_mlp_experiments(force_rebuild_splits: bool = False) -> dict[str, Any]:
    ensure_directories()
    splits = load_splits(force_rebuild=force_rebuild_splits)

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    numeric_columns = splits["numeric_columns"]
    categorical_columns = splits["categorical_columns"]
    validation_preprocessor = build_mlp_preprocessor(numeric_columns, categorical_columns)
    X_train_transformed = as_float32(validation_preprocessor.fit_transform(X_train))
    X_val_transformed = as_float32(validation_preprocessor.transform(X_val))

    records: list[dict[str, Any]] = []
    artifacts_by_run: dict[str, dict[str, Any]] = {}

    baseline_record, baseline_artifacts = _train_candidate_from_preprocessed(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        params=dict(config.MLP_BASE_PARAMS),
        stage="baseline",
    )
    baseline_artifacts["preprocessor"] = validation_preprocessor
    records.append(baseline_record)
    artifacts_by_run[baseline_record["run_name"]] = baseline_artifacts

    architecture_records = []
    for hidden_layer_sizes in config.MLP_ARCHITECTURES:
        params = dict(config.MLP_BASE_PARAMS)
        params["hidden_layer_sizes"] = hidden_layer_sizes
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            params=params,
            stage="architecture",
        )
        artifacts["preprocessor"] = validation_preprocessor
        architecture_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    architecture_frame = pd.DataFrame(architecture_records)
    best_architecture_row = select_best_row(architecture_frame)

    activation_records = []
    for activation in config.MLP_ACTIVATIONS:
        params = dict(config.MLP_BASE_PARAMS)
        params["hidden_layer_sizes"] = best_architecture_row["hidden_layer_sizes"]
        params["activation"] = activation
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            params=params,
            stage="activation",
        )
        artifacts["preprocessor"] = validation_preprocessor
        activation_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    activation_frame = pd.DataFrame(activation_records)
    best_activation_row = select_best_row(activation_frame)

    learning_rate_records = []
    for learning_rate_init in config.MLP_LEARNING_RATES:
        params = dict(config.MLP_BASE_PARAMS)
        params["hidden_layer_sizes"] = best_architecture_row["hidden_layer_sizes"]
        params["activation"] = best_activation_row["activation"]
        params["learning_rate_init"] = learning_rate_init
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            params=params,
            stage="learning_rate",
        )
        artifacts["preprocessor"] = validation_preprocessor
        learning_rate_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    learning_rate_frame = pd.DataFrame(learning_rate_records)
    best_learning_rate_row = select_best_row(learning_rate_frame)

    max_iter_records = []
    for max_iter in config.MLP_MAX_ITERS:
        params = dict(config.MLP_BASE_PARAMS)
        params["hidden_layer_sizes"] = best_architecture_row["hidden_layer_sizes"]
        params["activation"] = best_activation_row["activation"]
        params["learning_rate_init"] = float(best_learning_rate_row["learning_rate_init"])
        params["max_iter"] = max_iter
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            params=params,
            stage="max_iter",
        )
        artifacts["preprocessor"] = validation_preprocessor
        max_iter_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    validation_frame = pd.DataFrame(records).sort_values(
        by=[config.SELECTION_PRIMARY_METRIC, config.SELECTION_SECONDARY_METRIC],
        ascending=False,
    )
    save_dataframe(validation_frame, config.MLP_VALIDATION_RESULTS_PATH)

    best_validation_row = select_best_row(validation_frame)
    best_artifacts = artifacts_by_run[best_validation_row["run_name"]]

    _plot_loss_curve(best_artifacts["loss_curve"], config.MLP_LOSS_CURVE_FIGURE)
    _plot_architecture_comparison(architecture_frame, config.MLP_ARCHITECTURE_FIGURE)

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    final_preprocessor = build_mlp_preprocessor(numeric_columns, categorical_columns)
    X_train_val_transformed = as_float32(final_preprocessor.fit_transform(X_train_val))
    X_test_transformed = as_float32(final_preprocessor.transform(X_test))

    final_params = dict(best_artifacts["params"])
    final_model = MLPClassifier(**final_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_model, final_train_time = timed_call(final_model.fit, X_train_val_transformed, y_train_val)

    test_probabilities = final_model.predict_proba(X_test_transformed)[:, 1]
    final_test_metrics = compute_classification_metrics(y_test, test_probabilities)

    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "best_validation_record": best_validation_row.to_dict(),
        },
        config.MLP_MODEL_PATH,
    )

    summary = {
        "model": "MLP",
        "best_config": final_params,
        "best_validation_metrics": {
            key: best_validation_row[key]
            for key in ["accuracy", "precision", "recall", "f1", "auc_pr"]
        },
        "final_test_metrics": final_test_metrics,
        "final_train_time_seconds": final_train_time,
        "validation_results_path": str(config.MLP_VALIDATION_RESULTS_PATH),
        "figures": {
            "loss_curve": str(config.MLP_LOSS_CURVE_FIGURE),
            "architecture_comparison": str(config.MLP_ARCHITECTURE_FIGURE),
        },
        "notes": "scaled inputs with dense one-hot encoding",
        "best_validation_run": best_validation_row["run_name"],
        "best_validation_stage": best_validation_row["stage"],
        "best_validation_description": parameter_string(final_params),
    }
    save_json(config.MLP_SUMMARY_PATH, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate MLP experiments.")
    parser.add_argument("--force-rebuild-splits", action="store_true", help="Rebuild train/validation/test artifacts.")
    args = parser.parse_args()

    summary = run_mlp_experiments(force_rebuild_splits=args.force_rebuild_splits)
    print("MLP experiments complete.")
    print(summary["best_validation_description"])
    print(summary["final_test_metrics"])


if __name__ == "__main__":
    main()
