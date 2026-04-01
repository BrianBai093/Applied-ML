from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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


def build_xgb_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )


def _plot_logloss(evals_result: dict[str, Any], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(evals_result["validation_0"]["logloss"], label="train")
    axis.plot(evals_result["validation_1"]["logloss"], label="validation")
    axis.set_title("XGBoost Train vs Validation Logloss")
    axis.set_xlabel("Boosting rounds")
    axis.set_ylabel("Logloss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_learning_rate_comparison(results: pd.DataFrame, output_path: Path) -> None:
    plot_frame = results.sort_values("learning_rate").copy()
    x_positions = range(len(plot_frame))

    figure, axis = plt.subplots(figsize=(8, 5))
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
    axis.set_title("XGBoost Learning Rate Comparison")
    axis.set_xlabel("Learning Rate")
    axis.set_ylabel("Validation Score")
    axis.set_xticks(list(x_positions), [str(value) for value in plot_frame["learning_rate"]])
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_feature_importance(feature_names: list[str], importances: Any, output_path: Path) -> None:
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .sort_values("importance", ascending=True)
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(importance_frame["feature"], importance_frame["importance"])
    axis.set_title("XGBoost Feature Importance (Top 20)")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _fit_xgb(model: XGBClassifier, X_train: Any, y_train: pd.Series, eval_set: list[tuple[Any, pd.Series]]) -> tuple[XGBClassifier, float]:
    fit_kwargs = {
        "eval_set": eval_set,
        "verbose": False,
    }
    try:
        return timed_call(
            model.fit,
            X_train,
            y_train,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
            **fit_kwargs,
        )
    except TypeError:
        model.set_params(early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS)
        return timed_call(model.fit, X_train, y_train, **fit_kwargs)


def _make_run_name(stage: str, params: dict[str, Any]) -> str:
    pieces = [stage]
    for key in sorted(params):
        value = str(params[key]).replace(" ", "")
        pieces.append(f"{key}-{value}")
    return "__".join(pieces).replace("/", "_")


def _train_candidate_from_preprocessed(
    X_train_transformed: Any,
    y_train: pd.Series,
    X_val_transformed: Any,
    y_val: pd.Series,
    feature_names: list[str],
    params: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_name = _make_run_name(stage, params)
    model = XGBClassifier(**params)
    model, train_time = _fit_xgb(
        model,
        X_train_transformed,
        y_train,
        eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
    )

    validation_probabilities = model.predict_proba(X_val_transformed)[:, 1]
    metrics = compute_classification_metrics(y_val, validation_probabilities)
    evals_result = model.evals_result()

    history_payload = {
        "run_name": run_name,
        "stage": stage,
        "params": params,
        "evals_result": evals_result,
        "feature_count": len(feature_names),
    }
    save_json(config.XGB_HISTORY_DIR / f"{run_name}.json", history_payload)

    best_iteration = getattr(model, "best_iteration", None)
    boosting_rounds = (best_iteration + 1) if best_iteration is not None else params["n_estimators"]
    record = {
        "model": "XGBoost",
        "stage": stage,
        "run_name": run_name,
        **params,
        **metrics,
        "train_time_seconds": train_time,
        "best_iteration": best_iteration,
        "selected_boosting_rounds": boosting_rounds,
        "feature_count": len(feature_names),
    }
    artifacts = {
        "model": model,
        "feature_names": feature_names,
        "params": params,
        "evals_result": evals_result,
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
    preprocessor = build_xgb_preprocessor(numeric_columns, categorical_columns)
    X_train_transformed = as_float32(preprocessor.fit_transform(X_train))
    X_val_transformed = as_float32(preprocessor.transform(X_val))
    feature_names = preprocessor.get_feature_names_out().tolist()
    record, artifacts = _train_candidate_from_preprocessed(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        feature_names,
        params,
        stage,
    )
    artifacts["preprocessor"] = preprocessor
    return record, artifacts


def _extract_final_params(best_artifacts: dict[str, Any]) -> dict[str, Any]:
    final_params = dict(best_artifacts["params"])
    selected_rounds = best_artifacts["record"]["selected_boosting_rounds"]
    final_params["n_estimators"] = int(selected_rounds)
    final_params.pop("early_stopping_rounds", None)
    return final_params


def run_xgb_experiments(force_rebuild_splits: bool = False) -> dict[str, Any]:
    ensure_directories()
    splits = load_splits(force_rebuild=force_rebuild_splits)

    X_train = splits["X_train"]
    X_val = splits["X_val"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    numeric_columns = splits["numeric_columns"]
    categorical_columns = splits["categorical_columns"]
    validation_preprocessor = build_xgb_preprocessor(numeric_columns, categorical_columns)
    X_train_transformed = as_float32(validation_preprocessor.fit_transform(X_train))
    X_val_transformed = as_float32(validation_preprocessor.transform(X_val))
    feature_names = validation_preprocessor.get_feature_names_out().tolist()

    records: list[dict[str, Any]] = []
    artifacts_by_run: dict[str, dict[str, Any]] = {}

    baseline_record, baseline_artifacts = _train_candidate_from_preprocessed(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        feature_names,
        params=dict(config.XGB_BASE_PARAMS),
        stage="baseline",
    )
    baseline_artifacts["preprocessor"] = validation_preprocessor
    records.append(baseline_record)
    artifacts_by_run[baseline_record["run_name"]] = baseline_artifacts

    learning_rate_records = []
    for learning_rate in config.XGB_LEARNING_RATES:
        params = dict(config.XGB_BASE_PARAMS)
        params["learning_rate"] = learning_rate
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            feature_names,
            params=params,
            stage="learning_rate",
        )
        artifacts["preprocessor"] = validation_preprocessor
        learning_rate_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    learning_rate_frame = pd.DataFrame(learning_rate_records)
    best_learning_rate = select_best_row(learning_rate_frame)["learning_rate"]

    complexity_records = []
    for max_depth, subsample in product(config.XGB_MAX_DEPTHS, config.XGB_SUBSAMPLES):
        params = dict(config.XGB_BASE_PARAMS)
        params["learning_rate"] = best_learning_rate
        params["max_depth"] = max_depth
        params["subsample"] = subsample
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            feature_names,
            params=params,
            stage="complexity",
        )
        artifacts["preprocessor"] = validation_preprocessor
        complexity_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    complexity_frame = pd.DataFrame(complexity_records)
    best_complexity = select_best_row(complexity_frame)

    regularization_records = []
    for reg_alpha, reg_lambda in product(config.XGB_REG_ALPHAS, config.XGB_REG_LAMBDAS):
        params = dict(config.XGB_BASE_PARAMS)
        params["learning_rate"] = best_learning_rate
        params["max_depth"] = int(best_complexity["max_depth"])
        params["subsample"] = float(best_complexity["subsample"])
        params["reg_alpha"] = reg_alpha
        params["reg_lambda"] = reg_lambda
        record, artifacts = _train_candidate_from_preprocessed(
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            feature_names,
            params=params,
            stage="regularization",
        )
        artifacts["preprocessor"] = validation_preprocessor
        regularization_records.append(record)
        records.append(record)
        artifacts_by_run[record["run_name"]] = artifacts

    validation_frame = pd.DataFrame(records).sort_values(
        by=[config.SELECTION_PRIMARY_METRIC, config.SELECTION_SECONDARY_METRIC],
        ascending=False,
    )
    save_dataframe(validation_frame, config.XGB_VALIDATION_RESULTS_PATH)

    best_validation_row = select_best_row(validation_frame)
    best_artifacts = artifacts_by_run[best_validation_row["run_name"]]

    _plot_logloss(best_artifacts["evals_result"], config.XGB_TRAIN_VAL_LOGLOSS_FIGURE)
    _plot_learning_rate_comparison(learning_rate_frame, config.XGB_LEARNING_RATE_FIGURE)

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    final_preprocessor = build_xgb_preprocessor(numeric_columns, categorical_columns)
    X_train_val_transformed = as_float32(final_preprocessor.fit_transform(X_train_val))
    X_test_transformed = as_float32(final_preprocessor.transform(X_test))

    final_params = _extract_final_params(best_artifacts)
    final_model = XGBClassifier(**final_params)
    final_model, final_train_time = timed_call(final_model.fit, X_train_val_transformed, y_train_val, verbose=False)
    test_probabilities = final_model.predict_proba(X_test_transformed)[:, 1]
    final_test_metrics = compute_classification_metrics(y_test, test_probabilities)
    final_feature_names = final_preprocessor.get_feature_names_out().tolist()
    _plot_feature_importance(final_feature_names, final_model.feature_importances_, config.XGB_FEATURE_IMPORTANCE_FIGURE)

    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "feature_names": final_feature_names,
            "best_validation_record": best_validation_row.to_dict(),
        },
        config.XGB_MODEL_PATH,
    )

    summary = {
        "model": "XGBoost",
        "best_config": final_params,
        "best_validation_metrics": {
            key: best_validation_row[key]
            for key in ["accuracy", "precision", "recall", "f1", "auc_pr"]
        },
        "final_test_metrics": final_test_metrics,
        "final_train_time_seconds": final_train_time,
        "validation_results_path": str(config.XGB_VALIDATION_RESULTS_PATH),
        "figures": {
            "train_val_logloss": str(config.XGB_TRAIN_VAL_LOGLOSS_FIGURE),
            "learning_rate_comparison": str(config.XGB_LEARNING_RATE_FIGURE),
            "feature_importance": str(config.XGB_FEATURE_IMPORTANCE_FIGURE),
        },
        "notes": "early stopping used during validation search; XGBoost ran on GPU",
        "best_validation_run": best_validation_row["run_name"],
        "best_validation_stage": best_validation_row["stage"],
        "best_validation_description": parameter_string(final_params),
    }
    save_json(config.XGB_SUMMARY_PATH, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost experiments.")
    parser.add_argument("--force-rebuild-splits", action="store_true", help="Rebuild train/validation/test artifacts.")
    args = parser.parse_args()

    summary = run_xgb_experiments(force_rebuild_splits=args.force_rebuild_splits)
    print("XGBoost experiments complete.")
    print(summary["best_validation_description"])
    print(summary["final_test_metrics"])


if __name__ == "__main__":
    main()
