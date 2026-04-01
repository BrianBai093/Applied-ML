from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.features import add_domain_features, prepare_feature_frame
from src.utils import ensure_directories, save_json


def load_training_frame() -> pd.DataFrame:
    return pd.read_csv(config.RAW_TRAIN_PATH)


def load_column_descriptions() -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(config.COLUMN_DESCRIPTION_PATH, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error if last_error is not None else RuntimeError("Failed to read column descriptions.")


def identify_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [column for column in frame.columns if column not in categorical_columns]
    return numeric_columns, categorical_columns


def build_splits() -> dict[str, object]:
    ensure_directories()

    raw_frame = load_training_frame()
    features, label = prepare_feature_frame(raw_frame)
    numeric_columns, categorical_columns = identify_feature_types(features)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        label,
        test_size=config.FIRST_SPLIT_TEST_SIZE,
        stratify=label,
        random_state=config.RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.SECOND_SPLIT_TEST_SIZE,
        stratify=y_temp,
        random_state=config.RANDOM_STATE,
    )

    payload = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": features.columns.tolist(),
    }
    joblib.dump(payload, config.SPLITS_ARTIFACT_PATH)

    column_descriptions = load_column_descriptions()
    summary = {
        "raw_train_path": str(config.RAW_TRAIN_PATH),
        "column_description_path": str(config.COLUMN_DESCRIPTION_PATH),
        "n_rows": int(raw_frame.shape[0]),
        "n_columns": int(raw_frame.shape[1]),
        "n_model_features": int(features.shape[1]),
        "target_rate": float(label.mean()),
        "split_sizes": {
            "train": int(X_train.shape[0]),
            "validation": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "feature_types": {
            "numeric": len(numeric_columns),
            "categorical": len(categorical_columns),
        },
        "engineered_features": list(config.ENGINEERED_FEATURES),
        "top_missing_columns": raw_frame.isna().mean().sort_values(ascending=False).head(15).to_dict(),
        "key_column_descriptions": column_descriptions[
            column_descriptions["Row"].isin(
                [
                    config.TARGET_COLUMN,
                    "DAYS_EMPLOYED",
                    "AMT_INCOME_TOTAL",
                    "AMT_CREDIT",
                    "AMT_ANNUITY",
                    "EXT_SOURCE_1",
                    "EXT_SOURCE_2",
                    "EXT_SOURCE_3",
                ]
            )
        ].to_dict(orient="records"),
        "days_employed_365243_count": int((raw_frame["DAYS_EMPLOYED"] == 365243).sum())
        if "DAYS_EMPLOYED" in raw_frame.columns
        else 0,
    }
    save_json(config.DATA_SUMMARY_PATH, summary)
    return payload


def load_splits(force_rebuild: bool = False) -> dict[str, object]:
    if force_rebuild or not config.SPLITS_ARTIFACT_PATH.exists():
        return build_splits()
    return joblib.load(config.SPLITS_ARTIFACT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Home Credit train/validation/test splits.")
    parser.add_argument("--force", action="store_true", help="Rebuild split artifacts even if they exist.")
    args = parser.parse_args()

    payload = load_splits(force_rebuild=args.force)
    print("Prepared splits:")
    print(f"  train={payload['X_train'].shape}")
    print(f"  validation={payload['X_val'].shape}")
    print(f"  test={payload['X_test'].shape}")
    print(f"Saved summary to {config.DATA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
