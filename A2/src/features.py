from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import safe_divide


def apply_basic_cleaning(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if "DAYS_EMPLOYED" in cleaned.columns:
        cleaned["DAYS_EMPLOYED"] = cleaned["DAYS_EMPLOYED"].replace(365243, np.nan)
    return cleaned


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = apply_basic_cleaning(frame)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(enriched.columns):
        enriched["CREDIT_INCOME_RATIO"] = safe_divide(
            enriched["AMT_CREDIT"], enriched["AMT_INCOME_TOTAL"]
        )

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(enriched.columns):
        enriched["ANNUITY_INCOME_RATIO"] = safe_divide(
            enriched["AMT_ANNUITY"], enriched["AMT_INCOME_TOTAL"]
        )

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(enriched.columns):
        enriched["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            enriched["AMT_INCOME_TOTAL"], enriched["CNT_FAM_MEMBERS"]
        )

    ext_source_columns = [
        column
        for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if column in enriched.columns
    ]
    if ext_source_columns:
        enriched["EXT_SOURCE_MEAN"] = enriched[ext_source_columns].mean(axis=1)

    return enriched


def prepare_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    transformed = add_domain_features(frame)
    label = transformed[config.TARGET_COLUMN].copy()
    columns_to_drop = [config.TARGET_COLUMN, *config.ID_COLUMNS]
    columns_to_drop = [column for column in columns_to_drop if column in transformed.columns]
    features = transformed.drop(columns=columns_to_drop)
    return features, label
