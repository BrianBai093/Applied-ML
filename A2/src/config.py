from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "report"

TARGET_COLUMN = "TARGET"
ID_COLUMNS = ("SK_ID_CURR",)
RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
FIRST_SPLIT_TEST_SIZE = 0.30
SECOND_SPLIT_TEST_SIZE = 0.50

SELECTION_PRIMARY_METRIC = "auc_pr"
SELECTION_SECONDARY_METRIC = "f1"


def _resolve_existing_path(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


RAW_TRAIN_PATH = _resolve_existing_path(
    (
        DATA_DIR / "raw" / "application_train.csv",
        DATA_DIR / "application_train.csv",
    )
)
RAW_TEST_PATH = _resolve_existing_path(
    (
        DATA_DIR / "raw" / "application_test.csv",
        DATA_DIR / "application_test.csv",
    )
)
COLUMN_DESCRIPTION_PATH = _resolve_existing_path(
    (
        DATA_DIR / "raw" / "HomeCredit_columns_description.csv",
        DATA_DIR / "HomeCredit_columns_description.csv",
    )
)

SPLITS_ARTIFACT_PATH = PROCESSED_DIR / "modeling_splits.joblib"
DATA_SUMMARY_PATH = PROCESSED_DIR / "data_summary.json"

XGB_VALIDATION_RESULTS_PATH = METRICS_DIR / "xgb_validation_results.csv"
MLP_VALIDATION_RESULTS_PATH = METRICS_DIR / "mlp_validation_results.csv"
FINAL_TEST_METRICS_PATH = METRICS_DIR / "final_test_metrics.csv"
TRAINING_TIME_SUMMARY_PATH = METRICS_DIR / "training_time_summary.csv"
BEST_CONFIGS_PATH = METRICS_DIR / "best_configs.json"
XGB_SUMMARY_PATH = METRICS_DIR / "xgb_summary.json"
MLP_SUMMARY_PATH = METRICS_DIR / "mlp_summary.json"
XGB_HISTORY_DIR = METRICS_DIR / "xgb_histories"
MLP_HISTORY_DIR = METRICS_DIR / "mlp_histories"

XGB_MODEL_PATH = MODELS_DIR / "xgb_best_model.joblib"
MLP_MODEL_PATH = MODELS_DIR / "mlp_best_model.joblib"

XGB_TRAIN_VAL_LOGLOSS_FIGURE = FIGURES_DIR / "xgb_train_val_logloss.png"
XGB_LEARNING_RATE_FIGURE = FIGURES_DIR / "xgb_learning_rate_comparison.png"
XGB_FEATURE_IMPORTANCE_FIGURE = FIGURES_DIR / "xgb_feature_importance.png"
MLP_LOSS_CURVE_FIGURE = FIGURES_DIR / "mlp_loss_curve.png"
MLP_ARCHITECTURE_FIGURE = FIGURES_DIR / "mlp_architecture_comparison.png"
FINAL_METRICS_FIGURE = FIGURES_DIR / "final_metrics_comparison.png"

ALL_REQUIRED_DIRS = (
    PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    REPORT_DIR,
    XGB_HISTORY_DIR,
    MLP_HISTORY_DIR,
)

ENGINEERED_FEATURES = (
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "EXT_SOURCE_MEAN",
)

XGB_BASE_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda",
}
XGB_LEARNING_RATES = [0.01, 0.1, 0.3]
XGB_MAX_DEPTHS = [3, 5, 7]
XGB_SUBSAMPLES = [0.6, 0.8, 1.0]
XGB_REG_ALPHAS = [0.0, 0.1, 1.0]
XGB_REG_LAMBDAS = [1.0, 5.0, 10.0]
XGB_EARLY_STOPPING_ROUNDS = 50

MLP_BASE_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "learning_rate_init": 0.001,
    "max_iter": 50,
    "alpha": 1e-4,
    "batch_size": 512,
    "random_state": RANDOM_STATE,
}
MLP_ARCHITECTURES = [
    (64,),
    (128, 64),
    (256, 128, 64),
]
MLP_ACTIVATIONS = ["relu", "tanh"]
MLP_LEARNING_RATES = [0.001, 0.01, 0.1]
MLP_MAX_ITERS = [30, 50, 80]
