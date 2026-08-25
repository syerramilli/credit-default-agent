"""Sub-agent specs for the ML pipeline. Each sub-agent gets a narrow tool
surface and a system prompt that repeats the one hard rule of this project:
schemas, dtypes, aggregate stats, and small samples only -- never raw data.
"""

from credit_agent.config import SUBAGENT_MODEL
from credit_agent.tools_data import (
    get_categorical_breakdown,
    get_correlation_matrix,
    get_missing_report,
    get_sample_rows,
    get_schema,
    get_summary_stats,
    load_dataset,
)
from credit_agent.tools_eval import compare_models, evaluate_model
from credit_agent.tools_features import (
    drop_columns,
    one_hot_encode,
    scale_numeric,
    train_test_split_dataset,
)
from credit_agent.tools_model import list_trained_models, train_classifier, tune_hyperparameters

_NO_RAW_DATA_RULE = (
    "Hard rule: you must never try to dump or reason over raw row-level data. "
    "Work only from schemas, dtypes, aggregate statistics (means, quantiles, "
    "correlations, value counts), and -- only if genuinely needed -- a single "
    "small capped sample. This keeps token cost bounded. If a tool result "
    "looks too large, narrow your request (fewer columns) instead of asking "
    "for more rows."
)

data_profiler = {
    "name": "data-profiler",
    "description": (
        "Loads the dataset and reports on its structure: schema, dtypes, "
        "missingness, summary statistics, categorical breakdowns, "
        "correlations, and small samples. Call this first, before any "
        "feature engineering or modeling."
    ),
    "system_prompt": (
        "You are a data profiling specialist. Your job is to characterize a "
        "registered dataset for teammates who will engineer features and "
        "train models on it, WITHOUT ever passing them or reasoning over raw "
        "rows yourself beyond a small sample. Report shape, dtypes, missing "
        "values, distributions, and correlations relevant to the target "
        "column. Summarize your findings clearly in your final response -- "
        "the caller cannot see your intermediate tool calls. " + _NO_RAW_DATA_RULE
    ),
    "tools": [
        load_dataset,
        get_schema,
        get_missing_report,
        get_summary_stats,
        get_categorical_breakdown,
        get_correlation_matrix,
        get_sample_rows,
    ],
    "model": SUBAGENT_MODEL,
}

feature_engineer = {
    "name": "feature-engineer",
    "description": (
        "Applies feature engineering (encoding, scaling, dropping columns) "
        "and produces the stratified train/test split used for modeling. "
        "Call this after data-profiler and before model-trainer."
    ),
    "system_prompt": (
        "You are a feature engineering specialist. Given a registered raw "
        "dataset and the profiling findings from a teammate, transform it "
        "into a model-ready dataset (encode categoricals, scale numerics as "
        "appropriate, drop identifier columns) and produce a stratified "
        "train/test split. Always finish by calling train_test_split_dataset "
        "and report the resulting dataset names, shapes, and class balance "
        "in your final response so the caller can hand them to model "
        "training. " + _NO_RAW_DATA_RULE
    ),
    "tools": [
        get_schema,
        one_hot_encode,
        scale_numeric,
        drop_columns,
        train_test_split_dataset,
    ],
    "model": SUBAGENT_MODEL,
}

model_trainer = {
    "name": "model-trainer",
    "description": (
        "Trains one or more classifiers on a prepared training dataset and "
        "reports cross-validated metrics. Call this after feature-engineer."
    ),
    "system_prompt": (
        "You are a model training specialist. Given a registered training "
        "dataset name and target column, train the classifier(s) you're "
        "asked for (logistic_regression, random_forest, gradient_boosting), "
        "using train_classifier, which handles cross-validation for you. If "
        "asked to tune or optimize hyperparameters, use tune_hyperparameters "
        "instead, which runs an Optuna search and registers the best model "
        "it finds -- start with a modest n_trials (10-20) unless told "
        "otherwise, since each trial is a full cross-validation fit. Report "
        "the model names and their cross-validated metrics (and, for tuned "
        "models, the best hyperparameters found) in your final response. "
        + _NO_RAW_DATA_RULE
    ),
    "tools": [train_classifier, tune_hyperparameters, list_trained_models],
    "model": SUBAGENT_MODEL,
}

evaluator = {
    "name": "evaluator",
    "description": (
        "Evaluates trained model(s) against the held-out test set and "
        "reports classification metrics, a confusion matrix, and top "
        "feature importances. Call this last, and use it to recommend a "
        "winning model."
    ),
    "system_prompt": (
        "You are a model evaluation specialist. Given registered model "
        "name(s) and a test dataset name, evaluate them with evaluate_model "
        "or compare_models. Report the metrics clearly -- under class "
        "imbalance, treat PR AUC (average precision) as the primary ranking "
        "metric and ROC AUC as secondary. Call out the top feature "
        "importances (signed coefficients for logistic regression, "
        "mean-absolute TreeSHAP values for random forest / XGBoost -- note "
        "these two are not on the same scale and shouldn't be compared "
        "numerically against each other, only within a model), and state "
        "which model you'd recommend and why. " + _NO_RAW_DATA_RULE
    ),
    "tools": [evaluate_model, compare_models],
    "model": SUBAGENT_MODEL,
}

ALL_SUBAGENTS = [data_profiler, feature_engineer, model_trainer, evaluator]
