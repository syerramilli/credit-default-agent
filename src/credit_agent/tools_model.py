"""Tools for the model-trainer sub-agent.

Training happens server-side against the registered DataFrame; only
cross-validated aggregate metrics come back to the model, never predictions
or row-level data.
"""

from __future__ import annotations

from typing import Literal

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from xgboost import XGBClassifier

from langchain_core.tools import tool

from credit_agent import data_registry as reg
from credit_agent.config import MAX_TUNE_TRIALS
from credit_agent.guardrails import round_floats

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ALGORITHMS = {
    "logistic_regression": lambda **kw: LogisticRegression(max_iter=1000, **kw),
    "random_forest": lambda **kw: RandomForestClassifier(random_state=42, **kw),
    "gradient_boosting": lambda **kw: XGBClassifier(
        random_state=42, eval_metric="logloss", **kw
    ),
}

# Search spaces are deliberately narrow -- wide enough to matter, narrow
# enough that a MAX_TUNE_TRIALS-trial TPE search actually converges instead
# of wandering. `logistic_regression` omits `max_iter`: it's fixed by the
# `_ALGORITHMS` factory above and would collide with it if suggested here.
_SEARCH_SPACES = {
    "logistic_regression": lambda trial: {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
    },
    "random_forest": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    },
    "gradient_boosting": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    },
}


@tool
def train_classifier(
    train_dataset_name: str,
    target_column: str,
    algorithm: Literal["logistic_regression", "random_forest", "gradient_boosting"],
    model_name: str,
    hyperparams: dict | None = None,
) -> dict:
    """Train a classifier on a registered training dataset and register the
    fitted model under `model_name`. Runs 5-fold cross-validation on the
    training set and returns mean/std accuracy, precision, recall, f1, and
    roc_auc -- aggregate metrics only, never predictions or row data.
    `algorithm` must be one of: logistic_regression, random_forest,
    gradient_boosting (XGBoost). `hyperparams` are passed straight to the
    underlying estimator constructor (e.g. {"n_estimators": 200, "max_depth": 6})."""
    if algorithm not in _ALGORITHMS:
        return {"error": f"Unknown algorithm '{algorithm}'.", "available": list(_ALGORITHMS)}

    df = reg.get_dataset(train_dataset_name)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found.", "available_columns": list(df.columns)}

    feature_columns = [c for c in df.columns if c != target_column]
    X, y = df[feature_columns], df[target_column]

    try:
        estimator = _ALGORITHMS[algorithm](**(hyperparams or {}))
    except TypeError as e:
        return {"error": f"Invalid hyperparams for {algorithm}: {e}"}

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(estimator, X, y, cv=5, scoring=scoring)

    estimator.fit(X, y)
    reg.register_model(model_name, estimator, feature_columns, target_column)

    metrics = {
        metric: {
            "mean": float(cv_results[f"test_{metric}"].mean()),
            "std": float(cv_results[f"test_{metric}"].std()),
        }
        for metric in scoring
    }
    return round_floats(
        {
            "model_name": model_name,
            "algorithm": algorithm,
            "train_rows": int(df.shape[0]),
            "n_features": len(feature_columns),
            "cv_metrics": metrics,
        }
    )


@tool
def tune_hyperparameters(
    train_dataset_name: str,
    target_column: str,
    algorithm: Literal["logistic_regression", "random_forest", "gradient_boosting"],
    model_name: str,
    n_trials: int = 20,
    scoring: str = "roc_auc",
) -> dict:
    """Search for good hyperparameters for `algorithm` using Optuna (TPE
    sampler), scoring each trial by 5-fold cross-validated `scoring` (default
    roc_auc) on a registered training dataset. Fits the best-found estimator
    on the full training set and registers it under `model_name` -- same
    contract as train_classifier, but with tuned hyperparameters instead of
    caller-supplied ones. `n_trials` is capped at MAX_TUNE_TRIALS (each trial
    is a full 5-fold CV fit, so cost scales with n_trials -- start small).
    Returns the best hyperparameters found, the best CV score, and the same
    aggregate CV metrics as train_classifier -- never predictions or row
    data."""
    if algorithm not in _ALGORITHMS:
        return {"error": f"Unknown algorithm '{algorithm}'.", "available": list(_ALGORITHMS)}

    df = reg.get_dataset(train_dataset_name)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found.", "available_columns": list(df.columns)}

    feature_columns = [c for c in df.columns if c != target_column]
    X, y = df[feature_columns], df[target_column]

    n_trials = max(1, min(n_trials, MAX_TUNE_TRIALS))
    build_params = _SEARCH_SPACES[algorithm]

    def objective(trial: optuna.Trial) -> float:
        estimator = _ALGORITHMS[algorithm](**build_params(trial))
        return float(cross_val_score(estimator, X, y, cv=5, scoring=scoring).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    estimator = _ALGORITHMS[algorithm](**best_params)

    scoring_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(estimator, X, y, cv=5, scoring=scoring_list)

    estimator.fit(X, y)
    reg.register_model(model_name, estimator, feature_columns, target_column)

    metrics = {
        metric: {
            "mean": float(cv_results[f"test_{metric}"].mean()),
            "std": float(cv_results[f"test_{metric}"].std()),
        }
        for metric in scoring_list
    }
    return round_floats(
        {
            "model_name": model_name,
            "algorithm": algorithm,
            "n_trials": n_trials,
            "tuned_metric": scoring,
            "best_score": study.best_value,
            "best_params": best_params,
            "train_rows": int(df.shape[0]),
            "n_features": len(feature_columns),
            "cv_metrics": metrics,
        }
    )


@tool
def list_trained_models() -> dict:
    """List the names, algorithms, and feature counts of all models trained
    so far in this session."""
    return {
        name: {"algorithm": type(m["estimator"]).__name__, "n_features": len(m["feature_columns"])}
        for name, m in reg.MODELS.items()
    } or {"note": "no models trained yet"}
