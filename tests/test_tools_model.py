"""model-trainer tools: train_classifier (fixed hyperparams) and
tune_hyperparameters (Optuna search). Kept fast by using the small synthetic
dataset and, for tuning, a tiny monkeypatched trial budget -- these are unit
tests for wiring/contracts, not a benchmark of search quality."""

from __future__ import annotations

import pytest

from credit_agent import data_registry as reg
from credit_agent import tools_model
from credit_agent.tools_model import list_trained_models, train_classifier, tune_hyperparameters

TARGET = reg.TARGET_COLUMN
SCORING_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


@pytest.mark.parametrize("algorithm", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_train_classifier_registers_model_and_reports_cv_metrics(split_names, algorithm):
    train_name, _ = split_names
    result = train_classifier.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": algorithm,
            "model_name": f"m_{algorithm}",
        }
    )
    assert result["algorithm"] == algorithm
    assert set(result["cv_metrics"]) == SCORING_KEYS
    for metric_stats in result["cv_metrics"].values():
        assert set(metric_stats) == {"mean", "std"}

    model = reg.get_model(f"m_{algorithm}")
    assert model["target"] == TARGET
    assert TARGET not in model["feature_columns"]


def test_train_classifier_unknown_algorithm_errors(split_names):
    train_name, _ = split_names
    result = train_classifier.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "not_a_real_algorithm",
            "model_name": "m1",
        }
    )
    assert "error" in result
    assert "m1" not in reg.MODELS


def test_train_classifier_missing_target_column_errors(split_names):
    train_name, _ = split_names
    result = train_classifier.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": "not_a_column",
            "algorithm": "logistic_regression",
            "model_name": "m1",
        }
    )
    assert "error" in result


def test_train_classifier_invalid_hyperparams_errors(split_names):
    train_name, _ = split_names
    result = train_classifier.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "random_forest",
            "model_name": "m1",
            "hyperparams": {"not_a_real_kwarg": 123},
        }
    )
    assert "error" in result


def test_tune_hyperparameters_registers_best_model(split_names, monkeypatch):
    train_name, _ = split_names
    # keep the search tiny -- this is a wiring test, not a search-quality test
    monkeypatch.setattr(tools_model, "MAX_TUNE_TRIALS", 2)

    result = tune_hyperparameters.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "logistic_regression",
            "model_name": "tuned_logreg",
            "n_trials": 2,
        }
    )

    assert result["n_trials"] == 2
    assert "C" in result["best_params"]
    assert set(result["cv_metrics"]) == SCORING_KEYS
    assert "tuned_logreg" in reg.MODELS


def test_tune_hyperparameters_clamps_n_trials_to_cap(split_names, monkeypatch):
    train_name, _ = split_names
    monkeypatch.setattr(tools_model, "MAX_TUNE_TRIALS", 2)

    result = tune_hyperparameters.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "logistic_regression",
            "model_name": "tuned_logreg",
            "n_trials": 999,  # way over the cap
        }
    )

    assert result["n_trials"] == 2


def test_tune_hyperparameters_unknown_algorithm_errors(split_names):
    train_name, _ = split_names
    result = tune_hyperparameters.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "not_a_real_algorithm",
            "model_name": "m1",
            "n_trials": 1,
        }
    )
    assert "error" in result


def test_list_trained_models_empty():
    result = list_trained_models.invoke({})
    assert result == {"note": "no models trained yet"}


def test_list_trained_models_after_training(split_names):
    train_name, _ = split_names
    train_classifier.invoke(
        {
            "train_dataset_name": train_name,
            "target_column": TARGET,
            "algorithm": "logistic_regression",
            "model_name": "m1",
        }
    )
    result = list_trained_models.invoke({})
    assert result["m1"]["algorithm"] == "LogisticRegression"
    assert result["m1"]["n_features"] == 4
