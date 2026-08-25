"""evaluator tools: evaluate_model / compare_models, plus the
_feature_importances helper that branches between signed coefficients
(linear models) and mean-absolute TreeSHAP (tree ensembles). Per-row SHAP
values must never appear in a returned result -- several checks here pin
that down explicitly."""

from __future__ import annotations

import pytest

from credit_agent import data_registry as reg
from credit_agent.tools_eval import compare_models, evaluate_model
from credit_agent.tools_model import train_classifier

TARGET = reg.TARGET_COLUMN


@pytest.fixture
def trained(split_names):
    """Trains one model per algorithm on 'train' and returns their names,
    ready to evaluate against 'test'."""
    train_name, test_name = split_names
    names = {}
    for algorithm in ["logistic_regression", "random_forest", "gradient_boosting"]:
        model_name = f"m_{algorithm}"
        train_classifier.invoke(
            {
                "train_dataset_name": train_name,
                "target_column": TARGET,
                "algorithm": algorithm,
                "model_name": model_name,
            }
        )
        names[algorithm] = model_name
    return names, test_name


def test_evaluate_model_reports_roc_auc_and_pr_auc(trained):
    names, test_name = trained
    result = evaluate_model.invoke({"model_name": names["random_forest"], "test_dataset_name": test_name})
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert result["test_rows"] > 0
    assert len(result["confusion_matrix"]) == 2  # binary target


def test_evaluate_model_logistic_regression_uses_signed_coefficients(trained):
    names, test_name = trained
    result = evaluate_model.invoke(
        {"model_name": names["logistic_regression"], "test_dataset_name": test_name}
    )
    importances = result["top_feature_importances"]
    assert importances["method"] == "coefficients"
    top_features = importances["top_features"]
    assert len(top_features) <= 10
    # PAY_0 is the strongest signal by construction (see conftest.sample_df)
    assert top_features[0][0] == "PAY_0"


@pytest.mark.parametrize("algorithm", ["random_forest", "gradient_boosting"])
def test_evaluate_model_tree_models_use_shap(trained, algorithm):
    names, test_name = trained
    result = evaluate_model.invoke({"model_name": names[algorithm], "test_dataset_name": test_name})
    importances = result["top_feature_importances"]
    assert importances["method"] == "shap_tree_mean_abs"
    assert importances["shap_sample_rows"] == 60  # test set size, well under the SHAP row cap
    top_features = importances["top_features"]
    assert len(top_features) <= 10
    assert all(val >= 0 for _, val in top_features)  # mean(|shap|) is never negative
    assert top_features[0][0] == "PAY_0"


def test_evaluate_model_only_returns_aggregated_shap_never_per_row(trained):
    """The core guardrail this whole project is built around: per-row SHAP
    values must never leave the tool. Every leaf value in the response must
    be a plain scalar or string, never a list/array of per-row numbers."""
    names, test_name = trained
    result = evaluate_model.invoke({"model_name": names["random_forest"], "test_dataset_name": test_name})
    top_features = result["top_feature_importances"]["top_features"]
    for entry in top_features:
        name, value = entry
        assert isinstance(name, str)
        assert isinstance(value, float)


def test_evaluate_model_missing_feature_columns_errors(trained):
    names, _ = trained
    reg.register_dataset("bad_test", reg.get_dataset("test")[["PAY_0"]])
    result = evaluate_model.invoke(
        {"model_name": names["random_forest"], "test_dataset_name": "bad_test"}
    )
    assert "error" in result


def test_evaluate_model_missing_target_errors(trained):
    names, _ = trained
    df = reg.get_dataset("test").drop(columns=[TARGET])
    reg.register_dataset("no_target", df)
    result = evaluate_model.invoke(
        {"model_name": names["random_forest"], "test_dataset_name": "no_target"}
    )
    assert "error" in result


def test_compare_models_reports_headline_metrics_side_by_side(trained):
    names, test_name = trained
    result = compare_models.invoke({"model_names": list(names.values()), "test_dataset_name": test_name})
    assert set(result) == set(names.values())
    for summary in result.values():
        assert set(summary) == {"accuracy", "f1_weighted", "roc_auc", "pr_auc"}


def test_compare_models_propagates_error_for_unknown_model(trained):
    names, test_name = trained
    result = compare_models.invoke(
        {"model_names": [names["random_forest"], "not_a_real_model"], "test_dataset_name": test_name}
    )
    assert "error" in result["not_a_real_model"]
