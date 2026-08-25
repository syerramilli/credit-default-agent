"""Tools for the evaluator sub-agent.

Runs the fitted model against a held-out test set and returns aggregate
metrics (classification report, confusion matrix counts, ROC AUC, PR AUC,
top feature importances) -- never individual predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from langchain_core.tools import tool

from credit_agent import data_registry as reg
from credit_agent.config import MAX_SHAP_SAMPLE_ROWS
from credit_agent.guardrails import round_floats


def _feature_importances(estimator, X: pd.DataFrame, feature_columns: list[str]) -> dict | None:
    """Top-10 feature importances, computed the way that's appropriate for
    the model family: signed coefficients for linear models (e.g. logistic
    regression), mean-absolute TreeSHAP values for tree ensembles (random
    forest, XGBoost). SHAP is computed on at most MAX_SHAP_SAMPLE_ROWS rows
    to bound cost; only the aggregated top-10 list is ever returned -- the
    per-row SHAP matrix never leaves this function."""
    if hasattr(estimator, "coef_"):
        pairs = sorted(zip(feature_columns, estimator.coef_[0]), key=lambda p: -abs(p[1]))
        return {
            "method": "coefficients",
            "note": "signed logistic-regression coefficients (unscaled features skew magnitude)",
            "top_features": [(name, float(val)) for name, val in pairs[:10]],
        }

    if hasattr(estimator, "feature_importances_"):
        sample = (
            X if len(X) <= MAX_SHAP_SAMPLE_ROWS else X.sample(n=MAX_SHAP_SAMPLE_ROWS, random_state=42)
        )
        shap_values = shap.TreeExplainer(estimator).shap_values(sample)

        # Normalize the various shapes SHAP hands back for binary classifiers
        # across sklearn/xgboost estimators and shap versions down to a
        # single (n_samples, n_features) array for the positive class.
        if isinstance(shap_values, list):  # older SHAP + sklearn RF: [class0, class1]
            shap_values = shap_values[-1]
        elif np.asarray(shap_values).ndim == 3:  # newer SHAP + sklearn RF: (n, features, classes)
            shap_values = shap_values[:, :, -1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        pairs = sorted(zip(feature_columns, mean_abs), key=lambda p: -p[1])
        return {
            "method": "shap_tree_mean_abs",
            "note": "mean(|TreeSHAP value|) per feature, positive class, over a sample of the input rows",
            "shap_sample_rows": int(len(sample)),
            "top_features": [(name, float(val)) for name, val in pairs[:10]],
        }

    return None


@tool
def evaluate_model(model_name: str, test_dataset_name: str) -> dict:
    """Evaluate a registered fitted model against a registered test dataset.
    Returns a classification report, confusion matrix (small aggregate
    counts), ROC AUC, PR AUC (average precision -- the better-suited ranking
    metric under class imbalance), and the top 10 feature
    importances/coefficients if the model exposes them: signed coefficients
    for logistic regression, mean-absolute TreeSHAP values for random forest
    / XGBoost. Never returns per-row predictions or SHAP values."""
    model = reg.get_model(model_name)
    estimator, feature_columns, target = model["estimator"], model["feature_columns"], model["target"]

    df = reg.get_dataset(test_dataset_name)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return {"error": f"Test dataset is missing feature columns: {missing}"}
    if target not in df.columns:
        return {"error": f"Test dataset missing target column '{target}'."}

    X, y = df[feature_columns], df[target]
    preds = estimator.predict(X)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()

    result = {
        "model_name": model_name,
        "test_rows": int(df.shape[0]),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [str(c) for c in sorted(y.unique())],
    }

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)[:, 1]
        result["roc_auc"] = float(roc_auc_score(y, proba))
        result["pr_auc"] = float(average_precision_score(y, proba))

    importances = _feature_importances(estimator, X, feature_columns)
    if importances is not None:
        result["top_feature_importances"] = importances

    return round_floats(result)


@tool
def compare_models(model_names: list[str], test_dataset_name: str) -> dict:
    """Evaluate several registered models against the same test dataset and
    return their headline metrics (accuracy, f1, roc_auc, pr_auc) side by
    side, to support picking a winner."""
    summary = {}
    for name in model_names:
        try:
            full = evaluate_model.invoke({"model_name": name, "test_dataset_name": test_dataset_name})
        except reg.RegistryError as e:
            # evaluate_model raises (rather than returning {"error": ...}) for
            # an unknown model/dataset name, so this must be caught here --
            # one bad name in the list shouldn't fail every other model's
            # comparison.
            summary[name] = {"error": str(e)}
            continue
        if "error" in full:
            summary[name] = full
            continue
        report = full["classification_report"]
        summary[name] = {
            "accuracy": report.get("accuracy"),
            "f1_weighted": report.get("weighted avg", {}).get("f1-score"),
            "roc_auc": full.get("roc_auc"),
            "pr_auc": full.get("pr_auc"),
        }
    return summary
