"""Tools for the feature-engineer sub-agent.

Transformations run server-side over the real DataFrame and are saved back
into the registry under a new name; tools only ever return the resulting
schema/shape, not the transformed data itself.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from langchain_core.tools import tool

from credit_agent import data_registry as reg


@tool
def one_hot_encode(dataset_name: str, columns: list[str], new_dataset_name: str) -> dict:
    """One-hot encode the given categorical columns of a registered dataset
    and save the result under `new_dataset_name`. Returns the resulting
    schema, not the data."""
    df = reg.get_dataset(dataset_name)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}", "available_columns": list(df.columns)}
    encoded = pd.get_dummies(df, columns=columns, dtype=int)
    reg.register_dataset(new_dataset_name, encoded)
    return {
        "dataset_name": new_dataset_name,
        "rows": int(encoded.shape[0]),
        "columns": list(encoded.columns),
    }


@tool
def scale_numeric(dataset_name: str, columns: list[str], new_dataset_name: str) -> dict:
    """Standard-scale (zero mean, unit variance) the given numeric columns of
    a registered dataset and save the result under `new_dataset_name`.
    Returns the resulting schema and the fitted mean/scale per column
    (aggregate parameters, not row-level data)."""
    df = reg.get_dataset(dataset_name).copy()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}", "available_columns": list(df.columns)}
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    reg.register_dataset(new_dataset_name, df)
    return {
        "dataset_name": new_dataset_name,
        "rows": int(df.shape[0]),
        "scaled_columns": columns,
        "mean_": [round(float(m), 4) for m in scaler.mean_],
        "scale_": [round(float(s), 4) for s in scaler.scale_],
    }


@tool
def drop_columns(dataset_name: str, columns: list[str], new_dataset_name: str) -> dict:
    """Drop the given columns from a registered dataset and save the result
    under `new_dataset_name`. Useful for dropping ID columns before
    training."""
    df = reg.get_dataset(dataset_name)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}", "available_columns": list(df.columns)}
    new_df = df.drop(columns=columns)
    reg.register_dataset(new_dataset_name, new_df)
    return {"dataset_name": new_dataset_name, "rows": int(new_df.shape[0]), "columns": list(new_df.columns)}


@tool
def train_test_split_dataset(
    dataset_name: str,
    target_column: str,
    test_size: float = 0.2,
    new_train_name: str = "train",
    new_test_name: str = "test",
    random_state: int = 42,
) -> dict:
    """Split a registered dataset into stratified train/test sets (stratified
    on `target_column`) and register both under new names. Returns shapes and
    class balance, not the rows themselves."""
    df = reg.get_dataset(dataset_name)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found.", "available_columns": list(df.columns)}

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_column]
    )
    reg.register_dataset(new_train_name, train_df)
    reg.register_dataset(new_test_name, test_df)
    return {
        "train_dataset_name": new_train_name,
        "test_dataset_name": new_test_name,
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "train_class_balance": {str(k): int(v) for k, v in train_df[target_column].value_counts().items()},
        "test_class_balance": {str(k): int(v) for k, v in test_df[target_column].value_counts().items()},
    }
