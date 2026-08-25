"""In-process registries for dataframes and fitted models.

This is the key trick that keeps raw data out of the LLM's context: every
dataset and model lives here, addressed only by a short string name. Tools
take/return names (and small JSON summaries) -- never the objects
themselves -- so nothing forces a DataFrame to be serialized into a prompt.
"""

from __future__ import annotations

import pandas as pd

from credit_agent.config import DATA_PATH

DATASETS: dict[str, pd.DataFrame] = {}
MODELS: dict[str, dict] = {}  # name -> {"estimator": ..., "feature_columns": [...], "target": ...}

TARGET_COLUMN = "default_payment_next_month"


class RegistryError(ValueError):
    """Raised when a tool references a dataset/model name that doesn't exist.
    The message is written to be useful back to the LLM (it lists what does
    exist) so the agent can self-correct instead of the process crashing."""


def _fetch_and_cache() -> pd.DataFrame:
    """Download the UCI 'Default of Credit Card Clients' dataset (id=350)
    and cache it locally as a single CSV with descriptive column names and a
    renamed target column. `ucimlrepo` returns the raw feature columns as
    X1..X23 -- meaningless to both a human and an LLM reading the schema --
    so we rename them from the dataset's own `variables` metadata, which
    carries each Xi's real name (e.g. X1 -> LIMIT_BAL) in its
    `description` field."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "ucimlrepo is not installed. Run `pip install ucimlrepo` or "
            "manually place the dataset CSV at the DATA_PATH configured in .env."
        ) from e

    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_col = targets.columns[0]

    rename_map = {
        row["name"]: row["description"]
        for _, row in dataset.variables.iterrows()
        if row["role"] == "Feature" and isinstance(row["description"], str)
    }
    features = features.rename(columns=rename_map)

    df = features.copy()
    df[TARGET_COLUMN] = targets[target_col].values

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_raw_dataset(name: str = "raw") -> dict:
    """Load the dataset (from local cache, or download it if missing) and
    register it under `name`. Returns a small summary, not the data."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = _fetch_and_cache()

    register_dataset(name, df)
    return {
        "dataset_name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET_COLUMN,
        "source": str(DATA_PATH),
    }


def register_dataset(name: str, df: pd.DataFrame) -> None:
    DATASETS[name] = df


def get_dataset(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS)) or "(none loaded yet)"
        raise RegistryError(
            f"No dataset named '{name}' is registered. Available datasets: {available}. "
            "Did you forget to call load_dataset first?"
        )
    return DATASETS[name]


def register_model(name: str, estimator, feature_columns: list[str], target: str) -> None:
    MODELS[name] = {
        "estimator": estimator,
        "feature_columns": feature_columns,
        "target": target,
    }


def get_model(name: str) -> dict:
    if name not in MODELS:
        available = ", ".join(sorted(MODELS)) or "(none trained yet)"
        raise RegistryError(
            f"No model named '{name}' is registered. Available models: {available}. "
            "Did you forget to call train_classifier first?"
        )
    return MODELS[name]


def list_datasets() -> dict:
    return {name: dataset_summary(df) for name, df in DATASETS.items()}


def dataset_summary(df: pd.DataFrame) -> dict:
    return {"rows": int(df.shape[0]), "columns": list(df.columns)}
