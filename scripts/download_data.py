#!/usr/bin/env python
"""Standalone convenience script: download and cache the UCI dataset without
starting an agent run. Run with: python scripts/download_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_agent.data_registry import _fetch_and_cache  # noqa: E402

if __name__ == "__main__":
    df = _fetch_and_cache()
    print(f"Downloaded and cached {df.shape[0]} rows x {df.shape[1]} columns.")
