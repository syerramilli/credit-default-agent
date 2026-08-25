"""Central config: env vars, model ids, and the guardrail constants that cap
how much data any tool is allowed to hand back to the LLM."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "credit_default.csv"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Models are passed to deepagents as "anthropic:<model-id>" strings, which
# LangChain resolves to a ChatAnthropic instance for you. Override via env
# vars if you want to trade quality for cost while iterating.
ORCHESTRATOR_MODEL = f"anthropic:{os.getenv('ORCHESTRATOR_MODEL', 'claude-opus-5')}"
SUBAGENT_MODEL = f"anthropic:{os.getenv('SUBAGENT_MODEL', 'claude-opus-5')}"

# --- Guardrails ---------------------------------------------------------
# The whole point of this project: the LLM only ever sees schemas, dtypes,
# aggregate statistics, and (occasionally) a small, capped sample. Every
# tool in tools_*.py must route its output through these limits before it
# reaches the model's context window. Never widen these to "just make a
# tool work" — narrow the tool's request instead (e.g. ask for fewer
# columns), since the cap is the whole point.
MAX_SAMPLE_ROWS = 10          # hard ceiling on rows returned by any sample tool
MAX_CATEGORIES_SHOWN = 15     # top-k categories before collapsing into "other"
MAX_CORRELATION_COLUMNS = 25  # refuse a full corr matrix beyond this many columns
STAT_DECIMALS = 4             # rounding applied to every float sent to the model

# Not a data-size cap like the ones above, but the same instinct: bound the
# blast radius of a single tool call. Each Optuna trial is a full k-fold CV
# fit, so n_trials scales runtime/cost directly.
MAX_TUNE_TRIALS = 25

# Rows used when computing TreeSHAP feature importances. Also a compute cap,
# not an output cap -- per-row SHAP values never leave the tool anyway (they
# get aggregated to a mean-abs-per-feature summary before returning), but
# TreeSHAP cost scales with rows and this keeps it bounded on larger datasets.
MAX_SHAP_SAMPLE_ROWS = 2000
