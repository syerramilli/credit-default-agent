# credit-default-agent

A minimal boilerplate for [LangChain `deepagents`](https://github.com/langchain-ai/deepagents):
one orchestrator agent that delegates an end-to-end ML workflow to four
specialist sub-agents, over the UCI [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
dataset (30,000 rows, 23 features, binary target).

## The one rule this project is built around

**The LLM never sees raw data.** Every tool a sub-agent can call does its
work on the real pandas DataFrame in Python, then returns only:

- schemas / dtypes
- aggregate statistics (means, quantiles, correlations, value counts)
- a small, hard-capped sample (10 rows max), only when asked for

DataFrames and fitted models live in an in-process registry
([`data_registry.py`](src/credit_agent/data_registry.py)) addressed by
string name. Tools take/return names and small JSON summaries — never the
objects — so there's no code path where a full table gets serialized into a
prompt. The caps live in [`config.py`](src/credit_agent/config.py) and are
enforced in [`guardrails.py`](src/credit_agent/guardrails.py); every
`tools_*.py` module routes its output through them.

## Architecture

```
                    ┌────────────────────┐
                    │    orchestrator     │   plans the pipeline (write_todos),
                    │  (claude-opus-5)    │   delegates each stage, writes report_<ts>.md
                    └──────────┬──────────┘
             ┌──────────┬──────┴───────┬──────────────┐
             ▼          ▼              ▼              ▼
      data-profiler  feature-engineer  model-trainer  evaluator
      schema, dtypes  encode/scale/    train 2+       classification
      missingness,    split into       classifiers,   report, confusion
      stats, corr     train/test       cross-val      matrix, feature
                                        metrics        importances, verdict
```

Each sub-agent gets its own narrow tool surface (see
[`subagents.py`](src/credit_agent/subagents.py)) — the data-profiler can't
train models, the model-trainer can't touch raw columns, etc. That
separation is what makes `deepagents`' context isolation useful here: each
sub-agent's tool-call trace stays out of the orchestrator's context, so only
its final summary comes back up.

## Setup

```bash
cd credit-default-agent
uv venv --python 3.12 && source .venv/bin/activate   
uv sync
cp .env.example .env                                 
```

Dataset download happens automatically on first run (via `ucimlrepo`) and is
cached to `data/credit_default.csv`. To pre-fetch it separately:

```bash
python scripts/download_data.py
```

## Run it

```bash
python -m credit_agent.cli
```

This streams each sub-agent's progress to stdout and writes the final report
to `outputs/report_<timestamp>.md` — each run gets its own timestamped
report (and scratch `pipeline_state_<timestamp>.md`), so re-running never
overwrites a previous run's output. The exact path is also printed at the
end of the run. Pass a different goal as an argument to steer it, or
`--quiet` to only print the final message:

```bash
python -m credit_agent.cli "Just profile the dataset and summarize class imbalance."
```

## Cost / model configuration

`ANTHROPIC_API_KEY` is required. `ORCHESTRATOR_MODEL` and `SUBAGENT_MODEL`
(see `.env`) default to `claude-opus-5`. Since the orchestrator only plans
and delegates while the sub-agents do the well-specified execution work,
dropping `SUBAGENT_MODEL` to `claude-sonnet-5` or `claude-haiku-4-5` is a
reasonable way to cut cost once you've seen the boilerplate work end to end
— combined with the schema/stats-only rule above, that's the main lever for
keeping an agentic pipeline like this affordable.

## Extending this

- Add a tool: write an `@tool`-decorated function in the relevant
  `tools_*.py`, route its return value through `guardrails.py`, and add it
  to the right sub-agent's `"tools"` list in `subagents.py`.
- Add a sub-agent: copy the shape of an existing entry in `subagents.py`
  (`name`, `description`, `system_prompt`, `tools`, optional `model`), add
  it to `ALL_SUBAGENTS`, and mention it in the orchestrator's
  `ORCHESTRATOR_PROMPT` in `agent.py`.
- Swap the dataset: replace `data_registry._fetch_and_cache` and
  `TARGET_COLUMN`; everything downstream is dataset-agnostic.
