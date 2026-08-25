"""Builds the orchestrator deep agent that delegates to the ML sub-agents."""

from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from credit_agent.config import ORCHESTRATOR_MODEL, OUTPUTS_DIR
from credit_agent.data_registry import TARGET_COLUMN
from credit_agent.subagents import ALL_SUBAGENTS


def _orchestrator_prompt(report_filename: str, state_filename: str) -> str:
    return f"""\
You are the orchestrator of an end-to-end machine learning pipeline over the
UCI "Default of Credit Card Clients" dataset. The target column is
`{TARGET_COLUMN}` (1 = the client defaulted on their next payment).

You do not touch the data yourself. You plan the pipeline and delegate each
stage to the right specialist sub-agent via the `task` tool:

1. data-profiler   -- load the dataset, understand its schema/distributions
2. feature-engineer -- encode/scale/clean it, produce a train/test split
3. model-trainer    -- train at least two different classifier algorithms
4. evaluator        -- evaluate every trained model on the test set and
                        recommend a winner

Use write_todos to lay out this plan before delegating. Delegate stages in
order -- each stage's sub-agent needs the dataset/model names produced by
the previous stage, so pass those names along explicitly when you invoke the
next sub-agent, and keep track of them yourself in a scratch file named
`{state_filename}` (read_file/write_file) as you go.

When every stage is complete, write a concise Markdown report to
`{report_filename}` (via write_file) summarizing: the dataset, the feature
engineering applied, each model's cross-validated and test metrics, the top
predictive features, and your final model recommendation with reasoning.
This filename is specific to this run -- always use it exactly as given, so
this run's output never overwrites a previous run's.

Hard rule, and the reason this whole pipeline is built this way: no sub-agent
should ever pull raw row-level data into context. Every tool available to
every sub-agent already enforces this by returning only schemas, dtypes,
aggregate statistics, or small capped samples -- lean on that, don't fight
it, and don't ask a sub-agent to "show me the data".
"""


def build_orchestrator(run_id: str | None = None):
    """Build the orchestrator agent for one run. Each run is given its own
    timestamped report/scratch filenames (outputs/report_<run_id>.md,
    outputs/pipeline_state_<run_id>.md) baked into its system prompt, so
    consecutive runs never clobber each other's output. Returns
    `(agent, report_path)`."""
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    report_filename = f"report_{run_id}.md"
    state_filename = f"pipeline_state_{run_id}.md"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    agent = create_deep_agent(
        model=ORCHESTRATOR_MODEL,
        system_prompt=_orchestrator_prompt(report_filename, state_filename),
        subagents=ALL_SUBAGENTS,
        backend=FilesystemBackend(root_dir=str(OUTPUTS_DIR), virtual_mode=True),
    )
    return agent, OUTPUTS_DIR / report_filename
