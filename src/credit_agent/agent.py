"""Builds the orchestrator deep agent that delegates to the ML sub-agents."""

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from credit_agent.config import ORCHESTRATOR_MODEL, OUTPUTS_DIR
from credit_agent.data_registry import TARGET_COLUMN
from credit_agent.subagents import ALL_SUBAGENTS

ORCHESTRATOR_PROMPT = f"""\
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
next sub-agent, and use read_file/todos to keep track of them yourself.

When every stage is complete, write a concise Markdown report to
`report.md` (via write_file) summarizing: the dataset, the feature
engineering applied, each model's cross-validated and test metrics, the top
predictive features, and your final model recommendation with reasoning.

Hard rule, and the reason this whole pipeline is built this way: no sub-agent
should ever pull raw row-level data into context. Every tool available to
every sub-agent already enforces this by returning only schemas, dtypes,
aggregate statistics, or small capped samples -- lean on that, don't fight
it, and don't ask a sub-agent to "show me the data".
"""


def build_orchestrator():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=ORCHESTRATOR_MODEL,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=ALL_SUBAGENTS,
        backend=FilesystemBackend(root_dir=str(OUTPUTS_DIR), virtual_mode=True),
    )
