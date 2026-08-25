"""agent.py's prompt-building is pure string templating and is tested
directly, without constructing a real orchestrator (that requires an
ANTHROPIC_API_KEY and isn't something a unit suite should depend on)."""

from __future__ import annotations

from credit_agent.agent import _orchestrator_prompt
from credit_agent.data_registry import TARGET_COLUMN


def test_prompt_embeds_the_exact_run_specific_filenames():
    prompt = _orchestrator_prompt("report_20260824-120000.md", "pipeline_state_20260824-120000.md")
    assert "report_20260824-120000.md" in prompt
    assert "pipeline_state_20260824-120000.md" in prompt
    assert TARGET_COLUMN in prompt


def test_prompt_never_mentions_the_old_fixed_filenames():
    """Regression guard for the timestamping fix: a stray hardcoded
    `report.md`/`pipeline_state.md` in the prompt would silently reintroduce
    the overwrite-on-rerun bug."""
    prompt = _orchestrator_prompt("report_abc.md", "pipeline_state_abc.md")
    assert "report.md" not in prompt
    assert "pipeline_state.md" not in prompt
