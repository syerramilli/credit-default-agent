"""cli.py wiring: argparse plumbing and the streaming vs. --quiet branches,
exercised with a fake agent so no ANTHROPIC_API_KEY or network call is
needed."""

from __future__ import annotations

from pathlib import Path

from credit_agent import cli


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeAgentQuiet:
    def __init__(self, captured_inputs: list):
        self._captured_inputs = captured_inputs

    def invoke(self, inputs):
        self._captured_inputs.append(inputs)
        return {"messages": [_FakeMessage("final answer")]}


class _FakeAgentStreaming:
    def __init__(self, captured_inputs: list):
        self._captured_inputs = captured_inputs

    def stream(self, inputs, stream_mode="updates"):
        self._captured_inputs.append(inputs)
        yield {"data-profiler": {"messages": [_FakeMessage("profiled the data")]}}
        yield {"evaluator": {"messages": [_FakeMessage("recommend random forest")]}}


def test_quiet_mode_prints_only_final_message(monkeypatch, capsys):
    captured_inputs: list = []
    report_path = Path("outputs/report_test.md")
    monkeypatch.setattr(
        cli, "build_orchestrator", lambda: (_FakeAgentQuiet(captured_inputs), report_path)
    )
    monkeypatch.setattr("sys.argv", ["credit_agent", "--quiet"])

    cli.main()

    out = capsys.readouterr().out
    assert "final answer" in out
    assert str(report_path) in out
    assert captured_inputs[0]["messages"][0]["content"] == cli.DEFAULT_GOAL


def test_streaming_mode_prints_each_subagent_update(monkeypatch, capsys):
    captured_inputs: list = []
    report_path = Path("outputs/report_test.md")
    monkeypatch.setattr(
        cli, "build_orchestrator", lambda: (_FakeAgentStreaming(captured_inputs), report_path)
    )
    monkeypatch.setattr("sys.argv", ["credit_agent"])

    cli.main()

    out = capsys.readouterr().out
    assert "--- data-profiler ---" in out
    assert "profiled the data" in out
    assert "--- evaluator ---" in out
    assert "recommend random forest" in out
    assert str(report_path) in out


def test_custom_goal_argument_is_passed_through_to_the_agent(monkeypatch, capsys):
    captured_inputs: list = []
    report_path = Path("outputs/report_test.md")
    monkeypatch.setattr(
        cli, "build_orchestrator", lambda: (_FakeAgentQuiet(captured_inputs), report_path)
    )
    custom_goal = "Just profile the dataset."
    monkeypatch.setattr("sys.argv", ["credit_agent", custom_goal, "--quiet"])

    cli.main()

    assert captured_inputs[0]["messages"][0]["content"] == custom_goal
