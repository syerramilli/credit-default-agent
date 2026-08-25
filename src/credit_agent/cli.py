"""Entry point: `python -m credit_agent.cli` runs the orchestrator on the
default end-to-end ML goal and streams its progress to stdout."""

import argparse

from credit_agent.agent import build_orchestrator

DEFAULT_GOAL = (
    "Run the full ML pipeline end to end: profile the dataset, engineer "
    "features and split it, train at least two different classifier "
    "algorithms, evaluate them on the held-out test set, and write the "
    "final report with your model recommendation."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the credit-default deep agent.")
    parser.add_argument(
        "goal",
        nargs="?",
        default=DEFAULT_GOAL,
        help="The task to give the orchestrator (defaults to the full pipeline).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final response, not intermediate step updates.",
    )
    args = parser.parse_args()

    agent, report_path = build_orchestrator()
    inputs = {"messages": [{"role": "user", "content": args.goal}]}

    if args.quiet:
        result = agent.invoke(inputs)
        print(result["messages"][-1].content)
    else:
        for update in agent.stream(inputs, stream_mode="updates"):
            for node, payload in update.items():
                messages = payload.get("messages") if isinstance(payload, dict) else None
                if not messages:
                    continue
                for msg in messages:
                    content = getattr(msg, "content", None)
                    if content:
                        print(f"\n--- {node} ---\n{content}")

    print(f"\nDone. Check {report_path} for the final report.")


if __name__ == "__main__":
    main()
