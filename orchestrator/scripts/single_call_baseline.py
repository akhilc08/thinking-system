"""Baseline #1 from docs/research/design-recommendations.md's 4-tier ladder:
a single LLM call, no retrieval, no debate, no evolution -- "just ask the
model for an idea about X." Uses the same claude_code backend a real
orchestrator run uses, so a comparison against the swarm's output isolates
the swarm's *process* (retrieval/debate/evolution), not the underlying model.
"""
import argparse

from orchestrator.claude_code_backend import ClaudeCodeBackend

_SYSTEM_PROMPT = (
    "You are a research scientist. Propose one good, concrete research "
    "hypothesis for the given topic."
)


def generate_baseline_idea(topic: str, backend=None) -> str:
    backend = backend or ClaudeCodeBackend()
    return backend.call("baseline", _SYSTEM_PROMPT, f"Topic: {topic}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out", help="Write the idea text to this file")
    args = parser.parse_args()

    idea = generate_baseline_idea(args.topic)
    print(idea)
    if args.out:
        with open(args.out, "w") as f:
            f.write(idea)


if __name__ == "__main__":
    main()
