"""CLI entry point: python -m orchestrator.cli --topic "..." --domain "..." \
    --subfield "..." --backend openrouter --rounds 3 --k 1
"""
import argparse
import asyncio

from config.orchestrator import DB_PATH, DEFAULT_GENERATION_MODEL, OPENROUTER_API_KEY
from orchestrator.backend import OpenRouterBackend
from orchestrator.claude_code_backend import ClaudeCodeBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


def build_backend(kind: str):
    if kind == "claude_code":
        return ClaudeCodeBackend()
    return OpenRouterBackend(model=DEFAULT_GENERATION_MODEL, api_key=OPENROUTER_API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ideation orchestrator on a topic.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--subfield", required=True)
    parser.add_argument("--backend", choices=["openrouter", "claude_code"], default="openrouter")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    config = RunConfig(
        topic=args.topic, domain=args.domain, subfield=args.subfield,
        backend_kind=args.backend, rounds=args.rounds, k_final=args.k,
    )
    backend = build_backend(args.backend)
    embedding_backend = SentenceTransformerBackend()
    store = RunStore(DB_PATH)
    memory = MemoryStore(DB_PATH)

    summary = asyncio.run(run_supervisor(config, backend, embedding_backend, store, memory))

    print(
        f"Run {summary.run_id}: {summary.status}, {summary.total_model_calls} calls "
        f"({summary.compute_multiplier:.1f}x AgentIdeaBench's Active budget)"
    )
    for rank, candidate in enumerate(summary.final_candidates, start=1):
        print(f"\n[{rank}] (elo {candidate.elo:.0f}) {candidate.text}")


if __name__ == "__main__":
    main()
