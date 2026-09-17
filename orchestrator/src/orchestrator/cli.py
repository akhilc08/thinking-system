"""CLI entry point: python -m orchestrator.cli --topic "..." --domain "..." \
    --subfield "..." --backend openrouter --rounds 3 --k 1
"""
import argparse
import asyncio
import json

from config import DB_PATH, DEFAULT_GENERATION_MODEL, OPENROUTER_API_KEY
from orchestrator.backend import OpenRouterBackend
from orchestrator.claude_code_backend import ClaudeCodeBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.report import export_run
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
    parser.add_argument("--pool-size", type=int, default=24, help="Max survivors kept after each round's debate (lower = more aggressive cutting)")
    parser.add_argument("--max-calls", type=int, default=200, help="Hard cap on model calls for the whole run")
    parser.add_argument("--round-robin", action="store_true", help="Debate every non-duplicate pair in the pool each round instead of a capped adjacent-Elo set (quadratic in pool size)")
    parser.add_argument("--export", help="Write full run history (all candidates incl. pruned, Elo snapshots, debates) as JSON to this path")
    args = parser.parse_args()

    config = RunConfig(
        topic=args.topic, domain=args.domain, subfield=args.subfield,
        backend_kind=args.backend, rounds=args.rounds, k_final=args.k,
        pool_size=args.pool_size, max_model_calls=args.max_calls,
        round_robin_debates=args.round_robin,
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

    if args.export:
        with open(args.export, "w") as f:
            json.dump(export_run(store, summary.run_id), f, indent=2)
        print(f"\nFull run history exported to {args.export}")


if __name__ == "__main__":
    main()
