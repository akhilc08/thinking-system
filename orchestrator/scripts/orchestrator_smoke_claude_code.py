"""Manual smoke test (spec §8 item 3): same as
orchestrator_smoke_openrouter.py, with backend swapped to ClaudeCodeBackend
— proves the pluggable-backend design actually works with just a backend
swap (spec §5.1). Requires the `claude` CLI installed and authenticated;
also requires OPENROUTER_API_KEY for the embedding-similarity service
(Proximity always uses local sentence-transformers, no API key needed
there — see Task 6 — but the topic/domain match the OpenRouter smoke
script's for a fair comparison of the two backends' output on request).

Run: python scripts/orchestrator_smoke_claude_code.py
Runs `claude -p ...` subprocesses that consume the user's Claude Code plan.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.claude_code_backend import ClaudeCodeBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


async def main() -> None:
    config = RunConfig(
        topic="graph neural network expressiveness", domain="Computer Science",
        subfield="graph neural network expressiveness", backend_kind="claude_code",
        rounds=1, pool_size=6, n_divergent_per_round=3, debate_pairs_per_round=1, k_final=1,
        max_model_calls=30,
    )
    backend = ClaudeCodeBackend()
    embedding_backend = SentenceTransformerBackend()
    db_path = Path("/tmp/orchestrator_smoke_claude_code.db")
    store = RunStore(db_path)
    memory = MemoryStore(db_path)

    summary = await run_supervisor(config, backend, embedding_backend, store, memory)
    print(f"Status: {summary.status}, calls: {summary.total_model_calls}")
    for candidate in summary.final_candidates:
        print(f"\nHypothesis (elo {candidate.elo:.0f}): {candidate.text}")


if __name__ == "__main__":
    asyncio.run(main())
