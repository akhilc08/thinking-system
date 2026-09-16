"""Manual smoke test (spec §8 item 2): OpenRouter backend, one topic, small
round budget. Confirms a valid hypothesis is produced, then scores it
through the benchmark harness (Plan 1) as a boundary integration check —
this is the only place orchestrator code touches `benchmark`.

Run: python scripts/orchestrator_smoke_openrouter.py
Requires OPENROUTER_API_KEY set in the environment. Makes real OpenRouter
API calls and, if `benchmark` is available, 3 more calls to score the
result — costs a small amount of real money.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DEFAULT_GENERATION_MODEL, OPENROUTER_API_KEY
from orchestrator.backend import OpenRouterBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


async def main() -> None:
    config = RunConfig(
        topic="graph neural network expressiveness", domain="Computer Science",
        subfield="graph neural network expressiveness", backend_kind="openrouter",
        rounds=1, pool_size=6, n_divergent_per_round=3, debate_pairs_per_round=1, k_final=1,
        max_model_calls=30,
    )
    backend = OpenRouterBackend(model=DEFAULT_GENERATION_MODEL, api_key=OPENROUTER_API_KEY)
    embedding_backend = SentenceTransformerBackend()
    db_path = Path("/tmp/orchestrator_smoke_openrouter.db")
    store = RunStore(db_path)
    memory = MemoryStore(db_path)

    summary = await run_supervisor(config, backend, embedding_backend, store, memory)
    print(f"Status: {summary.status}, calls: {summary.total_model_calls}")
    for candidate in summary.final_candidates:
        print(f"\nHypothesis (elo {candidate.elo:.0f}): {candidate.text}")

    try:
        from benchmark.models import Hypothesis
        from benchmark.scorer import score_hypothesis
    except ImportError:
        print("\n(benchmark package not available — skipping scoring step)")
        return

    for candidate in summary.final_candidates:
        hyp = Hypothesis(
            topic=config.topic, domain=config.domain, subfield=config.subfield,
            text=candidate.text, source="our_swarm",
        )
        result = score_hypothesis(hyp)
        print(f"\nBenchmark score for this hypothesis: {result.weighted_total:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
