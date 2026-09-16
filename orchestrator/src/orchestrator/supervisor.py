"""Supervisor (spec §5.2): holds run config, drives the round loop,
enforces the hard call-cap (spec §7), and performs final selection
(spec §5.3) — K=1 (benchmark mode) or diversity-aware K>1 (real use).
"""
from orchestrator.evolution import evolve_round
from orchestrator.generation import generate_round
from orchestrator.meta_review import synthesize_meta_review
from orchestrator.models import MemoryEntry, RunSummary
from orchestrator.proximity import Proximity
from orchestrator.ranking import rank_round
from orchestrator.reflection import filter_passed, reflect_round

AGENTIDEABENCH_ACTIVE_BUDGET = 10


class CallBudgetExceeded(Exception):
    pass


class CountingBackend:
    def __init__(self, inner, max_calls: int):
        self.inner = inner
        self.max_calls = max_calls
        self.count = 0

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        if self.count >= self.max_calls:
            raise CallBudgetExceeded(f"Model call budget of {self.max_calls} exceeded")
        self.count += 1
        return self.inner.call(role, system_prompt, user_prompt)


def _trim_pool(pool: list, pool_size: int) -> list:
    if len(pool) <= pool_size:
        return pool
    return sorted(pool, key=lambda c: c.elo, reverse=True)[:pool_size]


async def run_supervisor(config, backend, embedding_backend, store, memory) -> RunSummary:
    store.create_run(config)
    proximity = Proximity(embedding_backend)

    if store.get_run_status(config.run_id) == "completed":
        pool = _trim_pool(store.get_pool(config.run_id), config.pool_size)
        proximity.compute_embeddings(pool)
        final_candidates = proximity.select_diverse_topk(pool, config.k_final, config.diverse_selection_similarity_threshold)
        return RunSummary(run_id=config.run_id, total_model_calls=0, compute_multiplier=0.0, final_candidates=final_candidates, status="completed")

    counting_backend = CountingBackend(backend, config.max_model_calls)
    start_round = store.last_completed_round(config.run_id) + 1
    pool = _trim_pool(store.get_pool(config.run_id), config.pool_size) if start_round > 1 else []
    meta_review_guidance = store.get_last_meta_review_guidance(config.run_id) if start_round > 1 else ""
    truncated = False

    try:
        for round_num in range(start_round, config.rounds + 1):
            store.record_round_start(config.run_id, round_num)

            new_candidates = await generate_round(config, counting_backend, memory, proximity, round_num, meta_review_guidance)
            pool.extend(new_candidates)

            critiques = await reflect_round(pool, counting_backend)
            pool = filter_passed(pool, critiques)

            elo_before = {c.candidate_id: c.elo for c in pool}
            debates = await rank_round(pool, counting_backend, proximity, config.debate_pairs_per_round)
            by_id = {c.candidate_id: c for c in pool}
            for verdict in debates:
                store.save_debate(
                    config.run_id, round_num, verdict,
                    (elo_before[verdict.candidate_a_id], elo_before[verdict.candidate_b_id]),
                    (by_id[verdict.candidate_a_id].elo, by_id[verdict.candidate_b_id].elo),
                )

            pool = _trim_pool(pool, config.pool_size)

            evolved = await evolve_round(pool, counting_backend, round_num, config.topic)
            pool.extend(evolved)

            meta_review_guidance = synthesize_meta_review(counting_backend, pool, critiques, debates)

            for candidate in pool:
                store.save_candidate(config.run_id, candidate)
            for critique in critiques:
                store.save_critique(config.run_id, critique)
            store.record_round_complete(config.run_id, round_num, meta_review_guidance)
    except CallBudgetExceeded:
        truncated = True

    for candidate in pool:
        store.save_candidate(config.run_id, candidate)

    proximity.compute_embeddings(pool)
    final_candidates = proximity.select_diverse_topk(pool, config.k_final, config.diverse_selection_similarity_threshold)
    store.save_final_selection(config.run_id, final_candidates)

    for candidate in final_candidates:
        memory.write(
            MemoryEntry(
                domain=config.domain, subfield=config.subfield, text=candidate.text,
                elo=candidate.elo, run_id=config.run_id, embedding=candidate.embedding,
            ),
            proximity, config.duplicate_similarity_threshold,
        )

    compute_multiplier = counting_backend.count / AGENTIDEABENCH_ACTIVE_BUDGET
    if truncated:
        store.mark_run_truncated(config.run_id, counting_backend.count, compute_multiplier)
    else:
        store.mark_run_completed(config.run_id, counting_backend.count, compute_multiplier)

    return RunSummary(
        run_id=config.run_id, total_model_calls=counting_backend.count, compute_multiplier=compute_multiplier,
        final_candidates=final_candidates, status="truncated_call_budget" if truncated else "completed",
    )
