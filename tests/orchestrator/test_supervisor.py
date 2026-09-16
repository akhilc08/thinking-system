from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrator.memory import MemoryStore
from orchestrator.models import Candidate, DebateVerdict, RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import CallBudgetExceeded, CountingBackend, run_supervisor


def _config(**overrides):
    defaults = dict(
        topic="t", domain="Computer Science", subfield="t", rounds=2,
        pool_size=10, n_divergent_per_round=2, debate_pairs_per_round=1, k_final=1,
        max_model_calls=100, run_id="run-1",
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _fake_embedding_backend():
    backend = Mock()
    backend.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    return backend


def _candidate(text, elo=1200.0):
    return Candidate(text=text, round_num=1, source_strategy="divergent", elo=elo, embedding=[1.0, 0.0])


@pytest.mark.asyncio
async def test_run_supervisor_completes_and_threads_meta_review_guidance(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config()

    generate_calls = []

    async def fake_generate_round(cfg, backend, mem, prox, round_num, meta_review_guidance=""):
        generate_calls.append(meta_review_guidance)
        return [_candidate(f"idea round {round_num}")]

    async def fake_reflect_round(pool, backend):
        return []

    def fake_filter_passed(pool, critiques):
        return pool

    async def fake_rank_round(pool, backend, proximity, n_pairs):
        return []

    async def fake_evolve_round(pool, backend, round_num, topic):
        return []

    with patch("orchestrator.supervisor.generate_round", fake_generate_round), \
         patch("orchestrator.supervisor.reflect_round", fake_reflect_round), \
         patch("orchestrator.supervisor.filter_passed", fake_filter_passed), \
         patch("orchestrator.supervisor.rank_round", fake_rank_round), \
         patch("orchestrator.supervisor.evolve_round", fake_evolve_round), \
         patch("orchestrator.supervisor.synthesize_meta_review", return_value="guidance from this round"):
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    assert summary.status == "completed"
    assert generate_calls == ["", "guidance from this round"]  # round 2 received round 1's guidance
    assert store.get_run_status(config.run_id) == "completed"


@pytest.mark.asyncio
async def test_run_supervisor_truncates_when_call_budget_exceeded(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config(max_model_calls=0)

    async def fake_generate_round(cfg, backend, mem, prox, round_num, meta_review_guidance=""):
        backend.call("generation", "sys", "usr")  # triggers CallBudgetExceeded immediately
        return []

    with patch("orchestrator.supervisor.generate_round", fake_generate_round):
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    assert summary.status == "truncated_call_budget"
    assert store.get_run_status(config.run_id) == "truncated_call_budget"


@pytest.mark.asyncio
async def test_run_supervisor_skips_already_completed_run(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config(rounds=1)
    store.create_run(config)
    store.save_candidate(config.run_id, _candidate("existing idea"))
    store.mark_run_completed(config.run_id, total_model_calls=5, compute_multiplier=0.5)

    with patch("orchestrator.supervisor.generate_round", AsyncMock()) as fake_generate:
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    fake_generate.assert_not_called()
    assert summary.status == "completed"
    assert len(summary.final_candidates) == 1


def test_counting_backend_raises_past_budget():
    inner = Mock()
    inner.call.return_value = "response"
    counting = CountingBackend(inner, max_calls=2)

    counting.call("role", "sys", "usr")
    counting.call("role", "sys", "usr")
    with pytest.raises(CallBudgetExceeded):
        counting.call("role", "sys", "usr")
    assert inner.call.call_count == 2
