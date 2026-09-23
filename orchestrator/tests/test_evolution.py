import json
from unittest.mock import Mock, patch

import pytest

from orchestrator.evolution import evolve_round
from orchestrator.models import Candidate


def _idea_json(hypothesis):
    return json.dumps({
        "title": hypothesis, "problem": "p", "hypothesis": hypothesis, "mechanism": "m",
        "method": "me", "prediction": "pr", "falsification": "f", "novelty_delta": "n",
    })


class _FakeBackend:
    def __init__(self, ideas):
        self.ideas = list(ideas)
        self.call_count = 0

    def call(self, role, system_prompt, user_prompt):
        idea = self.ideas[self.call_count]
        self.call_count += 1
        return _idea_json(idea)


@pytest.mark.asyncio
async def test_evolve_round_always_includes_deliberate_divergence():
    pool = [
        Candidate(text="idea A", round_num=1, source_strategy="divergent", elo=1300.0),
        Candidate(text="idea B", round_num=1, source_strategy="divergent", elo=1250.0),
        Candidate(text="idea C", round_num=1, source_strategy="divergent", elo=1220.0),
        Candidate(text="idea D", round_num=1, source_strategy="divergent", elo=1200.0),
    ]
    backend = _FakeBackend(["diverged", "combined1", "combined2", "grounded", "analogized"])

    with patch("orchestrator.evolution.search_literature", return_value=[]):
        result = await evolve_round(pool, backend, round_num=2, topic="t")

    strategies = [c.source_strategy for c in result]
    assert "deliberate_divergence" in strategies
    assert strategies.count("combine") == 2  # second combine pass replaces the dropped `simplify`
    assert len(result) == 5  # divergence + combine(top3) + combine(ranks 2-4) + ground + analogical


@pytest.mark.asyncio
async def test_evolve_round_skips_combine_when_pool_has_one_candidate():
    pool = [Candidate(text="idea A", round_num=1, source_strategy="divergent", elo=1300.0)]
    backend = _FakeBackend(["diverged", "grounded", "analogized"])

    with patch("orchestrator.evolution.search_literature", return_value=[]):
        result = await evolve_round(pool, backend, round_num=2, topic="t")

    strategies = {c.source_strategy for c in result}
    assert "combine" not in strategies
    assert len(result) == 3


@pytest.mark.asyncio
async def test_evolve_round_runs_only_one_combine_when_pool_has_exactly_three():
    pool = [
        Candidate(text="idea A", round_num=1, source_strategy="divergent", elo=1300.0),
        Candidate(text="idea B", round_num=1, source_strategy="divergent", elo=1250.0),
        Candidate(text="idea C", round_num=1, source_strategy="divergent", elo=1200.0),
    ]
    backend = _FakeBackend(["diverged", "combined1", "grounded", "analogized"])

    with patch("orchestrator.evolution.search_literature", return_value=[]):
        result = await evolve_round(pool, backend, round_num=2, topic="t")

    strategies = [c.source_strategy for c in result]
    assert strategies.count("combine") == 1  # only 3 in pool -- no distinct ranks-2-4 window to combine
    assert len(result) == 4


@pytest.mark.asyncio
async def test_evolve_round_returns_empty_list_for_empty_pool_without_calling_backend():
    backend = Mock()

    result = await evolve_round([], backend, round_num=1, topic="t")

    assert result == []
    backend.call.assert_not_called()
