import json
from unittest.mock import Mock, patch

import pytest

from orchestrator.generation import generate_round
from orchestrator.models import RunConfig


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
        idea = self.ideas[self.call_count % len(self.ideas)]
        self.call_count += 1
        return _idea_json(idea)


@pytest.mark.asyncio
async def test_generate_round_produces_n_divergent_candidates():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t", n_divergent_per_round=3)
    backend = _FakeBackend(["idea A", "idea B", "idea C"])
    memory = Mock()
    memory.query.return_value = []
    proximity = Mock()
    proximity.compute_embeddings = Mock()
    proximity.is_near_duplicate = Mock(return_value=False)

    with patch("orchestrator.generation.search_literature", return_value=[]):
        result = await generate_round(config, backend, memory, proximity, round_num=1)

    assert len(result) == 3
    assert {c.structured["hypothesis"] for c in result} == {"idea A", "idea B", "idea C"}
    assert all("Hypothesis: idea" in c.text for c in result)
    assert all(c.source_strategy == "divergent" for c in result)
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_generate_round_drops_memory_near_duplicates():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t", n_divergent_per_round=2)
    backend = _FakeBackend(["idea A", "idea B"])
    memory = Mock()
    memory.query.return_value = []
    proximity = Mock()
    proximity.compute_embeddings = Mock()
    proximity.is_near_duplicate = Mock(side_effect=[True, False])  # first candidate is a dup, second isn't

    with patch("orchestrator.generation.search_literature", return_value=[]):
        result = await generate_round(config, backend, memory, proximity, round_num=1)

    assert len(result) == 1
