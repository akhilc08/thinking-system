import json
from unittest.mock import Mock

import pytest

from orchestrator.models import Candidate
from orchestrator.ranking import elo_update, rank_round


def test_elo_update_equal_ratings_winner_gains_half_k():
    new_a, new_b = elo_update(1200.0, 1200.0, "A", k_factor=32.0)
    assert abs(new_a - 1216.0) < 1e-9
    assert abs(new_b - 1184.0) < 1e-9


def test_elo_update_upset_produces_bigger_swing_than_expected_win():
    # B (rated much lower) beats A (rated much higher): an upset.
    upset_a, upset_b = elo_update(1400.0, 1000.0, "B", k_factor=32.0)
    # A (rated much higher) beats B: an expected win.
    expected_a, expected_b = elo_update(1400.0, 1000.0, "A", k_factor=32.0)
    upset_swing = abs(upset_b - 1000.0)
    expected_swing = abs(expected_a - 1400.0)
    assert upset_swing > expected_swing


@pytest.mark.asyncio
async def test_rank_round_applies_verdicts_to_candidate_elo():
    a = Candidate(text="A", round_num=1, source_strategy="divergent", elo=1200.0)
    b = Candidate(text="B", round_num=1, source_strategy="divergent", elo=1200.0)

    backend = Mock()
    backend.call.return_value = json.dumps({"winner": "A", "reasoning": "A is more feasible"})

    proximity = Mock()
    proximity.pick_debate_pairs.return_value = [(a, b)]

    verdicts = await rank_round([a, b], backend, proximity, n_pairs=1)

    assert len(verdicts) == 1
    assert verdicts[0].winner_id == a.candidate_id
    assert a.elo > 1200.0
    assert b.elo < 1200.0
