from benchmark.scorer import aggregate_dimension_scores


def test_aggregate_drops_highest_score_per_dimension_then_weights():
    per_critic_scores = {
        "qwen/qwen3.6-plus": {
            "originality": 9, "feasibility": 6, "clarity": 7, "impact": 8, "specificity": 6,
        },
        "moonshotai/kimi-k2.6": {
            "originality": 5, "feasibility": 7, "clarity": 6, "impact": 5, "specificity": 5,
        },
        "z-ai/glm-5.1": {
            "originality": 6, "feasibility": 5, "clarity": 5, "impact": 6, "specificity": 4,
        },
    }
    per_dim, weighted = aggregate_dimension_scores(per_critic_scores)

    # originality: [9, 5, 6] -> drop 9 -> mean([5, 6]) = 5.5
    assert per_dim["originality"] == 5.5
    # feasibility: [6, 7, 5] -> drop 7 -> mean([6, 5]) = 5.5
    assert per_dim["feasibility"] == 5.5
    # clarity: [7, 6, 5] -> drop 7 -> mean([6, 5]) = 5.5
    assert per_dim["clarity"] == 5.5
    # impact: [8, 5, 6] -> drop 8 -> mean([5, 6]) = 5.5
    assert per_dim["impact"] == 5.5
    # specificity: [6, 5, 4] -> drop 6 -> mean([5, 4]) = 4.5
    assert per_dim["specificity"] == 4.5

    # weighted = (2*5.5 + 1.0*5.5 + 0.5*5.5 + 1.5*5.5 + 0.5*4.5) / 5.5
    expected = (2.0 * 5.5 + 1.0 * 5.5 + 0.5 * 5.5 + 1.5 * 5.5 + 0.5 * 4.5) / 5.5
    assert abs(weighted - expected) < 1e-9


def test_aggregate_handles_two_critics_by_using_lower_of_the_two():
    # trimmed_mean with len < 2 items after nothing to drop still applies
    # vendor's own rule: len(vs) < 2 returns vs[0]; with exactly 2 values,
    # trimmed_mean still drops the highest and returns the single remaining one.
    per_critic_scores = {
        "qwen/qwen3.6-plus": {"originality": 8, "feasibility": 5, "clarity": 5, "impact": 5, "specificity": 5},
        "moonshotai/kimi-k2.6": {"originality": 4, "feasibility": 5, "clarity": 5, "impact": 5, "specificity": 5},
    }
    per_dim, _ = aggregate_dimension_scores(per_critic_scores)
    assert per_dim["originality"] == 4  # drop 8, left with [4]


from unittest.mock import patch

import pytest

from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis


def _fake_scores(originality):
    return {
        "originality": {"score": originality, "reasoning": "r"},
        "feasibility": {"score": 6, "reasoning": "r"},
        "clarity": {"score": 6, "reasoning": "r"},
        "impact": {"score": 6, "reasoning": "r"},
        "specificity": {"score": 6, "reasoning": "r"},
    }


# The real vendored config.json ships 5 critic_models: the 3 that produced the
# published leaderboard, followed by 2 extras. Fixtures mirror that shape so
# tests exercise the slicing rather than an artificially correct-length list.
_REAL_CONFIG_SHAPED_CRITICS = [
    "qwen/qwen3.6-plus",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "minimax/minimax-m2.7",
    "deepseek/deepseek-v4-flash",
]
_PUBLISHED_CRITICS = set(_REAL_CONFIG_SHAPED_CRITICS[:3])


def test_score_hypothesis_uses_only_the_three_published_critics():
    """Only the first 3 of CRITIC_MODELS are scored, matching the vendored scripts.

    vendor/AgentIdeaBench/config.json lists 5 critic_models, but both scripts
    that produced the published leaderboard slice it:
    e13_litverify_rubric8.py:454 and e16_v4_full_lit8d.py:104 both do
    `critics = list(cfg.CRITIC_MODELS)[:n_critics]` with n_critics defaulting
    to 3. Their README says outright: "Keep the critics fixed. The published
    totals use qwen/qwen3.6-plus, moonshotai/kimi-k2.6 and z-ai/glm-5.1.
    Changing the critic ensemble changes the scale."

    Iterating all 5 would produce a drop-highest-of-5 aggregate compared
    against a drop-highest-of-3 baseline -- silently not comparable.
    """
    hyp = Hypothesis("t", "Computer Science", "t", "We hypothesize that...", "our_swarm")

    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = list(_REAL_CONFIG_SHAPED_CRITICS)
        # Five available responses: if the code called all 5 critics it would
        # succeed here, so the assertions below -- not a StopIteration -- are
        # what pins the behavior.
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (_fake_scores(5), "raw2"),
            (_fake_scores(6), "raw3"),
            (_fake_scores(1), "raw4"),
            (_fake_scores(2), "raw5"),
        ]
        result = score_hypothesis(hyp)

    assert mock_score_one.call_count == 3
    assert result.n_critics == 3
    # Proves the *slicing*, not merely the count: exactly the first 3, in
    # config order, and neither extra leaked in.
    assert set(result.per_critic.keys()) == _PUBLISHED_CRITICS
    assert "minimax/minimax-m2.7" not in result.per_critic
    assert "deepseek/deepseek-v4-flash" not in result.per_critic
    # The critic models actually passed to score_one are the published 3.
    called_critics = [call.args[1] for call in mock_score_one.call_args_list]
    assert called_critics == _REAL_CONFIG_SHAPED_CRITICS[:3]


def test_score_hypothesis_calls_all_three_critics_and_aggregates():
    hyp = Hypothesis("t", "Computer Science", "t", "We hypothesize that...", "our_swarm")

    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        # Larger-than-3 pool, matching the real config.json shape.
        mock_cfg.CRITIC_MODELS = list(_REAL_CONFIG_SHAPED_CRITICS)
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (_fake_scores(5), "raw2"),
            (_fake_scores(6), "raw3"),
            (_fake_scores(1), "raw4"),
            (_fake_scores(2), "raw5"),
        ]
        result = score_hypothesis(hyp)

    assert mock_score_one.call_count == 3
    # originality [9, 5, 6] -> drop 9 -> mean([5, 6]) = 5.5
    assert result.per_dimension_trimmed["originality"] == 5.5
    assert set(result.per_critic.keys()) == {
        "qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1",
    }
    assert result.hypothesis is hyp
    # A clean, fully-3-critic score -- directly comparable to published numbers.
    assert result.n_critics == 3


def test_score_hypothesis_skips_a_critic_whose_call_failed():
    hyp = Hypothesis("t", "Computer Science", "t", "text", "our_swarm")
    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        # Larger-than-3 pool, matching the real config.json shape: the failed
        # critic must NOT be silently backfilled from the 2 extras.
        mock_cfg.CRITIC_MODELS = list(_REAL_CONFIG_SHAPED_CRITICS)
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (None, "ERR: parse failure"),
            (_fake_scores(6), "raw3"),
            (_fake_scores(1), "raw4"),
            (_fake_scores(2), "raw5"),
        ]
        result = score_hypothesis(hyp)

    assert mock_score_one.call_count == 3

    assert len(result.per_critic) == 2
    # originality [9, 6], len < 2 rule doesn't apply since we have 2 values ->
    # trimmed_mean([9, 6]) drops 9, returns [6]
    assert result.per_dimension_trimmed["originality"] == 6
    # A degraded 2-critic score must be *recorded* as such, so a consumer can
    # tell it apart from a clean 3-critic one (it isn't directly comparable to
    # published 3-critic numbers).
    assert result.n_critics == 2


def test_score_hypothesis_raises_a_clear_error_when_every_critic_fails():
    # Realistic first-run failure: bad/expired OPENROUTER_API_KEY or a
    # rate-limit burst makes all 3 score_one calls return (None, raw).
    # Without an explicit guard this used to blow up deep inside the weighted
    # sum as `TypeError: unsupported operand type(s) for *: 'NoneType' and
    # 'float'`, whose traceback points at arithmetic, not the real cause.
    hyp = Hypothesis("t", "Computer Science", "quantum error correction", "text", "our_swarm")
    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]
        mock_score_one.side_effect = [
            (None, "ERR: 401 Unauthorized"),
            (None, "ERR: 401 Unauthorized"),
            (None, "ERR: 401 Unauthorized"),
        ]
        with pytest.raises(RuntimeError) as excinfo:
            score_hypothesis(hyp)

    message = str(excinfo.value)
    # The message must actually help someone debug this.
    assert "quantum error correction" in message       # which hypothesis
    assert "our_swarm" in message                      # where it came from
    assert "qwen/qwen3.6-plus" in message              # which critics were tried
    assert "OPENROUTER_API_KEY" in message             # what to go check
