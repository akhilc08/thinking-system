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


def test_score_hypothesis_calls_all_three_critics_and_aggregates():
    hyp = Hypothesis("t", "Computer Science", "t", "We hypothesize that...", "our_swarm")

    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (_fake_scores(5), "raw2"),
            (_fake_scores(6), "raw3"),
        ]
        result = score_hypothesis(hyp)

    assert mock_score_one.call_count == 3
    # originality [9, 5, 6] -> drop 9 -> mean([5, 6]) = 5.5
    assert result.per_dimension_trimmed["originality"] == 5.5
    assert set(result.per_critic.keys()) == {
        "qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1",
    }
    assert result.hypothesis is hyp


def test_score_hypothesis_skips_a_critic_whose_call_failed():
    hyp = Hypothesis("t", "Computer Science", "t", "text", "our_swarm")
    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (None, "ERR: parse failure"),
            (_fake_scores(6), "raw3"),
        ]
        result = score_hypothesis(hyp)

    assert len(result.per_critic) == 2
    # originality [9, 6], len < 2 rule doesn't apply since we have 2 values ->
    # trimmed_mean([9, 6]) drops 9, returns [6]
    assert result.per_dimension_trimmed["originality"] == 6
