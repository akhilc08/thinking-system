"""Scores a hypothesis using AgentIdeaBench's real critic pipeline.

Aggregation logic mirrors experiments/e16_v4_full_lit8d.py::analyze() exactly:
per dimension, drop the single highest of the critics' scores, average the
rest, then combine dimensions with SCORING_WEIGHTS / sum(SCORING_WEIGHTS).
"""
from benchmark.vendor_bootstrap import SCORING_DIMS, SCORING_WEIGHTS, trimmed_mean


def aggregate_dimension_scores(
    per_critic_scores: dict[str, dict[str, int]],
) -> tuple[dict[str, float], float]:
    per_dimension_trimmed: dict[str, float] = {}
    for dim in SCORING_DIMS:
        values = [scores[dim] for scores in per_critic_scores.values()]
        per_dimension_trimmed[dim] = trimmed_mean(values)

    weight_sum = sum(SCORING_WEIGHTS.values())
    weighted_total = sum(
        per_dimension_trimmed[dim] * SCORING_WEIGHTS[dim] for dim in SCORING_DIMS
    ) / weight_sum

    return per_dimension_trimmed, weighted_total
