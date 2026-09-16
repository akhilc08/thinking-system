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


from benchmark.evidence import build_evidence
from benchmark.models import DimensionScore, ScoreResult
from benchmark.vendor_bootstrap import LIT8D_SYSTEM, score_one, vendor_config


def score_hypothesis(hypothesis, cutoff_date: str = "2026-05-31") -> ScoreResult:
    item_id = f"{hypothesis.source}|{hypothesis.subfield[:40]}"
    references = build_evidence(item_id, hypothesis.text, cutoff_date)

    per_critic: dict[str, dict[str, DimensionScore]] = {}
    for critic in vendor_config.CRITIC_MODELS:
        raw_scores, _raw_response = score_one(
            hypothesis.text, critic, hypothesis.domain, LIT8D_SYSTEM, references,
        )
        if raw_scores is None:
            continue
        per_critic[critic] = {
            dim: DimensionScore(dim, v["score"], v["reasoning"])
            for dim, v in raw_scores.items()
        }

    per_critic_int_scores = {
        critic: {dim: ds.score for dim, ds in dims.items()}
        for critic, dims in per_critic.items()
    }
    per_dimension_trimmed, weighted_total = aggregate_dimension_scores(per_critic_int_scores)

    return ScoreResult(
        hypothesis=hypothesis,
        per_critic=per_critic,
        per_dimension_trimmed=per_dimension_trimmed,
        weighted_total=weighted_total,
    )
