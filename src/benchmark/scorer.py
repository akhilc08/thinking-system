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


# The vendored config.json ships 5 critic_models, but only the first 3
# produced the published leaderboard. Both vendored production scripts slice
# the list rather than iterating it:
#   experiments/e13_litverify_rubric8.py:454  critics = list(cfg.CRITIC_MODELS)[:n_critics]
#   experiments/e16_v4_full_lit8d.py:104      critics = list(cfg.CRITIC_MODELS)[:n_critics]
# with n_critics defaulting to 3. Their README is explicit: "Keep the critics
# fixed. The published totals use qwen/qwen3.6-plus, moonshotai/kimi-k2.6 and
# z-ai/glm-5.1. Changing the critic ensemble changes the scale."
#
# Scoring all 5 would build a drop-highest-of-5 aggregate and compare it
# against a drop-highest-of-3 baseline -- a silent scale mismatch. We slice the
# same way they do rather than hardcoding the three model names, so this tracks
# config.json if their roster is ever reordered or extended.
PUBLISHED_N_CRITICS = 3


def score_hypothesis(hypothesis, cutoff_date: str = "2026-05-31") -> ScoreResult:
    item_id = f"{hypothesis.source}|{hypothesis.subfield[:40]}"
    references = build_evidence(item_id, hypothesis.text, cutoff_date)

    critics = list(vendor_config.CRITIC_MODELS)[:PUBLISHED_N_CRITICS]
    per_critic: dict[str, dict[str, DimensionScore]] = {}
    for critic in critics:
        raw_scores, _raw_response = score_one(
            hypothesis.text, critic, hypothesis.domain, LIT8D_SYSTEM, references,
        )
        if raw_scores is None:
            continue
        per_critic[critic] = {
            dim: DimensionScore(dim, v["score"], v["reasoning"])
            for dim, v in raw_scores.items()
        }

    if not per_critic:
        # Without this guard, aggregate_dimension_scores({}) yields None for
        # every dimension and the weighted sum dies on `None * float` -- a
        # TypeError whose traceback points at arithmetic rather than at the
        # real cause (no critic responded at all).
        raise RuntimeError(
            "No critic returned a usable score for hypothesis "
            f"{hypothesis.source}|{hypothesis.subfield!r} "
            f"(domain: {hypothesis.domain}). All {len(critics)} critic calls "
            f"failed or returned unparseable output: {', '.join(critics)}. "
            "This usually means a bad/expired OPENROUTER_API_KEY, an exhausted "
            "credit balance, or a rate-limit burst -- check "
            "vendor/AgentIdeaBench/.env and retry."
        )

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
        n_critics=len(per_critic),
    )
