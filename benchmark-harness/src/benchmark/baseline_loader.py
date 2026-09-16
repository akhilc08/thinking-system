"""Reads AgentIdeaBench's published per-critic scores and aggregates them the
same way we aggregate our own hypotheses, so a comparison is apples-to-apples.
"""
import pandas as pd

from benchmark.scorer import aggregate_dimension_scores
from benchmark.vendor_bootstrap import SCORING_DIMS
from project_config.benchmark import VENDOR_ROOT

_SCORES_PATH = VENDOR_ROOT / "release_data" / "core" / "lit8d_scores_3seed.csv.gz"
_DIM_COLUMNS = {dim: f"score_{dim}" for dim in SCORING_DIMS}


def _load_scores_dataframe() -> pd.DataFrame:
    return pd.read_csv(_SCORES_PATH)


def _weighted_total_for_one_idea(group: pd.DataFrame) -> float:
    """Aggregate one idea's per-critic rows into a single weighted total."""
    per_critic_scores = {}
    for _, row in group.iterrows():
        per_critic_scores[row["critic_model"]] = {
            dim: row[col] for dim, col in _DIM_COLUMNS.items()
        }
    _per_dim, weighted_total = aggregate_dimension_scores(per_critic_scores)
    return weighted_total


def load_published_score(
    idea_model: str,
    track: str,
    subdomain: str,
    idea_index: int | None = None,
) -> float | None:
    """Published weighted-total score for one released cell.

    With ``idea_index`` given, returns the score for that exact hypothesis --
    an apples-to-apples comparison against a single re-scored hypothesis.
    With ``idea_index=None`` (the default), averages the per-idea weighted
    totals across every released hypothesis for the cell.
    """
    df = _load_scores_dataframe()
    mask = (
        (df["idea_model"] == idea_model)
        & (df["track"] == track)
        & (df["subdomain"] == subdomain)
    )
    if idea_index is not None:
        mask = mask & (df["idea_index"] == idea_index)
    rows = df[mask]
    if rows.empty:
        return None

    # Drop critic rows whose scoring call failed: lit8d_scores_3seed.csv.gz
    # marks these error="parse_fail" with NaN in every score_* column. Left in,
    # those NaNs propagate through aggregation and this function returns nan --
    # a float, so no `is None` check catches it, and `f"{nan:.2f}"` prints
    # "nan" without erroring. The vendor's own analyze()
    # (experiments/e16_v4_full_lit8d.py) does the equivalent by selecting
    # "WHERE variant=? AND scores_json IS NOT NULL".
    rows = rows.dropna(subset=list(_DIM_COLUMNS.values()))
    if rows.empty:
        # Every critic row for this cell errored -- no usable published score.
        return None

    if idea_index is not None:
        return _weighted_total_for_one_idea(rows)

    # A (idea_model, track, subdomain) triple can have multiple released
    # hypotheses (idea_index 1, 2, 3), each scored by the same critics.
    # Aggregate per idea_index first (so per_critic_scores never mixes rows
    # from two different ideas under one critic_model key), then average the
    # resulting weighted totals across ideas. Ideas left with no usable rows
    # by the dropna above simply drop out of the groupby and are skipped.
    per_idea_totals = [
        _weighted_total_for_one_idea(group) for _idx, group in rows.groupby("idea_index")
    ]
    return sum(per_idea_totals) / len(per_idea_totals)
