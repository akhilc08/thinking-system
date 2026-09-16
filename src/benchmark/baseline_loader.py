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


def load_published_score(idea_model: str, track: str, subdomain: str) -> float | None:
    df = _load_scores_dataframe()
    rows = df[
        (df["idea_model"] == idea_model)
        & (df["track"] == track)
        & (df["subdomain"] == subdomain)
    ]
    if rows.empty:
        return None

    per_critic_scores = {}
    for _, row in rows.iterrows():
        per_critic_scores[row["critic_model"]] = {
            dim: row[col] for dim, col in _DIM_COLUMNS.items()
        }

    _per_dim, weighted_total = aggregate_dimension_scores(per_critic_scores)
    return weighted_total
