"""
Manual parity check: pick one already-released hypothesis, re-score it
through our wrapper, compare against AgentIdeaBench's own published score.

Run: python scripts/benchmark_parity_check.py

Requires vendor/AgentIdeaBench/.env to have real OPENROUTER_API_KEY and
SEMANTIC_SCHOLAR_API_KEY values (see Task 1, Step 2). Costs a small number of
real OpenRouter API calls (3 critics x 1 hypothesis).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmark.baseline_loader import (
    _DIM_COLUMNS,
    _load_scores_dataframe,
    load_published_score,
)
from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis
from project_config.benchmark import VENDOR_ROOT

IDEAS_PATH = VENDOR_ROOT / "release_data" / "core" / "subdomain_ideas.csv.gz"


def _find_scored_idea_row(ideas):
    """First Active-track idea that actually has a usable published score.

    Only ~59% of track-C (idea_model, track, subdomain) cells in
    subdomain_ideas.csv.gz have matching rows in lit8d_scores_3seed.csv.gz, and
    the very first one does not -- so blindly taking .iloc[0] makes this script
    always bail out on "no published score found" without scoring anything.
    Match on the full (idea_model, track, subdomain, idea_index) tuple, against
    score rows that survive the errored-row filter, so the row we pick is
    guaranteed to produce a real number from load_published_score().
    """
    scores = _load_scores_dataframe()
    usable = scores.dropna(subset=list(_DIM_COLUMNS.values()))
    scored_keys = set(
        zip(
            usable["idea_model"],
            usable["track"],
            usable["subdomain"],
            usable["idea_index"],
        )
    )

    candidates = ideas[
        (ideas["track"] == "C") & (ideas["idea_text"].str.strip() != "")
    ]
    for _, row in candidates.iterrows():
        key = (row["idea_model"], row["track"], row["subdomain"], row["idea_index"])
        if key in scored_keys:
            return row
    return None


def main():
    ideas = pd.read_csv(IDEAS_PATH)
    row = _find_scored_idea_row(ideas)
    if row is None:
        print(
            "No released track-C hypothesis has a matching published score in "
            "lit8d_scores_3seed.csv.gz -- cannot run a parity check."
        )
        return

    published = load_published_score(
        row["idea_model"], row["track"], row["subdomain"], idea_index=row["idea_index"],
    )
    if published is None:
        print(
            f"No published score found for {row['idea_model']} / "
            f"{row['subdomain']} / idea_index {row['idea_index']}"
        )
        return

    hyp = Hypothesis(
        topic=row["subdomain"],
        domain=row["domain"],
        subfield=row["subdomain"],
        text=row["idea_text"],
        source="parity_check",
    )
    result = score_hypothesis(hyp)

    print(f"Subfield: {row['subdomain']} ({row['domain']})")
    print(f"Original model: {row['idea_model']}, track {row['track']}")
    print(
        f"Published score for this EXACT hypothesis (idea_index "
        f"{row['idea_index']}): {published:.2f}"
    )
    print(f"Our re-score of that same hypothesis: {result.weighted_total:.2f}")
    print(f"Difference: {abs(published - result.weighted_total):.2f}")
    print(f"Critics that returned a usable score: {result.n_critics}")
    print("\nPer-dimension (published critics may differ from ours by random selection):")
    for dim, value in result.per_dimension_trimmed.items():
        print(f"  {dim:14s}: {value:.2f}")


if __name__ == "__main__":
    main()
