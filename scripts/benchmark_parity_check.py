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

from benchmark.baseline_loader import load_published_score
from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis
from project_config.benchmark import VENDOR_ROOT

IDEAS_PATH = VENDOR_ROOT / "release_data" / "core" / "subdomain_ideas.csv.gz"


def main():
    ideas = pd.read_csv(IDEAS_PATH)
    # Pick the first Active-track (track "C") row with non-empty text.
    row = ideas[(ideas["track"] == "C") & (ideas["idea_text"].str.strip() != "")].iloc[0]

    published = load_published_score(row["idea_model"], row["track"], row["subdomain"])
    if published is None:
        print(f"No published score found for {row['idea_model']} / {row['subdomain']}")
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
    print(f"Published average (across all released hypotheses for this cell): {published:.2f}")
    print(f"Our re-score of this ONE specific hypothesis: {result.weighted_total:.2f}")
    print(f"Difference: {abs(published - result.weighted_total):.2f}")
    print("\nPer-dimension (published critics may differ from ours by random selection):")
    for dim, value in result.per_dimension_trimmed.items():
        print(f"  {dim:14s}: {value:.2f}")


if __name__ == "__main__":
    main()
