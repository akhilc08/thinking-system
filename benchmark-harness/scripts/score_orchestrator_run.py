"""Scores a completed orchestrator/ run's ideas using AgentIdeaBench's real
critic pipeline (benchmark.scorer.score_hypothesis) -- the same rubric
docs/research/design-recommendations.md calls for comparing the swarm's
output against baselines, and the same one AgentIdeaBench's own published
leaderboard uses (drop-highest-of-3 critics, weighted dimension sum).

Deliberately does NOT import the orchestrator package: it reads the plain
JSON an orchestrator run writes via `python -m orchestrator.cli --export
path.json`, so the two systems stay independent (see
docs/design/plans/2026-09-16-orchestrator.md's standalone constraint).
That JSON is the data contract between them.

Each idea scored here is 3 real OpenRouter critic calls plus a Semantic
Scholar evidence lookup -- unlike the orchestrator's own claude_code-backend
runs, this is billed API usage, not a Claude subscription. Check
OPENROUTER_API_KEY is set (repo-root .env) and be deliberate about how many
ideas you score.

Usage:
    python scripts/score_orchestrator_run.py <export.json> [--which final|all] [--offset N] [--limit N] [--cutoff-date YYYY-MM-DD] [--out results.json]

--offset/--limit slice the target list (after --which filtering) so a full
run can be scored in checkpointed batches, e.g. three calls with
--limit 11 --offset 0/11/22 for a 33-idea run.
"""
import argparse
import json

from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis


def score_run_export(
    export_path: str, which: str = "final", cutoff_date: str = "2026-05-31",
    offset: int = 0, limit: int | None = None,
) -> list:
    with open(export_path) as f:
        data = json.load(f)

    run = data["run"]
    by_id = {c["candidate_id"]: c for c in data["candidates"]}
    if which == "final":
        targets = [by_id[f["candidate_id"]] for f in data["final_selection"]]
    else:
        targets = data["candidates"]

    if limit is not None:
        targets = targets[offset:offset + limit]
    else:
        targets = targets[offset:]

    results = []
    for candidate in targets:
        hypothesis = Hypothesis(
            topic=run["topic"], domain=run["domain"], subfield=run["subfield"],
            text=candidate["text"], source="our_swarm",
        )
        result = score_hypothesis(hypothesis, cutoff_date=cutoff_date)
        results.append((candidate, result))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_path", help="Path to an orchestrator run's --export JSON")
    parser.add_argument("--which", choices=["final", "all"], default="final",
                         help="Score only the final-selected ideas (default), or every idea the run ever produced")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many targets before scoring (for batching)")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many targets (for batching)")
    parser.add_argument("--cutoff-date", default="2026-05-31")
    parser.add_argument("--out", help="Append {candidate_id: {weighted_total, per_dimension_trimmed, n_critics}} to this JSON file")
    args = parser.parse_args()

    results = score_run_export(
        args.export_path, which=args.which, cutoff_date=args.cutoff_date,
        offset=args.offset, limit=args.limit,
    )

    for candidate, result in results:
        print(f"\n[{candidate['candidate_id'][:8]}] elo {candidate['elo']:.0f} -> weighted {result.weighted_total:.2f} "
              f"(n_critics={result.n_critics})")
        for dim, score in result.per_dimension_trimmed.items():
            print(f"    {dim}: {score:.2f}")

    if results:
        avg = sum(r.weighted_total for _, r in results) / len(results)
        print(f"\nBatch average weighted score across {len(results)} idea(s): {avg:.2f}")

    if args.out:
        existing = {}
        try:
            with open(args.out) as f:
                existing = json.load(f)
        except FileNotFoundError:
            pass
        for candidate, result in results:
            existing[candidate["candidate_id"]] = {
                "weighted_total": result.weighted_total,
                "per_dimension_trimmed": result.per_dimension_trimmed,
                "n_critics": result.n_critics,
            }
        with open(args.out, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Appended to {args.out} ({len(existing)} ideas scored so far)")


if __name__ == "__main__":
    main()
