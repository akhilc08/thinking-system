"""Runs a batch of AgentIdeaBench's real subfields through the orchestrator
swarm, scores each subfield's final-3 picks with the identical critic
rubric, and compares against the published Claude Sonnet 5 score for that
exact subfield -- the apples-to-apples comparison discussed in this
session (not a made-up topic, and not the aggregate 40-subfield leaderboard
number, but per-subfield cells that sum to it).

Coordinates two independent venvs by path (orchestrator/.venv,
benchmark-harness/.venv) via subprocess -- this script itself imports
neither package, keeping both standalone.

Usage:
    python scripts/run_subfield_sweep.py --start 0 --count 5
    python scripts/run_subfield_sweep.py --subfields "large language model reasoning chain of thought,quantum sensing magnetometer atomic clock precision" --tag exp1
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ORCH_DIR = REPO_ROOT / "orchestrator"
BENCH_DIR = REPO_ROOT / "benchmark-harness"
ORCH_PY = ORCH_DIR / ".venv" / "bin" / "python"
BENCH_PY = BENCH_DIR / ".venv" / "bin" / "python"
SUBFIELDS_FILE = ORCH_DIR / "runs" / "all_40_subfields.json"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def run_generation(domain: str, subfield: str, export_path: Path) -> dict:
    cmd = [
        str(ORCH_PY), "-m", "orchestrator.cli",
        "--topic", subfield, "--domain", domain, "--subfield", subfield,
        "--backend", "claude_code", "--rounds", "3", "--k", "3", "--pool-size", "4",
        "--export", str(export_path),
    ]
    env = {"PYTHONPATH": "src:."}
    import os
    env = {**os.environ, **env}
    result = subprocess.run(cmd, cwd=str(ORCH_DIR), env=env, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def run_scoring(export_path: Path, scores_out: Path) -> dict:
    cmd = [
        str(BENCH_PY), "scripts/score_orchestrator_run.py",
        str(export_path), "--which", "final", "--out", str(scores_out),
    ]
    import os
    env = {**os.environ, "PYTHONPATH": "src:."}
    result = subprocess.run(cmd, cwd=str(BENCH_DIR), env=env, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def published_scores(domain: str, subfield: str) -> dict:
    code = (
        "from benchmark.baseline_loader import load_published_score\n"
        f"print(load_published_score('azure/anthropic/claude-sonnet-5', 'B', {subfield!r}))\n"
        f"print(load_published_score('azure/anthropic/claude-sonnet-5', 'C', {subfield!r}))\n"
    )
    import os
    env = {**os.environ, "PYTHONPATH": "src:."}
    result = subprocess.run([str(BENCH_PY), "-c", code], cwd=str(BENCH_DIR), env=env, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    static = float(lines[0]) if len(lines) > 0 and lines[0] != "None" else None
    active = float(lines[1]) if len(lines) > 1 and lines[1] != "None" else None
    return {"static": static, "active": active}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--subfields", help="Comma-separated exact subdomain names to run instead of --start/--count")
    parser.add_argument("--tag", default="sweep", help="Namespaces export/summary files under runs/<tag>/ (default 'sweep', the real 40-subfield run)")
    args = parser.parse_args()

    sweep_dir = ORCH_DIR / "runs" / args.tag
    summary_file = sweep_dir / "summary.json"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_subfields = json.loads(SUBFIELDS_FILE.read_text())
    if args.subfields:
        wanted = {s.strip() for s in args.subfields.split(",")}
        by_name = {item["subdomain"]: item for item in all_subfields}
        missing = wanted - set(by_name)
        if missing:
            print(f"Unknown subfield(s), not in the real 40: {missing}")
            sys.exit(1)
        batch = [by_name[s] for s in wanted]
    else:
        batch = all_subfields[args.start:args.start + args.count]

    summary = json.loads(summary_file.read_text()) if summary_file.exists() else []
    done_subfields = {row["subdomain"] for row in summary if row.get("status") == "done"}

    def record(row: dict) -> None:
        # Replace any prior (failed) attempt for this subfield instead of appending a duplicate.
        nonlocal summary
        summary = [r for r in summary if r["subdomain"] != row["subdomain"]] + [row]
        summary_file.write_text(json.dumps(summary, indent=2))

    for item in batch:
        domain, subfield = item["domain"], item["subdomain"]
        if subfield in done_subfields:
            print(f"SKIP (already done): {domain} | {subfield}")
            continue

        s = slug(subfield)
        export_path = sweep_dir / f"{s}_export.json"
        scores_path = sweep_dir / "scores.json"

        print(f"\n=== {domain} | {subfield} ===")
        print("generating...")
        gen = run_generation(domain, subfield, export_path)
        if (gen["returncode"] != 0 or not export_path.exists()):
            print("generation failed once, retrying...")
            gen = run_generation(domain, subfield, export_path)
        if gen["returncode"] != 0 or not export_path.exists():
            print("GENERATION FAILED (twice):", gen["stderr"][-1500:])
            record({"domain": domain, "subdomain": subfield, "status": "generation_failed"})
            continue

        print("scoring...")
        sc = run_scoring(export_path, scores_path)
        if sc["returncode"] != 0:
            print("SCORING FAILED:", sc["stderr"][-1500:])
            record({"domain": domain, "subdomain": subfield, "status": "scoring_failed"})
            continue

        run_data = json.loads(export_path.read_text())
        final_ids = [f["candidate_id"] for f in run_data["final_selection"]]
        all_scores = json.loads(scores_path.read_text())
        our_scores = [all_scores[cid]["weighted_total"] for cid in final_ids if cid in all_scores]
        our_avg = sum(our_scores) / len(our_scores) if our_scores else None

        pub = published_scores(domain, subfield)

        row = {
            "domain": domain, "subdomain": subfield,
            "our_final3_avg": our_avg,
            "published_static_sonnet5": pub["static"],
            "published_active_sonnet5": pub["active"],
            "total_model_calls": run_data["run"]["total_model_calls"],
            "status": "done",
        }
        record(row)

        print(f"our_final3_avg={our_avg} | published_static={pub['static']} | published_active={pub['active']} "
              f"| calls={row['total_model_calls']}")

    print(f"\nBatch done. {len(summary)} subfield(s) recorded so far in {summary_file}")


if __name__ == "__main__":
    main()
