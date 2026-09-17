"""Assembles a full, human-inspectable history of one run from RunStore:
every candidate ever produced (including pruned ones and why), every
debate with its Elo delta, and a per-round Elo snapshot — so a run can be
audited after the fact, not just its final top-k output.
"""


def export_run(store, run_id: str) -> dict:
    return {
        "run": store.get_run(run_id),
        "rounds": store.get_rounds(run_id),
        "candidates": store.get_candidates_detailed(run_id),
        "critiques": store.get_critiques(run_id),
        "debates": store.get_debates(run_id),
        "elo_snapshots": store.get_elo_snapshots(run_id),
        "final_selection": store.get_final_selection(run_id),
    }
