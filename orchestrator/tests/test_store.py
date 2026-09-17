from orchestrator.models import Candidate, DebateVerdict, RunConfig
from orchestrator.store import RunStore


def _config(**overrides):
    defaults = dict(topic="t", domain="Computer Science", subfield="t", run_id="run-1")
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_create_run_and_get_status(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    assert store.get_run_status(config.run_id) == "in_progress"


def test_create_run_is_idempotent(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    store.create_run(config)  # should not raise or duplicate
    assert store.get_run_status(config.run_id) == "in_progress"


def test_round_lifecycle_and_resumability(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    assert store.last_completed_round(config.run_id) == 0
    store.record_round_start(config.run_id, 1)
    store.record_round_complete(config.run_id, 1, "guidance for round 2")
    assert store.last_completed_round(config.run_id) == 1

    store.record_round_start(config.run_id, 2)  # round 2 never completes
    assert store.last_completed_round(config.run_id) == 1  # resume point stays at 1


def test_get_last_meta_review_guidance_returns_latest_completed_rounds_guidance(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    assert store.get_last_meta_review_guidance(config.run_id) == ""

    store.record_round_start(config.run_id, 1)
    store.record_round_complete(config.run_id, 1, "guidance from round 1")
    assert store.get_last_meta_review_guidance(config.run_id) == "guidance from round 1"

    store.record_round_start(config.run_id, 2)
    store.record_round_complete(config.run_id, 2, "guidance from round 2")
    assert store.get_last_meta_review_guidance(config.run_id) == "guidance from round 2"

    store.record_round_start(config.run_id, 3)  # round 3 never completes
    assert store.get_last_meta_review_guidance(config.run_id) == "guidance from round 2"


def test_save_candidate_and_get_pool(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    c1 = Candidate(text="idea one", round_num=1, source_strategy="divergent", elo=1210.0)
    c2 = Candidate(text="idea two", round_num=1, source_strategy="divergent", parent_ids=[c1.candidate_id])
    store.save_candidate(config.run_id, c1)
    store.save_candidate(config.run_id, c2)

    pool = store.get_pool(config.run_id)
    assert len(pool) == 2
    by_id = {c.candidate_id: c for c in pool}
    assert by_id[c1.candidate_id].elo == 1210.0
    assert by_id[c2.candidate_id].parent_ids == [c1.candidate_id]


def test_save_debate_records_elo_history(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    verdict = DebateVerdict(candidate_a_id="a", candidate_b_id="b", winner_id="a", reasoning="A is stronger")
    store.save_debate(config.run_id, 1, verdict, (1200.0, 1200.0), (1216.0, 1184.0))
    row = store._conn.execute("SELECT winner_id, elo_a_after, elo_b_after FROM debates").fetchone()
    assert row == ("a", 1216.0, 1184.0)


def test_init_schema_migrates_a_pre_existing_candidates_table(tmp_path):
    db_path = tmp_path / "test.db"
    # Simulate a .data/runs.db created before status/pruned_reason/structured_json existed.
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, run_id TEXT, round_num INTEGER, "
        "text TEXT, source_strategy TEXT, parent_ids TEXT, elo REAL)"
    )
    conn.commit()
    conn.close()

    store = RunStore(db_path)  # should migrate, not crash
    config = _config()
    store.create_run(config)
    c = Candidate(text="idea", round_num=1, source_strategy="divergent", structured={"hypothesis": "h"})
    store.save_candidate(config.run_id, c, status="active", pruned_reason=None)

    detailed = store.get_candidates_detailed(config.run_id)
    assert detailed[0]["status"] == "active"
    assert detailed[0]["structured"] == {"hypothesis": "h"}


def test_save_candidate_tracks_status_and_pruned_reason(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    kept = Candidate(text="idea kept", round_num=1, source_strategy="divergent", elo=1220.0)
    pruned = Candidate(text="idea pruned", round_num=1, source_strategy="divergent", elo=1180.0)
    store.save_candidate(config.run_id, kept, status="active")
    store.save_candidate(config.run_id, pruned, status="pruned_reflection", pruned_reason="incoherent")

    detailed = {c["candidate_id"]: c for c in store.get_candidates_detailed(config.run_id)}
    assert detailed[kept.candidate_id]["status"] == "active"
    assert detailed[kept.candidate_id]["pruned_reason"] is None
    assert detailed[pruned.candidate_id]["status"] == "pruned_reflection"
    assert detailed[pruned.candidate_id]["pruned_reason"] == "incoherent"

    # get_pool still returns every row regardless of status (pruned ideas stay visible)
    assert len(store.get_pool(config.run_id)) == 2


def test_elo_snapshots_recorded_per_round(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    store.save_elo_snapshot(config.run_id, 1, "c1", 1216.0)
    store.save_elo_snapshot(config.run_id, 2, "c1", 1232.0)

    snapshots = store.get_elo_snapshots(config.run_id)
    assert [(s["round_num"], s["elo"]) for s in snapshots] == [(1, 1216.0), (2, 1232.0)]


def test_save_final_selection_and_mark_completed(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    c1 = Candidate(text="idea one", round_num=1, source_strategy="divergent", elo=1300.0)
    store.save_final_selection(config.run_id, [c1])
    store.mark_run_completed(config.run_id, total_model_calls=42, compute_multiplier=4.2)

    row = store._conn.execute(
        "SELECT candidate_id FROM final_selection WHERE run_id = ? AND rank = 1", (config.run_id,)
    ).fetchone()
    assert row[0] == c1.candidate_id
    assert store.get_run_status(config.run_id) == "completed"
