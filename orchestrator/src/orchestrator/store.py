"""SQLite persistence for orchestrator runs (spec §6). Own schema — not
the vendored repo's — designed for the same idempotent/resumable-run
property: a run's status and its last-completed round are checked before
re-running a round, so an interrupted run can pick up where it left off.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from orchestrator.models import Candidate, CritiqueResult, DebateVerdict, RunConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topic TEXT, domain TEXT, subfield TEXT,
    config_json TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress',
    total_model_calls INTEGER DEFAULT 0,
    compute_multiplier REAL
);
CREATE TABLE IF NOT EXISTS rounds (
    run_id TEXT, round_num INTEGER,
    status TEXT NOT NULL DEFAULT 'in_progress',
    meta_review_guidance TEXT,
    PRIMARY KEY (run_id, round_num)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT, round_num INTEGER, text TEXT,
    source_strategy TEXT, parent_ids TEXT, elo REAL,
    status TEXT NOT NULL DEFAULT 'active',
    pruned_reason TEXT,
    structured_json TEXT
);
CREATE TABLE IF NOT EXISTS critiques (
    candidate_id TEXT, run_id TEXT, passed INTEGER, issues TEXT
);
CREATE TABLE IF NOT EXISTS debates (
    run_id TEXT, round_num INTEGER,
    candidate_a_id TEXT, candidate_b_id TEXT, winner_id TEXT, reasoning TEXT,
    elo_a_before REAL, elo_b_before REAL, elo_a_after REAL, elo_b_after REAL
);
CREATE TABLE IF NOT EXISTS final_selection (
    run_id TEXT, rank INTEGER, candidate_id TEXT
);
CREATE TABLE IF NOT EXISTS elo_snapshots (
    run_id TEXT, round_num INTEGER, candidate_id TEXT, elo REAL,
    PRIMARY KEY (run_id, round_num, candidate_id)
);
"""


class RunStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        # CREATE TABLE IF NOT EXISTS only helps brand-new tables; a
        # candidates table created before a column existed needs it added
        # explicitly, or every insert against an older local .data/runs.db
        # fails with "no column named ...".
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(candidates)")}
        additions = {
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "pruned_reason": "TEXT",
            "structured_json": "TEXT",
        }
        for column, decl in additions.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE candidates ADD COLUMN {column} {decl}")
        self._conn.commit()

    def create_run(self, config: RunConfig) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, topic, domain, subfield, config_json) VALUES (?, ?, ?, ?, ?)",
            (config.run_id, config.topic, config.domain, config.subfield, json.dumps(config.__dict__)),
        )
        self._conn.commit()

    def get_run_status(self, run_id: str) -> str | None:
        row = self._conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    def mark_run_completed(self, run_id: str, total_model_calls: int, compute_multiplier: float) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'completed', total_model_calls = ?, compute_multiplier = ? WHERE run_id = ?",
            (total_model_calls, compute_multiplier, run_id),
        )
        self._conn.commit()

    def mark_run_truncated(self, run_id: str, total_model_calls: int, compute_multiplier: float) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'truncated_call_budget', total_model_calls = ?, compute_multiplier = ? WHERE run_id = ?",
            (total_model_calls, compute_multiplier, run_id),
        )
        self._conn.commit()

    def record_round_start(self, run_id: str, round_num: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO rounds (run_id, round_num, status) VALUES (?, ?, 'in_progress')",
            (run_id, round_num),
        )
        self._conn.commit()

    def record_round_complete(self, run_id: str, round_num: int, meta_review_guidance: str) -> None:
        self._conn.execute(
            "UPDATE rounds SET status = 'completed', meta_review_guidance = ? WHERE run_id = ? AND round_num = ?",
            (meta_review_guidance, run_id, round_num),
        )
        self._conn.commit()

    def last_completed_round(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(round_num) FROM rounds WHERE run_id = ? AND status = 'completed'", (run_id,)
        ).fetchone()
        return row[0] if row and row[0] is not None else 0

    def get_last_meta_review_guidance(self, run_id: str) -> str:
        row = self._conn.execute(
            "SELECT meta_review_guidance FROM rounds WHERE run_id = ? AND status = 'completed' "
            "ORDER BY round_num DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else ""

    def save_candidate(
        self, run_id: str, candidate: Candidate, status: str = "active", pruned_reason: str | None = None
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO candidates (candidate_id, run_id, round_num, text, source_strategy, "
            "parent_ids, elo, status, pruned_reason, structured_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id, run_id, candidate.round_num, candidate.text,
                candidate.source_strategy, json.dumps(candidate.parent_ids), candidate.elo,
                status, pruned_reason, json.dumps(candidate.structured or {}),
            ),
        )
        self._conn.commit()

    def save_elo_snapshot(self, run_id: str, round_num: int, candidate_id: str, elo: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO elo_snapshots (run_id, round_num, candidate_id, elo) VALUES (?, ?, ?, ?)",
            (run_id, round_num, candidate_id, elo),
        )
        self._conn.commit()

    def save_critique(self, run_id: str, critique: CritiqueResult) -> None:
        self._conn.execute(
            "INSERT INTO critiques (candidate_id, run_id, passed, issues) VALUES (?, ?, ?, ?)",
            (critique.candidate_id, run_id, int(critique.passed), json.dumps(critique.issues)),
        )
        self._conn.commit()

    def save_debate(
        self, run_id: str, round_num: int, verdict: DebateVerdict,
        elo_before: tuple, elo_after: tuple,
    ) -> None:
        self._conn.execute(
            "INSERT INTO debates (run_id, round_num, candidate_a_id, candidate_b_id, winner_id, reasoning, "
            "elo_a_before, elo_b_before, elo_a_after, elo_b_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, round_num, verdict.candidate_a_id, verdict.candidate_b_id, verdict.winner_id,
                verdict.reasoning, elo_before[0], elo_before[1], elo_after[0], elo_after[1],
            ),
        )
        self._conn.commit()

    def save_final_selection(self, run_id: str, ranked_candidates: list) -> None:
        self._conn.execute("DELETE FROM final_selection WHERE run_id = ?", (run_id,))
        for rank, candidate in enumerate(ranked_candidates, start=1):
            self._conn.execute(
                "INSERT INTO final_selection (run_id, rank, candidate_id) VALUES (?, ?, ?)",
                (run_id, rank, candidate.candidate_id),
            )
        self._conn.commit()

    def get_pool(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT candidate_id, round_num, text, source_strategy, parent_ids, elo, structured_json "
            "FROM candidates WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            Candidate(
                candidate_id=row[0], round_num=row[1], text=row[2], source_strategy=row[3],
                parent_ids=json.loads(row[4]), elo=row[5],
                structured=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT run_id, topic, domain, subfield, status, total_model_calls, compute_multiplier "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return dict(
            run_id=row[0], topic=row[1], domain=row[2], subfield=row[3],
            status=row[4], total_model_calls=row[5], compute_multiplier=row[6],
        )

    def get_rounds(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT round_num, status, meta_review_guidance FROM rounds WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ).fetchall()
        return [dict(round_num=r[0], status=r[1], meta_review_guidance=r[2]) for r in rows]

    def get_candidates_detailed(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT candidate_id, round_num, text, source_strategy, parent_ids, elo, status, pruned_reason, "
            "structured_json FROM candidates WHERE run_id = ? ORDER BY round_num, elo DESC",
            (run_id,),
        ).fetchall()
        return [
            dict(
                candidate_id=r[0], round_num=r[1], text=r[2], source_strategy=r[3],
                parent_ids=json.loads(r[4]), elo=r[5], status=r[6], pruned_reason=r[7],
                structured=json.loads(r[8]) if r[8] else {},
            )
            for r in rows
        ]

    def get_critiques(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT candidate_id, passed, issues FROM critiques WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(candidate_id=r[0], passed=bool(r[1]), issues=json.loads(r[2])) for r in rows]

    def get_debates(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT round_num, candidate_a_id, candidate_b_id, winner_id, reasoning, "
            "elo_a_before, elo_b_before, elo_a_after, elo_b_after FROM debates WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ).fetchall()
        return [
            dict(
                round_num=r[0], candidate_a_id=r[1], candidate_b_id=r[2], winner_id=r[3], reasoning=r[4],
                elo_a_before=r[5], elo_b_before=r[6], elo_a_after=r[7], elo_b_after=r[8],
            )
            for r in rows
        ]

    def get_elo_snapshots(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT round_num, candidate_id, elo FROM elo_snapshots WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ).fetchall()
        return [dict(round_num=r[0], candidate_id=r[1], elo=r[2]) for r in rows]

    def get_final_selection(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT rank, candidate_id FROM final_selection WHERE run_id = ? ORDER BY rank", (run_id,)
        ).fetchall()
        return [dict(rank=r[0], candidate_id=r[1]) for r in rows]

    def close(self) -> None:
        self._conn.close()
