"""Cross-run persistent memory (spec §5.4, §6): keyed by (domain,
subfield), deduplicated on write using the same Proximity mechanism as the
live pool so the store doesn't degrade into near-duplicate clutter over
time (a gap EvoScientist's memory design has, per docs/research).
"""
import json
import sqlite3
from pathlib import Path

from orchestrator.models import Candidate, MemoryEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    memory_id TEXT PRIMARY KEY,
    domain TEXT, subfield TEXT, text TEXT, elo REAL, run_id TEXT,
    embedding TEXT
);
"""


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def query(self, domain: str, subfield: str) -> list:
        rows = self._conn.execute(
            "SELECT memory_id, domain, subfield, text, elo, run_id, embedding FROM memory "
            "WHERE domain = ? AND subfield = ?",
            (domain, subfield),
        ).fetchall()
        return [
            MemoryEntry(
                memory_id=row[0], domain=row[1], subfield=row[2], text=row[3], elo=row[4],
                run_id=row[5], embedding=json.loads(row[6]) if row[6] else None,
            )
            for row in rows
        ]

    def write(self, entry: MemoryEntry, proximity, threshold: float = 0.92) -> bool:
        existing = self.query(entry.domain, entry.subfield)
        if entry.embedding is not None and existing:
            as_candidates = [
                Candidate(text=e.text, round_num=0, source_strategy="memory", embedding=e.embedding)
                for e in existing
            ]
            probe = Candidate(text=entry.text, round_num=0, source_strategy="memory", embedding=entry.embedding)
            if proximity.is_near_duplicate(probe, as_candidates, threshold):
                return False

        self._conn.execute(
            "INSERT OR REPLACE INTO memory (memory_id, domain, subfield, text, elo, run_id, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.memory_id, entry.domain, entry.subfield, entry.text, entry.elo, entry.run_id,
                json.dumps(entry.embedding) if entry.embedding is not None else None,
            ),
        )
        self._conn.commit()
        return True

    def close(self) -> None:
        self._conn.close()
