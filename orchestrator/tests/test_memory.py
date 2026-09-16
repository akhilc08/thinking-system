from unittest.mock import Mock

from orchestrator.memory import MemoryStore
from orchestrator.models import MemoryEntry


def test_query_returns_entries_for_domain_and_subfield(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    entry = MemoryEntry(domain="Computer Science", subfield="gnn expressiveness", text="idea", elo=1250.0, run_id="r1")
    store.write(entry, proximity=Mock(is_near_duplicate=Mock(return_value=False)))

    results = store.query("Computer Science", "gnn expressiveness")
    assert len(results) == 1
    assert results[0].text == "idea"

    assert store.query("Computer Science", "other subfield") == []


def test_write_skips_near_duplicate(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    existing = MemoryEntry(
        domain="Computer Science", subfield="t", text="existing idea", elo=1200.0, run_id="r1", embedding=[1.0, 0.0]
    )
    store.write(existing, proximity=Mock(is_near_duplicate=Mock(return_value=False)))

    proximity = Mock(is_near_duplicate=Mock(return_value=True))
    new_entry = MemoryEntry(
        domain="Computer Science", subfield="t", text="near-duplicate idea", elo=1210.0, run_id="r2", embedding=[1.0, 0.0001]
    )
    written = store.write(new_entry, proximity)

    assert written is False
    assert len(store.query("Computer Science", "t")) == 1


def test_write_stores_a_non_duplicate_entry(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    entry = MemoryEntry(domain="Computer Science", subfield="t", text="idea", elo=1200.0, run_id="r1", embedding=[1.0, 0.0])
    written = store.write(entry, proximity=Mock(is_near_duplicate=Mock(return_value=False)))
    assert written is True
    assert len(store.query("Computer Science", "t")) == 1
