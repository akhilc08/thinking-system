from orchestrator.models import Candidate, MemoryEntry, RunConfig, new_id


def test_new_id_returns_unique_hex_strings():
    a, b = new_id(), new_id()
    assert a != b
    assert len(a) == 32


def test_candidate_defaults():
    c = Candidate(text="an idea", round_num=1, source_strategy="divergent")
    assert c.elo == 1200.0
    assert c.parent_ids == []
    assert c.embedding is None
    assert c.candidate_id

    c2 = Candidate(text="another idea", round_num=1, source_strategy="divergent")
    assert c.candidate_id != c2.candidate_id


def test_run_config_defaults():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t")
    assert config.backend_kind == "openrouter"
    assert config.rounds == 3
    assert config.k_final == 1
    assert config.round_robin_debates is False
    assert config.run_id


def test_memory_entry_holds_identifying_fields():
    entry = MemoryEntry(domain="Computer Science", subfield="t", text="idea", elo=1250.0, run_id="r1")
    assert entry.domain == "Computer Science"
    assert entry.memory_id
