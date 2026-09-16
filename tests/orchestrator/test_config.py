import importlib


def test_config_defaults_to_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "ss-key-456")
    from config import orchestrator
    importlib.reload(orchestrator)

    assert orchestrator.OPENROUTER_API_KEY == "test-key-123"
    assert orchestrator.SEMANTIC_SCHOLAR_API_KEY == "ss-key-456"
    assert orchestrator.DB_PATH.name == "runs.db"
    assert orchestrator.CLAUDE_CLI_PATH == "claude"
    assert orchestrator.OPENROUTER_BASE_URL.startswith("https://")
    assert isinstance(orchestrator.DEFAULT_EMBEDDING_MODEL, str)
