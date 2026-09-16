import importlib


def test_config_defaults_to_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "ss-key-456")
    import config
    importlib.reload(config)

    assert config.OPENROUTER_API_KEY == "test-key-123"
    assert config.SEMANTIC_SCHOLAR_API_KEY == "ss-key-456"
    assert config.DB_PATH.name == "runs.db"
    assert config.CLAUDE_CLI_PATH == "claude"
    assert config.OPENROUTER_BASE_URL.startswith("https://")
    assert isinstance(config.DEFAULT_EMBEDDING_MODEL, str)
