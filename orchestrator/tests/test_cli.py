from unittest.mock import Mock, patch

from orchestrator.cli import build_backend, main


def test_build_backend_openrouter():
    from orchestrator.backend import OpenRouterBackend
    backend = build_backend("openrouter")
    assert isinstance(backend, OpenRouterBackend)


def test_build_backend_claude_code():
    from orchestrator.claude_code_backend import ClaudeCodeBackend
    backend = build_backend("claude_code")
    assert isinstance(backend, ClaudeCodeBackend)


def test_main_wires_args_into_run_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["orchestrator", "--topic", "t", "--domain", "Computer Science", "--subfield", "t", "--rounds", "2", "--k", "3"],
    )
    fake_summary = Mock(run_id="r1", status="completed", total_model_calls=10, compute_multiplier=1.0, final_candidates=[])

    with patch("orchestrator.cli.build_backend", return_value=Mock()), \
         patch("orchestrator.cli.RunStore", return_value=Mock()), \
         patch("orchestrator.cli.MemoryStore", return_value=Mock()), \
         patch("orchestrator.cli.asyncio.run", return_value=fake_summary) as mock_run:
        main()

    assert mock_run.called
