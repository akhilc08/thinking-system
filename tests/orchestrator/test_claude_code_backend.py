import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from orchestrator.backend import BackendError
from orchestrator.claude_code_backend import ClaudeCodeBackend


def _fake_result(returncode=0, stdout="", stderr=""):
    return Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_call_returns_result_field_from_cli_json_output():
    backend = ClaudeCodeBackend(cli_path="claude")
    fake = _fake_result(stdout=json.dumps({"result": "a generated idea"}))
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake) as mock_run:
        result = backend.call("generation", "sys prompt", "user prompt")

    assert result == "a generated idea"
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd


def test_call_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(returncode=1, stderr="boom")
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_on_malformed_json_stdout(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(stdout="not json")
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_when_result_key_missing(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(stdout=json.dumps({"unexpected": "shape"}))
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_on_subprocess_timeout(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend(timeout=5.0)
    with patch(
        "orchestrator.claude_code_backend.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5.0),
    ):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")
