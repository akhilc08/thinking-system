from unittest.mock import Mock, patch

import pytest

from orchestrator.backend import BackendError, OpenRouterBackend, retry_with_backoff


def test_retry_with_backoff_succeeds_on_first_try():
    fn = Mock(return_value="ok")
    assert retry_with_backoff(fn) == "ok"
    assert fn.call_count == 1


def test_retry_with_backoff_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    fn = Mock(side_effect=[BackendError("boom"), "ok"])
    assert retry_with_backoff(fn, max_attempts=3) == "ok"
    assert fn.call_count == 2


def test_retry_with_backoff_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    fn = Mock(side_effect=BackendError("boom"))
    with pytest.raises(BackendError):
        retry_with_backoff(fn, max_attempts=3)
    assert fn.call_count == 3


def test_openrouter_backend_call_returns_message_content():
    backend = OpenRouterBackend(model="anthropic/claude-sonnet-5", api_key="key123")
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    with patch("orchestrator.backend.requests.post", return_value=fake_response) as mock_post:
        result = backend.call("generation", "sys prompt", "user prompt")

    assert result == "hello"
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer key123"
    assert kwargs["json"]["model"] == "anthropic/claude-sonnet-5"


def test_openrouter_backend_raises_on_non_200(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    backend = OpenRouterBackend(model="m", api_key="key123")
    fake_response = Mock(status_code=500, text="server error")
    with patch("orchestrator.backend.requests.post", return_value=fake_response):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_openrouter_backend_raises_on_malformed_response(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    backend = OpenRouterBackend(model="m", api_key="key123")
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {"unexpected": "shape"}
    with patch("orchestrator.backend.requests.post", return_value=fake_response):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")
