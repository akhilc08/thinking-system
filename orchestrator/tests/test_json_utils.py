import pytest

from orchestrator.json_utils import JSONRetryExhausted, call_with_json_retry, extract_json


def test_extract_json_parses_raw_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_parses_fenced_code_block():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_parses_bare_braces_in_prose():
    raw = 'Sure, my answer is {"a": 1} - let me know if you need more.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_returns_none_for_unparseable_text():
    assert extract_json("no json here at all") is None


class _FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, role, system_prompt, user_prompt):
        self.calls.append((role, system_prompt, user_prompt))
        return self.responses.pop(0)


def test_call_with_json_retry_succeeds_on_first_try():
    backend = _FakeBackend(['{"idea": "x"}'])
    result = call_with_json_retry(backend, "generation", "sys", "usr")
    assert result == {"idea": "x"}
    assert len(backend.calls) == 1


def test_call_with_json_retry_retries_after_malformed_response():
    backend = _FakeBackend(["not json", '{"idea": "x"}'])
    result = call_with_json_retry(backend, "generation", "sys", "usr")
    assert result == {"idea": "x"}
    assert len(backend.calls) == 2
    assert "could not be parsed" in backend.calls[1][2]


def test_call_with_json_retry_raises_after_max_attempts():
    backend = _FakeBackend(["not json", "still not json", "nope"])
    with pytest.raises(JSONRetryExhausted):
        call_with_json_retry(backend, "generation", "sys", "usr", max_attempts=3)
    assert len(backend.calls) == 3
