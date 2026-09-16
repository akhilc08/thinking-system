import json
from unittest.mock import patch

from benchmark.reference_fetcher import fetch_paper


def test_fetch_paper_returns_title_abstract_year(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    fake_response = {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture...",
        "year": 2017,
    }
    with patch("benchmark.reference_fetcher.ss_get", return_value=fake_response) as mock_get:
        result = fetch_paper("204e3073870fae3d05bcbc2f6a8e263d9b72e776")

    assert result == fake_response
    mock_get.assert_called_once()


def test_fetch_paper_returns_none_when_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    with patch("benchmark.reference_fetcher.ss_get", return_value=None):
        result = fetch_paper("unknown_id")
    assert result is None


def test_fetch_paper_uses_disk_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    fake_response = {"title": "Cached Paper", "abstract": "abstract text", "year": 2020}
    with patch("benchmark.reference_fetcher.ss_get", return_value=fake_response) as mock_get:
        fetch_paper("some_id")   # first call: hits ss_get, writes cache
        fetch_paper("some_id")   # second call: should read from cache, not call ss_get again
    assert mock_get.call_count == 1
