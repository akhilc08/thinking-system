from unittest.mock import Mock, patch

from orchestrator.retrieval import format_papers_for_prompt, search_literature


def test_search_literature_parses_hits():
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {
        "data": [
            {
                "title": "Contrastive Pretraining for Molecules", "abstract": "We introduce...",
                "year": 2023, "citationCount": 42, "url": "https://example.com/paper",
            }
        ]
    }
    with patch("orchestrator.retrieval.requests.get", return_value=fake_response) as mock_get:
        hits = search_literature("contrastive molecular pretraining", limit=5, api_key="key")

    assert len(hits) == 1
    assert hits[0].title == "Contrastive Pretraining for Molecules"
    assert hits[0].citation_count == 42
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["query"] == "contrastive molecular pretraining"


def test_search_literature_returns_empty_list_on_error():
    fake_response = Mock(status_code=500)
    with patch("orchestrator.retrieval.requests.get", return_value=fake_response):
        hits = search_literature("some query")
    assert hits == []


def test_format_papers_for_prompt_reports_no_papers_found():
    assert "no relevant papers" in format_papers_for_prompt([]).lower()


def test_format_papers_for_prompt_includes_titles():
    from orchestrator.retrieval import PaperHit
    hit = PaperHit(title="A Paper", abstract="short abstract text here", year=2024, citation_count=10, url="u")
    block = format_papers_for_prompt([hit])
    assert "A Paper" in block
