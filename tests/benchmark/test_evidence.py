from unittest.mock import patch

from benchmark.evidence import build_evidence


def test_build_evidence_formats_retrieved_prior_art():
    fake_queries = ["contrastive learning molecular property prediction"]
    fake_hits = [{
        "paperId": "abc123",
        "title": "Contrastive Pretraining for Molecules",
        "year": 2023,
        "publicationDate": "2023-04-01",
        "citationCount": 42,
        "abstract": "We introduce a contrastive pretraining method...",
        "query": fake_queries[0],
    }]
    with patch(
        "benchmark.evidence.build_evidence_for_item",
        return_value=(fake_queries, fake_hits, 0),
    ) as mock_build:
        block = build_evidence(
            item_id="our_swarm|graph-nn-expressiveness",
            idea_text="We hypothesize that...",
            cutoff_date="2026-05-31",
        )

    mock_build.assert_called_once_with(
        "our_swarm|graph-nn-expressiveness", "v4", "We hypothesize that...",
        "2026-05-31", None, None,
    )
    assert "Contrastive Pretraining for Molecules" in block
    assert "2026-05-31" in block


def test_build_evidence_reports_no_papers_found():
    with patch("benchmark.evidence.build_evidence_for_item", return_value=([], [], 0)):
        block = build_evidence("item", "idea text", "2026-05-31")
    assert "no relevant papers" in block.lower()
