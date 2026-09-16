"""Grounded-originality evidence, built the same way AgentIdeaBench builds it
for its own published lit8d scores: extract search queries from the
hypothesis, run a date-filtered Semantic Scholar search, format the hits.
"""
from benchmark.vendor_bootstrap import build_evidence_for_item, format_evidence_block


def build_evidence(item_id: str, idea_text: str, cutoff_date: str) -> str:
    _queries, hits, _n_self = build_evidence_for_item(
        item_id, "v4", idea_text, cutoff_date, None, None,
    )
    return format_evidence_block(hits, cutoff_date)
