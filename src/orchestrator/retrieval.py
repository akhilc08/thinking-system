"""Standalone Semantic Scholar search wrapper for the orchestrator's
retrieval-grounded Generation and Evolution stages (spec §5.2) —
independent of vendor/AgentIdeaBench's retrieval code, which belongs to
the benchmark harness only.
"""
from dataclasses import dataclass

import requests

from config.orchestrator import SEMANTIC_SCHOLAR_API_KEY

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,citationCount,url"


@dataclass
class PaperHit:
    title: str
    abstract: str
    year: int
    citation_count: int
    url: str


def search_literature(query: str, limit: int = 5, api_key: str = SEMANTIC_SCHOLAR_API_KEY) -> list:
    headers = {"x-api-key": api_key} if api_key else {}
    response = requests.get(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        params={"query": query, "limit": limit, "fields": FIELDS},
        headers=headers,
        timeout=15.0,
    )
    if response.status_code != 200:
        return []
    data = response.json()
    hits = []
    for paper in data.get("data", []):
        hits.append(PaperHit(
            title=paper.get("title") or "",
            abstract=paper.get("abstract") or "",
            year=paper.get("year") or 0,
            citation_count=paper.get("citationCount") or 0,
            url=paper.get("url") or "",
        ))
    return hits


def format_papers_for_prompt(hits: list) -> str:
    if not hits:
        return "No relevant papers found."
    lines = ["PRIOR LITERATURE:"]
    for hit in hits:
        lines.append(f"- {hit.title} ({hit.year}, {hit.citation_count} citations): {hit.abstract[:300]}")
    return "\n".join(lines)
