"""Refetches paper title/abstract/year by Semantic Scholar paper ID.

AgentIdeaBench's release ships only ss_paper_id values (abstracts are
publisher content their API license doesn't let them redistribute) — this
module refetches the text ourselves, under our own Semantic Scholar API
agreement, and caches it to disk since paper text never changes.
"""
import json

from benchmark.vendor_bootstrap import SS_BASE, ss_get, vendor_config
from project_config.benchmark import CACHE_DIR

FIELDS = "title,abstract,year"


def fetch_paper(ss_paper_id: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ss_paper_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    result = ss_get(
        f"{SS_BASE}/paper/{ss_paper_id}",
        {"fields": FIELDS},
        vendor_config.SEMANTIC_SCHOLAR_API_KEY,
        0.2,
    )
    if not result:
        return None

    paper = {
        "title": result.get("title") or "",
        "abstract": result.get("abstract") or "",
        "year": result.get("year"),
    }
    cache_path.write_text(json.dumps(paper))
    return paper
