"""Reflection stage (spec §5.2): cheap per-candidate critique gate before a
candidate enters the ranked pool. Not the final quality signal (that's
Ranking's job) — just a filter against obviously broken output.
"""
import asyncio

from orchestrator.json_utils import call_with_json_retry
from orchestrator.models import CritiqueResult

_SYSTEM_PROMPT = (
    "You are a Reflection agent. Check ONE research hypothesis for coherence, "
    "minimal feasibility, and concreteness (not for whether it's a *good* idea "
    "— just whether it's a well-formed, sane candidate to debate). Respond with "
    'ONLY a JSON object: {"passed": true|false, "issues": ["<issue>", ...]}.'
)


def _reflect_one(backend, candidate) -> CritiqueResult:
    parsed = call_with_json_retry(backend, "reflection", _SYSTEM_PROMPT, f"Hypothesis: {candidate.text}")
    return CritiqueResult(candidate_id=candidate.candidate_id, passed=bool(parsed["passed"]), issues=parsed.get("issues", []))


async def reflect_round(candidates: list, backend) -> list:
    return list(await asyncio.gather(*[
        asyncio.to_thread(_reflect_one, backend, candidate) for candidate in candidates
    ]))


def filter_passed(candidates: list, critiques: list) -> list:
    passed_ids = {c.candidate_id for c in critiques if c.passed}
    return [c for c in candidates if c.candidate_id in passed_ids]
