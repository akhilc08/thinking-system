import json

import pytest

from orchestrator.models import Candidate, CritiqueResult
from orchestrator.reflection import filter_passed, reflect_round


class _FakeBackend:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def call(self, role, system_prompt, user_prompt):
        return json.dumps(self.verdicts.pop(0))


@pytest.mark.asyncio
async def test_reflect_round_returns_one_critique_per_candidate():
    candidates = [
        Candidate(text="idea A", round_num=1, source_strategy="divergent"),
        Candidate(text="idea B", round_num=1, source_strategy="divergent"),
    ]
    backend = _FakeBackend([
        {"passed": True, "issues": []},
        {"passed": False, "issues": ["too vague"]},
    ])
    critiques = await reflect_round(candidates, backend)
    assert len(critiques) == 2
    assert {c.passed for c in critiques} == {True, False}


def test_filter_passed_keeps_only_passing_candidates():
    a = Candidate(text="idea A", round_num=1, source_strategy="divergent")
    b = Candidate(text="idea B", round_num=1, source_strategy="divergent")
    critiques = [
        CritiqueResult(candidate_id=a.candidate_id, passed=True),
        CritiqueResult(candidate_id=b.candidate_id, passed=False, issues=["too vague"]),
    ]
    result = filter_passed([a, b], critiques)
    assert result == [a]
