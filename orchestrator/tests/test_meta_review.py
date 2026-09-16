from unittest.mock import Mock

from orchestrator.models import Candidate, CritiqueResult, DebateVerdict
from orchestrator.meta_review import synthesize_meta_review


def test_synthesize_meta_review_calls_backend_with_round_summary():
    candidates = [Candidate(text="idea A", round_num=1, source_strategy="divergent")]
    critiques = [CritiqueResult(candidate_id="c1", passed=False, issues=["too vague"])]
    debates = [DebateVerdict(candidate_a_id="a", candidate_b_id="b", winner_id="a", reasoning="A was more feasible")]

    backend = Mock()
    backend.call.return_value = "Next round: push for more concrete, testable framing."

    guidance = synthesize_meta_review(backend, candidates, critiques, debates)

    assert guidance == "Next round: push for more concrete, testable framing."
    role, system_prompt, user_prompt = backend.call.call_args[0]
    assert role == "meta_review"
    assert "too vague" in user_prompt
    assert "more feasible" in user_prompt
