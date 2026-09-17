"""Ranking stage (spec §5.2, §9): pairwise debate + Elo, not absolute
scoring.
"""
import asyncio

from orchestrator.json_utils import call_with_json_retry
from orchestrator.models import DebateVerdict

_SYSTEM_PROMPT = (
    "You are a Ranking agent judging a debate between two research hypotheses, "
    "A and B, on the same topic. Argue which is stronger — more original, "
    "feasible, and impactful — then declare a winner. Respond with ONLY a JSON "
    'object: {"winner": "A"|"B", "reasoning": "<2-3 sentences>"}.'
)


def elo_update(rating_a: float, rating_b: float, winner: str, k_factor: float = 32.0) -> tuple:
    """Update Elo ratings using standard chess formula.

    Args:
        rating_a: Current rating for candidate A
        rating_b: Current rating for candidate B
        winner: "A" or "B" indicating who won the debate
        k_factor: K-factor controlling rating volatility (default 32.0)

    Returns:
        tuple of (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    actual_a = 1.0 if winner == "A" else 0.0
    actual_b = 1.0 - actual_a
    new_a = rating_a + k_factor * (actual_a - expected_a)
    new_b = rating_b + k_factor * (actual_b - expected_b)
    return new_a, new_b


def _debate_one(backend, candidate_a, candidate_b) -> DebateVerdict:
    """Run a single debate between two candidates.

    Args:
        backend: Backend object with call(role, system_prompt, user_prompt) method
        candidate_a: First candidate
        candidate_b: Second candidate

    Returns:
        DebateVerdict with winner_id and reasoning
    """
    user_prompt = f"Hypothesis A: {candidate_a.text}\n\nHypothesis B: {candidate_b.text}"
    parsed = call_with_json_retry(backend, "ranking", _SYSTEM_PROMPT, user_prompt)
    winner_id = candidate_a.candidate_id if parsed["winner"] == "A" else candidate_b.candidate_id
    return DebateVerdict(
        candidate_a_id=candidate_a.candidate_id, candidate_b_id=candidate_b.candidate_id,
        winner_id=winner_id, reasoning=parsed.get("reasoning", ""),
    )


def _apply_verdict(candidate_a, candidate_b, verdict: DebateVerdict) -> None:
    """Apply a debate verdict to update candidate Elo ratings.

    Args:
        candidate_a: First candidate (will be mutated)
        candidate_b: Second candidate (will be mutated)
        verdict: DebateVerdict from the debate
    """
    winner = "A" if verdict.winner_id == candidate_a.candidate_id else "B"
    new_a, new_b = elo_update(candidate_a.elo, candidate_b.elo, winner)
    candidate_a.elo, candidate_b.elo = new_a, new_b


async def rank_round(pool: list, backend, proximity, n_pairs: int, round_robin: bool = False) -> list:
    """Run a round of ranking debates and apply verdicts.

    Conducts pairwise debates between candidates selected by proximity.pick_debate_pairs
    (or, when round_robin is set, every non-duplicate pair via proximity.pick_all_pairs).
    Each debate updates the Elo ratings of both candidates.

    Args:
        pool: List of Candidate objects to rank
        backend: Backend object for LLM calls
        proximity: Object with pick_debate_pairs(pool, n_pairs) and pick_all_pairs(pool)
        n_pairs: Number of debate pairs to conduct (ignored when round_robin is set)
        round_robin: debate every non-duplicate pair once instead of a capped set

    Returns:
        list of DebateVerdict objects from all debates (verdicts are in order of pairs)
    """
    pairs = proximity.pick_all_pairs(pool) if round_robin else proximity.pick_debate_pairs(pool, n_pairs)
    verdicts = list(await asyncio.gather(*[
        asyncio.to_thread(_debate_one, backend, a, b) for a, b in pairs
    ]))
    for (candidate_a, candidate_b), verdict in zip(pairs, verdicts):
        _apply_verdict(candidate_a, candidate_b, verdict)
    return verdicts
