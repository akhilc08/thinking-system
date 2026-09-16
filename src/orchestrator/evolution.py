"""Evolution stage (spec §5.2, §9): five named mutation strategies applied
to the top of the ranked pool each round. Deliberate divergence is
mandatory every round — the only lever in the pipeline that reliably
produces novelty (AgentIdeaBench found retrieval improves
feasibility/clarity but not originality).
"""
import asyncio

from orchestrator.json_utils import call_with_json_retry
from orchestrator.models import Candidate
from orchestrator.retrieval import format_papers_for_prompt, search_literature

_RESPONSE_FORMAT = 'Respond with ONLY a JSON object: {"idea": "<the revised/new hypothesis>"}.'


def _combine(backend, top: list, round_num: int) -> Candidate:
    prompt = (
        "Combine the strongest sub-parts of these hypotheses into one better hypothesis:\n"
        + "\n".join(f"- {c.text}" for c in top) + f"\n\n{_RESPONSE_FORMAT}"
    )
    parsed = call_with_json_retry(backend, "evolution_combine", "You are an Evolution agent: combine.", prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="combine", parent_ids=[c.candidate_id for c in top])


def _simplify(backend, candidate, round_num: int) -> Candidate:
    prompt = f"Rewrite this hypothesis to be simpler and more directly testable:\n{candidate.text}\n\n{_RESPONSE_FORMAT}"
    parsed = call_with_json_retry(backend, "evolution_simplify", "You are an Evolution agent: simplify.", prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="simplify", parent_ids=[candidate.candidate_id])


def _ground_in_literature(backend, candidate, topic: str, round_num: int) -> Candidate:
    literature_block = format_papers_for_prompt(search_literature(topic))
    prompt = f"Strengthen this hypothesis using the literature below:\n{candidate.text}\n\n{literature_block}\n\n{_RESPONSE_FORMAT}"
    parsed = call_with_json_retry(backend, "evolution_ground", "You are an Evolution agent: ground in literature.", prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="ground_in_literature", parent_ids=[candidate.candidate_id])


def _analogical(backend, candidate, round_num: int) -> Candidate:
    prompt = f"Reframe this hypothesis by analogy to a mechanism from a different scientific domain:\n{candidate.text}\n\n{_RESPONSE_FORMAT}"
    parsed = call_with_json_retry(backend, "evolution_analogical", "You are an Evolution agent: analogical reasoning.", prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="analogical", parent_ids=[candidate.candidate_id])


def _deliberate_divergence(backend, pool: list, topic: str, round_num: int) -> Candidate:
    avoid_block = "\n".join(f"- {c.text}" for c in pool[:10])
    prompt = (
        f"Topic: {topic}\nPropose a hypothesis that is DELIBERATELY DIFFERENT in approach from all of these "
        f"(different mechanism, different methodology, or different framing entirely):\n{avoid_block}\n\n{_RESPONSE_FORMAT}"
    )
    parsed = call_with_json_retry(backend, "evolution_divergence", "You are an Evolution agent: deliberate divergence.", prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="deliberate_divergence", parent_ids=[])


async def evolve_round(pool: list, backend, round_num: int, topic: str) -> list:
    top = sorted(pool, key=lambda c: c.elo, reverse=True)
    top3 = top[:3]
    best = top[0]

    tasks = [asyncio.to_thread(_deliberate_divergence, backend, pool, topic, round_num)]
    if len(top3) >= 2:
        tasks.append(asyncio.to_thread(_combine, backend, top3, round_num))
    tasks.append(asyncio.to_thread(_simplify, backend, best, round_num))
    tasks.append(asyncio.to_thread(_ground_in_literature, backend, best, topic, round_num))
    tasks.append(asyncio.to_thread(_analogical, backend, best, round_num))

    return list(await asyncio.gather(*tasks))
