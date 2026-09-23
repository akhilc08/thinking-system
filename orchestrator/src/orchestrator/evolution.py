"""Evolution stage (spec §5.2, §9): named mutation strategies applied to the
top of the ranked pool each round. Deliberate divergence is mandatory every
round — the only lever in the pipeline that reliably produces novelty
(AgentIdeaBench found retrieval improves feasibility/clarity but not
originality).

`simplify` was dropped (2026-09-23): across 39 real AgentIdeaBench subfields
it had a 0% win rate (0/117 selected as a final idea), the only strategy
with zero wins, vs. `combine`'s 34.2% -- the rubric weights Originality 2x
and Specificity/Impact heavily, and simplifying an idea works directly
against that. Replaced with a second `combine` pass over a different
ranking window, since `combine` was the strongest performer by a wide
margin.
"""
import asyncio

from orchestrator.idea_schema import RESPONSE_FORMAT, render_idea
from orchestrator.json_utils import call_with_json_retry
from orchestrator.models import Candidate
from orchestrator.retrieval import format_papers_for_prompt, search_literature


def _combine(backend, top: list, round_num: int) -> Candidate:
    prompt = (
        "Combine the strongest sub-parts of these hypotheses into one better hypothesis:\n"
        + "\n\n".join(f"- {c.text}" for c in top) + f"\n\n{RESPONSE_FORMAT}"
    )
    parsed = call_with_json_retry(backend, "evolution_combine", "You are an Evolution agent: combine.", prompt)
    return Candidate(
        text=render_idea(parsed), round_num=round_num, source_strategy="combine",
        parent_ids=[c.candidate_id for c in top], structured=parsed,
    )


def _ground_in_literature(backend, candidate, topic: str, round_num: int) -> Candidate:
    literature_block = format_papers_for_prompt(search_literature(topic))
    prompt = f"Strengthen this hypothesis using the literature below:\n{candidate.text}\n\n{literature_block}\n\n{RESPONSE_FORMAT}"
    parsed = call_with_json_retry(backend, "evolution_ground", "You are an Evolution agent: ground in literature.", prompt)
    return Candidate(
        text=render_idea(parsed), round_num=round_num, source_strategy="ground_in_literature",
        parent_ids=[candidate.candidate_id], structured=parsed,
    )


def _analogical(backend, candidate, round_num: int) -> Candidate:
    prompt = f"Reframe this hypothesis by analogy to a mechanism from a different scientific domain:\n{candidate.text}\n\n{RESPONSE_FORMAT}"
    parsed = call_with_json_retry(backend, "evolution_analogical", "You are an Evolution agent: analogical reasoning.", prompt)
    return Candidate(
        text=render_idea(parsed), round_num=round_num, source_strategy="analogical",
        parent_ids=[candidate.candidate_id], structured=parsed,
    )


def _deliberate_divergence(backend, pool: list, topic: str, round_num: int) -> Candidate:
    avoid_block = "\n\n".join(f"- {c.text}" for c in pool[:10])
    prompt = (
        f"Topic: {topic}\nPropose a hypothesis that is DELIBERATELY DIFFERENT in approach from all of these "
        f"(different mechanism, different methodology, or different framing entirely):\n{avoid_block}\n\n{RESPONSE_FORMAT}"
    )
    parsed = call_with_json_retry(backend, "evolution_divergence", "You are an Evolution agent: deliberate divergence.", prompt)
    return Candidate(
        text=render_idea(parsed), round_num=round_num, source_strategy="deliberate_divergence",
        parent_ids=[], structured=parsed,
    )


async def evolve_round(pool: list, backend, round_num: int, topic: str) -> list:
    if not pool:
        return []
    top = sorted(pool, key=lambda c: c.elo, reverse=True)
    top3 = top[:3]
    best = top[0]

    tasks = [asyncio.to_thread(_deliberate_divergence, backend, pool, topic, round_num)]
    if len(top3) >= 2:
        tasks.append(asyncio.to_thread(_combine, backend, top3, round_num))
    if len(top) >= 4:
        # A second combine over a distinct window (ranks 2-4), not a repeat
        # of the first -- combine was the strongest strategy by a wide
        # margin (see module docstring), so it gets a second pass instead
        # of the dropped `simplify`.
        tasks.append(asyncio.to_thread(_combine, backend, top[1:4], round_num))
    tasks.append(asyncio.to_thread(_ground_in_literature, backend, best, topic, round_num))
    tasks.append(asyncio.to_thread(_analogical, backend, best, round_num))

    return list(await asyncio.gather(*tasks))
