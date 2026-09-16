"""Generation stage (spec §5.2): divergent(N)->convergent(k) per round.
Each candidate is retrieval-grounded and checked against memory before
being kept, so the swarm doesn't re-propose ideas it already tried.
"""
import asyncio

from orchestrator.json_utils import call_with_json_retry
from orchestrator.models import Candidate
from orchestrator.retrieval import format_papers_for_prompt, search_literature

_SYSTEM_PROMPT = (
    "You are a Generation agent in a research-ideation swarm. Produce ONE "
    "concrete, conceptually distinct research hypothesis for the given topic. "
    'Respond with ONLY a JSON object: {"idea": "<the hypothesis, 2-4 sentences>"}.'
)

_DIRECTION_HINTS = [
    "a mechanistic/causal explanation",
    "a measurement or evaluation gap",
    "a cross-domain analogy",
    "a contradiction in existing findings",
    "a scaling or efficiency angle",
    "an underexplored edge case",
]


def _build_memory_context(memory, domain: str, subfield: str) -> str:
    entries = memory.query(domain, subfield)
    if not entries:
        return "No prior ideas on record for this subfield."
    ranked = sorted(entries, key=lambda e: e.elo, reverse=True)[:5]
    lines = ["PAST IDEAS TRIED FOR THIS SUBFIELD (avoid repeating these; build on what scored well):"]
    for entry in ranked:
        lines.append(f"- (elo {entry.elo:.0f}) {entry.text}")
    return "\n".join(lines)


def _generate_one(backend, topic, direction_hint, literature_block, memory_block, meta_review_guidance, round_num):
    user_prompt = (
        f"Topic: {topic}\n"
        f"Take this specific angle/direction: {direction_hint}\n\n"
        f"{literature_block}\n\n{memory_block}\n\n"
        f"Guidance from last round's critiques: {meta_review_guidance or 'N/A (first round)'}"
    )
    parsed = call_with_json_retry(backend, "generation", _SYSTEM_PROMPT, user_prompt)
    return Candidate(text=parsed["idea"], round_num=round_num, source_strategy="divergent")


async def generate_round(config, backend, memory, proximity, round_num: int, meta_review_guidance: str = "") -> list:
    literature = search_literature(config.topic)
    literature_block = format_papers_for_prompt(literature)
    memory_block = _build_memory_context(memory, config.domain, config.subfield)

    hints = [_DIRECTION_HINTS[i % len(_DIRECTION_HINTS)] for i in range(config.n_divergent_per_round)]
    raw_candidates = list(await asyncio.gather(*[
        asyncio.to_thread(
            _generate_one, backend, config.topic, hint, literature_block, memory_block,
            meta_review_guidance, round_num,
        )
        for hint in hints
    ]))

    proximity.compute_embeddings(raw_candidates)

    memory_entries = memory.query(config.domain, config.subfield)
    memory_as_candidates = [
        Candidate(text=e.text, round_num=0, source_strategy="memory", embedding=e.embedding)
        for e in memory_entries if e.embedding is not None
    ]

    return [
        c for c in raw_candidates
        if not proximity.is_near_duplicate(c, memory_as_candidates, config.duplicate_similarity_threshold)
    ]
