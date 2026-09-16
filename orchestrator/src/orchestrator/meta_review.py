"""Meta-review stage (spec §5.2): end-of-round synthesis of recurring
critique patterns, fed into next round's Generation as extra context. This
is the round-over-round learning mechanism, distinct from cross-run
memory.
"""

_SYSTEM_PROMPT = (
    "You are a Meta-review agent. Summarize recurring patterns across this "
    "round's critiques and debate outcomes in 3-5 sentences of concrete, "
    "actionable guidance for the next round's idea generation. Plain text, no JSON."
)


def synthesize_meta_review(backend, candidates: list, critiques: list, debates: list) -> str:
    failed_issues = [issue for c in critiques if not c.passed for issue in c.issues]
    debate_reasons = [d.reasoning for d in debates]

    user_prompt = (
        f"This round produced {len(candidates)} candidates.\n"
        f"Reflection issues raised: {failed_issues or 'none'}\n"
        f"Debate reasoning excerpts: {debate_reasons}"
    )
    return backend.call("meta_review", _SYSTEM_PROMPT, user_prompt)
