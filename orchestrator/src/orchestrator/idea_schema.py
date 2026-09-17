"""Shared structured-idea schema for Generation and Evolution (spec §5.2).
A single free-text "idea" blob let agents pad with restated jargon instead
of committing to a testable claim; forcing each field to carry its own
distinct content (a mechanism that isn't just the hypothesis reworded, a
falsification condition that isn't the prediction negated) is what makes
ideas comparable in debate and legible outside the swarm.
"""

FIELDS = [
    ("title", "Title", "A short, specific name for the idea (<= 12 words)."),
    ("problem", "Problem/gap", "The specific unsolved problem or gap this targets, and why it matters."),
    ("hypothesis", "Hypothesis", "The actual claim, stated plainly in 1-2 sentences. No hedging, no jargon that isn't doing work."),
    ("mechanism", "Mechanism", "The causal story for why this would work. Must add information beyond the hypothesis, not restate it."),
    ("method", "Method", "A concrete way to test it: what you would build or measure, on what data or setup."),
    ("prediction", "Prediction", "The specific, measurable result that would confirm the hypothesis."),
    ("falsification", "Falsification", "The specific result that would prove the hypothesis wrong. Must not just be the negation of the prediction."),
    ("novelty_delta", "Related-work delta", "The closest existing idea or paper, and exactly how this differs from it."),
]

FIELD_KEYS = [key for key, _, _ in FIELDS]

RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object with exactly these keys: "
    + ", ".join(f'"{key}"' for key in FIELD_KEYS)
    + ". Field-by-field requirements:\n"
    + "\n".join(f"- {label} ({key}): {guide}" for key, label, guide in FIELDS)
)


def render_idea(fields: dict) -> str:
    lines = []
    for key, label, _ in FIELDS:
        value = (fields.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)
