"""Own multi-layer JSON-parse-with-retry for structured agent calls (spec
§7). Three extraction layers (direct parse, fenced code block, bare
braces), then a bounded number of re-prompts asking the model to correct
its own output.
"""
import json
import re


class JSONRetryExhausted(Exception):
    pass


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw_text: str) -> dict | None:
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_BLOCK_RE.search(raw_text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = _BRACE_RE.search(raw_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_with_json_retry(backend, role: str, system_prompt: str, user_prompt: str, max_attempts: int = 3) -> dict:
    prompt = user_prompt
    last_raw = ""
    for _attempt in range(max_attempts):
        last_raw = backend.call(role, system_prompt, prompt)
        parsed = extract_json(last_raw)
        if parsed is not None:
            return parsed
        prompt = (
            f"{user_prompt}\n\n"
            "Your previous response could not be parsed as JSON. "
            f"Previous response:\n{last_raw}\n\n"
            "Respond with ONLY a single valid JSON object, no prose, no markdown fences."
        )
    raise JSONRetryExhausted(f"Failed to parse JSON after {max_attempts} attempts. Last raw response: {last_raw!r}")
