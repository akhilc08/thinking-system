# Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-agent ideation swarm (`orchestrator/`) described in the
design spec §5 — a pluggable-backend agent system that runs rounds of
generation, reflection, debate-ranked evolution, and meta-review over a
topic, backed by persistent cross-run memory, to produce either one
hypothesis (benchmark mode) or the top-K diverse hypotheses (real use).

**Architecture:** A standalone Python package under `src/orchestrator/`,
independent of `vendor/AgentIdeaBench` and of `src/benchmark`'s internals —
it never imports from either. The only place it ever touches `benchmark` is
the two manual smoke-test scripts in Task 17, which construct a
`benchmark.models.Hypothesis` from the orchestrator's own output and call
`benchmark.scorer.score_hypothesis` on it as a boundary integration, never
as a package dependency of orchestrator's own code. All pipeline stages
(Generation, Reflection, Ranking, Evolution, Meta-review) are written
against the `ModelBackend` protocol (spec §5.1); concurrency within a round
uses `asyncio.gather` + `asyncio.to_thread` over the synchronous
backend/HTTP calls, not a separate distributed-queue service — "more
compute" means more rounds/pairs, without building custom queue
infrastructure the v1 scope doesn't call for. Run state and cross-run
memory persist to SQLite with the orchestrator's own schema (not the
vendored repo's). The embedding-similarity service (Proximity) uses a
local `sentence-transformers` model, not a hosted embeddings API — this
keeps it self-contained (no extra API key, no unverified endpoint contract)
and matches what AgentRxiv used for the same purpose per
`docs/research/existing-systems.md`.

**Tech Stack:** Python 3.9+ (matches Plan 1's benchmark-harness plan),
`requests` for HTTP (OpenRouter chat completions, Semantic Scholar),
`sentence-transformers` + `numpy` for embeddings/cosine similarity, stdlib
`sqlite3`, stdlib `subprocess` for the Claude Code backend, stdlib
`asyncio` for round-level concurrency, `pytest` + `pytest-asyncio` for
tests.

**Spec:** [`docs/superpowers/specs/2026-09-16-multiagent-ideation-system-design.md`](../specs/2026-09-16-multiagent-ideation-system-design.md) — implements §5 (Orchestrator) of that spec. §8's testing-plan items 2–3 (OpenRouter/Claude Code backend smoke tests) are covered by Task 17's manual scripts.

**Prerequisite:** This plan assumes Plan 1
([`2026-09-16-benchmark-harness.md`](./2026-09-16-benchmark-harness.md))'s
Task 1 has already run, so `pyproject.toml`, `.gitignore`, and the
`src/`/`tests/`/`config/` layout already exist. Task 1 below only adds
orchestrator-specific dependencies and files on top of that. If Plan 1
hasn't been run yet, create `pyproject.toml` fresh using the target content
shown in Task 1, Step 1 (a superset of Plan 1's own Task 1 content).

## Global Constraints

- Files under 500 lines (project CLAUDE.md).
- TDD London School (mock-first): every automated test mocks
  `ModelBackend.call`, `subprocess.run`, `requests` calls, and the local
  embedding model — nothing in the pytest suite makes a real network call,
  spawns a real `claude` subprocess, or loads a real sentence-transformers
  model. The exceptions are Task 17's manual smoke scripts, matching the
  same pattern as Plan 1's Task 8.
- Source in `/src`, tests in `/tests`, config in `/config` (project
  CLAUDE.md).
- `orchestrator/` never imports from `vendor/AgentIdeaBench` or from
  `src/benchmark`'s internal modules. It is a standalone system.
- `ModelBackend.call`'s signature is fixed by spec §5.1:
  `call(self, role: str, system_prompt: str, user_prompt: str) -> str`. Do
  not change it to `async def` — concurrency is layered on top via
  `asyncio.to_thread`.
- Hard cap on total model calls per run (spec §7), enforced via a counting
  wrapper (`CountingBackend`), never by trusting individual stages to
  self-limit.
- Ranking's Elo outcome is the only source of a candidate's persisted
  quality score — never write a model's self-reported score to memory or
  use it for ranking (spec §5.4 / §9's reward-hacking gap). Reflection's
  `passed`/`issues` are a coherence gate, not a quality score.
- Deliberate divergence (Evolution) runs every round, unconditionally —
  never gated behind a flag or a "should we diverge" check (spec §5.2,
  §9).

## File Structure

```
src/orchestrator/
  __init__.py
  models.py              — Candidate, CritiqueResult, DebateVerdict, RunConfig, MemoryEntry, RunSummary
  json_utils.py           — multi-layer JSON-parse-with-retry for structured agent calls
  backend.py              — ModelBackend protocol, retry_with_backoff, OpenRouterBackend
  claude_code_backend.py  — ClaudeCodeBackend (subprocess)
  embeddings.py           — EmbeddingBackend protocol, SentenceTransformerBackend, cosine_similarity
  store.py                — RunStore: SQLite run/round/candidate/debate/final-selection persistence
  proximity.py            — Proximity: debate pairing, near-dup checks, diverse top-k selection
  memory.py               — MemoryStore: cross-run memory, dedup-on-write
  retrieval.py            — standalone Semantic Scholar search wrapper
  generation.py           — Generation stage: divergent(N) -> convergent(k)
  reflection.py           — Reflection stage: cheap critique gate
  ranking.py               — Ranking stage: pairwise debate + Elo
  evolution.py             — Evolution stage: 5 named mutation strategies
  meta_review.py           — Meta-review stage: round synthesis -> next round's guidance
  supervisor.py             — round loop, call-budget enforcement, final selection
  cli.py                    — python -m orchestrator.cli entry point
config/orchestrator.py      — env-var-backed config (own, not vendor's)
tests/orchestrator/          — one test file per src module above
scripts/orchestrator_smoke_openrouter.py   — manual smoke test, OpenRouter backend
scripts/orchestrator_smoke_claude_code.py  — manual smoke test, Claude Code backend
```

---

## Task 1: Orchestrator package scaffolding

**Files:**
- Modify (or create if absent): `pyproject.toml`
- Create: `src/orchestrator/__init__.py`
- Create: `config/orchestrator.py`
- Test: `tests/orchestrator/test_config.py`

**Interfaces:**
- Produces: `config.orchestrator.PROJECT_ROOT`, `DB_PATH`, `OPENROUTER_API_KEY`,
  `SEMANTIC_SCHOLAR_API_KEY`, `CLAUDE_CLI_PATH`, `DEFAULT_GENERATION_MODEL`,
  `DEFAULT_EMBEDDING_MODEL`, `OPENROUTER_BASE_URL`.

- [ ] **Step 1: Update project scaffolding**

If `pyproject.toml` already exists (Plan 1 has run), edit it to match this
exact target state — add `numpy` to `dependencies`, add `pytest-asyncio` to
the `dev` extra, and add `asyncio_mode = "auto"` under
`[tool.pytest.ini_options]`:

```toml
[project]
name = "thinking-system"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "requests",
    "pandas",
    "numpy",
    "sentence-transformers",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
```

If `pyproject.toml` does not exist yet, create it fresh with the exact
content above.

Install dev deps: `pip install -e ".[dev]"`

- [ ] **Step 2: Write the failing test**

```python
# tests/orchestrator/test_config.py
import importlib


def test_config_defaults_to_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "ss-key-456")
    from config import orchestrator
    importlib.reload(orchestrator)

    assert orchestrator.OPENROUTER_API_KEY == "test-key-123"
    assert orchestrator.SEMANTIC_SCHOLAR_API_KEY == "ss-key-456"
    assert orchestrator.DB_PATH.name == "runs.db"
    assert orchestrator.CLAUDE_CLI_PATH == "claude"
    assert orchestrator.OPENROUTER_BASE_URL.startswith("https://")
    assert isinstance(orchestrator.DEFAULT_EMBEDDING_MODEL, str)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config.orchestrator'`

- [ ] **Step 4: Write minimal implementation**

```python
# config/orchestrator.py
"""Project-level config for the orchestrator. Independent of
vendor/AgentIdeaBench and of config/benchmark.py — the orchestrator is a
standalone system (see plan header)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / ".data" / "orchestrator" / "runs.db"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")

DEFAULT_GENERATION_MODEL = os.environ.get(
    "ORCHESTRATOR_GENERATION_MODEL", "anthropic/claude-sonnet-5"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ORCHESTRATOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

```python
# src/orchestrator/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config/orchestrator.py src/orchestrator/__init__.py tests/orchestrator/test_config.py
git commit -m "Add orchestrator package scaffolding and config"
```

---

## Task 2: Data models

**Files:**
- Create: `src/orchestrator/models.py`
- Test: `tests/orchestrator/test_models.py`

**Interfaces:**
- Consumes: nothing (pure data classes).
- Produces: `new_id() -> str`, `Candidate(text, round_num, source_strategy,
  parent_ids=[], candidate_id=<auto>, elo=1200.0, embedding=None)`,
  `CritiqueResult(candidate_id, passed, issues=[])`,
  `DebateVerdict(candidate_a_id, candidate_b_id, winner_id, reasoning)`,
  `RunConfig(topic, domain, subfield, backend_kind="openrouter", rounds=3,
  pool_size=24, n_divergent_per_round=6, debate_pairs_per_round=8,
  k_final=1, max_model_calls=200, duplicate_similarity_threshold=0.92,
  diverse_selection_similarity_threshold=0.85, run_id=<auto>)`,
  `MemoryEntry(domain, subfield, text, elo, run_id, embedding=None,
  memory_id=<auto>)`, `RunSummary(run_id, total_model_calls,
  compute_multiplier, final_candidates, status)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_models.py
from orchestrator.models import Candidate, MemoryEntry, RunConfig, new_id


def test_new_id_returns_unique_hex_strings():
    a, b = new_id(), new_id()
    assert a != b
    assert len(a) == 32


def test_candidate_defaults():
    c = Candidate(text="an idea", round_num=1, source_strategy="divergent")
    assert c.elo == 1200.0
    assert c.parent_ids == []
    assert c.embedding is None
    assert c.candidate_id

    c2 = Candidate(text="another idea", round_num=1, source_strategy="divergent")
    assert c.candidate_id != c2.candidate_id


def test_run_config_defaults():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t")
    assert config.backend_kind == "openrouter"
    assert config.rounds == 3
    assert config.k_final == 1
    assert config.run_id


def test_memory_entry_holds_identifying_fields():
    entry = MemoryEntry(domain="Computer Science", subfield="t", text="idea", elo=1250.0, run_id="r1")
    assert entry.domain == "Computer Science"
    assert entry.memory_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/models.py
"""Plain data holders for the orchestrator. No behavior lives here."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Candidate:
    text: str
    round_num: int
    source_strategy: str  # "divergent" | "combine" | "simplify" | "ground_in_literature" | "analogical" | "deliberate_divergence"
    parent_ids: list = field(default_factory=list)
    candidate_id: str = field(default_factory=new_id)
    elo: float = 1200.0
    embedding: list = None


@dataclass
class CritiqueResult:
    candidate_id: str
    passed: bool
    issues: list = field(default_factory=list)


@dataclass
class DebateVerdict:
    candidate_a_id: str
    candidate_b_id: str
    winner_id: str
    reasoning: str


@dataclass
class RunConfig:
    topic: str
    domain: str
    subfield: str
    backend_kind: str = "openrouter"  # "openrouter" | "claude_code"
    rounds: int = 3
    pool_size: int = 24
    n_divergent_per_round: int = 6
    debate_pairs_per_round: int = 8
    k_final: int = 1
    max_model_calls: int = 200
    duplicate_similarity_threshold: float = 0.92
    diverse_selection_similarity_threshold: float = 0.85
    run_id: str = field(default_factory=new_id)


@dataclass
class MemoryEntry:
    domain: str
    subfield: str
    text: str
    elo: float
    run_id: str
    embedding: list = None
    memory_id: str = field(default_factory=new_id)


@dataclass
class RunSummary:
    run_id: str
    total_model_calls: int
    compute_multiplier: float
    final_candidates: list
    status: str  # "completed" | "truncated_call_budget"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/models.py tests/orchestrator/test_models.py
git commit -m "Add orchestrator data models"
```

---

## Task 3: JSON-retry parsing utility

**Files:**
- Create: `src/orchestrator/json_utils.py`
- Test: `tests/orchestrator/test_json_utils.py`

**Interfaces:**
- Consumes: `ModelBackend`-shaped object (duck-typed, just needs `.call`).
- Produces: `JSONRetryExhausted(Exception)`, `extract_json(raw_text: str) ->
  dict | None`, `call_with_json_retry(backend, role, system_prompt,
  user_prompt, max_attempts=3) -> dict`.

This is the orchestrator's own multi-layer JSON-parse-with-retry
implementation — standalone, not imported from `vendor/AgentIdeaBench`
(see plan header). Every structured agent call in later tasks
(Reflection's pass/fail, Ranking's debate verdict, Generation/Evolution's
`{"idea": ...}`) goes through this.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_json_utils.py
import pytest

from orchestrator.json_utils import JSONRetryExhausted, call_with_json_retry, extract_json


def test_extract_json_parses_raw_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_parses_fenced_code_block():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_parses_bare_braces_in_prose():
    raw = 'Sure, my answer is {"a": 1} - let me know if you need more.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_returns_none_for_unparseable_text():
    assert extract_json("no json here at all") is None


class _FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, role, system_prompt, user_prompt):
        self.calls.append((role, system_prompt, user_prompt))
        return self.responses.pop(0)


def test_call_with_json_retry_succeeds_on_first_try():
    backend = _FakeBackend(['{"idea": "x"}'])
    result = call_with_json_retry(backend, "generation", "sys", "usr")
    assert result == {"idea": "x"}
    assert len(backend.calls) == 1


def test_call_with_json_retry_retries_after_malformed_response():
    backend = _FakeBackend(["not json", '{"idea": "x"}'])
    result = call_with_json_retry(backend, "generation", "sys", "usr")
    assert result == {"idea": "x"}
    assert len(backend.calls) == 2
    assert "could not be parsed" in backend.calls[1][2]


def test_call_with_json_retry_raises_after_max_attempts():
    backend = _FakeBackend(["not json", "still not json", "nope"])
    with pytest.raises(JSONRetryExhausted):
        call_with_json_retry(backend, "generation", "sys", "usr", max_attempts=3)
    assert len(backend.calls) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_json_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.json_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/json_utils.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_json_utils.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/json_utils.py tests/orchestrator/test_json_utils.py
git commit -m "Add JSON-parse-with-retry utility for structured agent calls"
```

---

## Task 4: ModelBackend protocol + OpenRouterBackend

**Files:**
- Create: `src/orchestrator/backend.py`
- Test: `tests/orchestrator/test_backend.py`

**Interfaces:**
- Consumes: `requests`; `config.orchestrator.OPENROUTER_BASE_URL`.
- Produces: `ModelBackend` (Protocol: `call(role, system_prompt,
  user_prompt) -> str`), `BackendError(Exception)`,
  `retry_with_backoff(fn, max_attempts=3, base_delay=1.0)`,
  `OpenRouterBackend(model, api_key, base_url=OPENROUTER_BASE_URL,
  timeout=60.0)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_backend.py
from unittest.mock import Mock, patch

import pytest

from orchestrator.backend import BackendError, OpenRouterBackend, retry_with_backoff


def test_retry_with_backoff_succeeds_on_first_try():
    fn = Mock(return_value="ok")
    assert retry_with_backoff(fn) == "ok"
    assert fn.call_count == 1


def test_retry_with_backoff_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    fn = Mock(side_effect=[BackendError("boom"), "ok"])
    assert retry_with_backoff(fn, max_attempts=3) == "ok"
    assert fn.call_count == 2


def test_retry_with_backoff_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    fn = Mock(side_effect=BackendError("boom"))
    with pytest.raises(BackendError):
        retry_with_backoff(fn, max_attempts=3)
    assert fn.call_count == 3


def test_openrouter_backend_call_returns_message_content():
    backend = OpenRouterBackend(model="anthropic/claude-sonnet-5", api_key="key123")
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    with patch("orchestrator.backend.requests.post", return_value=fake_response) as mock_post:
        result = backend.call("generation", "sys prompt", "user prompt")

    assert result == "hello"
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer key123"
    assert kwargs["json"]["model"] == "anthropic/claude-sonnet-5"


def test_openrouter_backend_raises_on_non_200(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    backend = OpenRouterBackend(model="m", api_key="key123")
    fake_response = Mock(status_code=500, text="server error")
    with patch("orchestrator.backend.requests.post", return_value=fake_response):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_openrouter_backend_raises_on_malformed_response(monkeypatch):
    monkeypatch.setattr("orchestrator.backend.time.sleep", lambda _s: None)
    backend = OpenRouterBackend(model="m", api_key="key123")
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {"unexpected": "shape"}
    with patch("orchestrator.backend.requests.post", return_value=fake_response):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.backend'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/backend.py
"""ModelBackend protocol and the OpenRouter implementation (spec §5.1).
All orchestrator pipeline stages are written against ModelBackend, never
against a specific provider — see also claude_code_backend.py for the
second implementation.
"""
import time
from typing import Protocol

import requests

from config.orchestrator import OPENROUTER_BASE_URL


class ModelBackend(Protocol):
    def call(self, role: str, system_prompt: str, user_prompt: str) -> str: ...


class BackendError(Exception):
    pass


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except BackendError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


class OpenRouterBackend:
    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_BASE_URL, timeout: float = 60.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        def _do_call() -> str:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise BackendError(f"OpenRouter call failed for role={role}: {response.status_code} {response.text}")
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise BackendError(f"Unexpected OpenRouter response shape: {data}") from exc

        return retry_with_backoff(_do_call)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_backend.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/backend.py tests/orchestrator/test_backend.py
git commit -m "Add ModelBackend protocol and OpenRouterBackend"
```

---

## Task 5: ClaudeCodeBackend

**Files:**
- Create: `src/orchestrator/claude_code_backend.py`
- Test: `tests/orchestrator/test_claude_code_backend.py`

**Interfaces:**
- Consumes: stdlib `subprocess`, `json`; `orchestrator.backend.BackendError,
  retry_with_backoff`; `config.orchestrator.CLAUDE_CLI_PATH`.
- Produces: `ClaudeCodeBackend(cli_path=CLAUDE_CLI_PATH, timeout=300.0)`
  implementing `.call(role, system_prompt, user_prompt) -> str`.

**Note on an unverified assumption:** this task assumes `claude -p "<prompt>"
--output-format json` produces a JSON object with a `"result"` key holding
the final text (this matches the CLI's documented headless-mode output
shape at time of writing). Step 6 below is a manual check to confirm this
against whatever CLI version is actually installed before relying on this
backend for a real run — do not skip it.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_claude_code_backend.py
import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from orchestrator.backend import BackendError
from orchestrator.claude_code_backend import ClaudeCodeBackend


def _fake_result(returncode=0, stdout="", stderr=""):
    return Mock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_call_returns_result_field_from_cli_json_output():
    backend = ClaudeCodeBackend(cli_path="claude")
    fake = _fake_result(stdout=json.dumps({"result": "a generated idea"}))
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake) as mock_run:
        result = backend.call("generation", "sys prompt", "user prompt")

    assert result == "a generated idea"
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd


def test_call_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(returncode=1, stderr="boom")
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_on_malformed_json_stdout(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(stdout="not json")
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_when_result_key_missing(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend()
    fake = _fake_result(stdout=json.dumps({"unexpected": "shape"}))
    with patch("orchestrator.claude_code_backend.subprocess.run", return_value=fake):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")


def test_call_raises_on_subprocess_timeout(monkeypatch):
    monkeypatch.setattr("orchestrator.claude_code_backend.time.sleep", lambda _s: None)
    backend = ClaudeCodeBackend(timeout=5.0)
    with patch(
        "orchestrator.claude_code_backend.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5.0),
    ):
        with pytest.raises(BackendError):
            backend.call("generation", "sys", "usr")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_claude_code_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.claude_code_backend'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/claude_code_backend.py
"""ClaudeCodeBackend: shells out to the `claude` CLI in headless mode so
swarm runs consume the user's Claude Code plan instead of metered API
tokens (spec §5.1). One subprocess per call. The system prompt is
prepended to the single -p prompt string rather than relying on a
headless-mode system-prompt flag whose exact name may vary by CLI version
— see Task 5's manual verification step.
"""
import json
import subprocess
import time

from config.orchestrator import CLAUDE_CLI_PATH
from orchestrator.backend import BackendError, retry_with_backoff


class ClaudeCodeBackend:
    def __init__(self, cli_path: str = CLAUDE_CLI_PATH, timeout: float = 300.0):
        self.cli_path = cli_path
        self.timeout = timeout

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        def _do_call() -> str:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            try:
                result = subprocess.run(
                    [self.cli_path, "-p", full_prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise BackendError(f"Claude CLI subprocess failed for role={role}: {exc}") from exc

            if result.returncode != 0:
                raise BackendError(f"Claude CLI exited {result.returncode} for role={role}: {result.stderr}")

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise BackendError(f"Claude CLI produced non-JSON stdout for role={role}: {result.stdout!r}") from exc

            text = data.get("result")
            if text is None:
                raise BackendError(f"Claude CLI JSON output missing 'result' key for role={role}: {data}")
            return text

        return retry_with_backoff(_do_call)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_claude_code_backend.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Manually verify the CLI's actual JSON output shape**

Run: `claude -p "Say hi in one word" --output-format json | python -m json.tool`

Confirm the key holding the final response text is `"result"`. If the
installed CLI version uses a different key name, update `data.get("result")`
in `claude_code_backend.py` before relying on this backend for a real run
(Task 17).

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/claude_code_backend.py tests/orchestrator/test_claude_code_backend.py
git commit -m "Add ClaudeCodeBackend (subprocess-based)"
```

---

## Task 6: Embeddings + cosine similarity

**Files:**
- Create: `src/orchestrator/embeddings.py`
- Test: `tests/orchestrator/test_embeddings.py`

**Interfaces:**
- Consumes: `sentence_transformers.SentenceTransformer`, `numpy`;
  `config.orchestrator.DEFAULT_EMBEDDING_MODEL`.
- Produces: `EmbeddingBackend` (Protocol: `embed(texts: list[str]) ->
  list[list[float]]`), `SentenceTransformerBackend(model_name=
  DEFAULT_EMBEDDING_MODEL)`, `cosine_similarity(a, b) -> float`.

Local embeddings (not a hosted API) — self-contained, no extra API key,
deterministic to mock in tests. The model is loaded lazily on first
`.embed()` call, not in `__init__`, so constructing a
`SentenceTransformerBackend` in a test never triggers a real model
download.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_embeddings.py
from unittest.mock import Mock, patch

from orchestrator.embeddings import SentenceTransformerBackend, cosine_similarity


def test_cosine_similarity_of_identical_vectors_is_one():
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embed_lazily_loads_model_and_returns_vectors():
    backend = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
    fake_model = Mock()
    fake_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with patch("orchestrator.embeddings.SentenceTransformer", return_value=fake_model) as mock_ctor:
        result = backend.embed(["idea one", "idea two"])
        backend.embed(["idea three"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_ctor.assert_called_once_with("all-MiniLM-L6-v2")  # loaded once, reused on 2nd call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.embeddings'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/embeddings.py
"""Embedding backend for Proximity's similarity computations (spec §5.2).
Local sentence-transformers model — self-contained, no extra API key,
matches what AgentRxiv used for the same purpose
(docs/research/existing-systems.md). Independent of ModelBackend: the
embedding backend always uses this, regardless of whether the swarm's
text-generation backend is OpenRouter or Claude Code (spec §5.1's backend
choice only covers text-generation agents).
"""
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from config.orchestrator import DEFAULT_EMBEDDING_MODEL


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().encode(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_embeddings.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/embeddings.py tests/orchestrator/test_embeddings.py
git commit -m "Add local sentence-transformers embedding backend"
```

---

## Task 7: RunStore (SQLite run persistence)

**Files:**
- Create: `src/orchestrator/store.py`
- Test: `tests/orchestrator/test_store.py`

**Interfaces:**
- Consumes: stdlib `sqlite3`, `json`; `orchestrator.models.Candidate,
  CritiqueResult, DebateVerdict, RunConfig`.
- Produces: `RunStore(db_path)` with `init_schema()`, `create_run(config)`,
  `get_run_status(run_id) -> str | None`, `mark_run_completed(run_id,
  total_model_calls, compute_multiplier)`, `mark_run_truncated(run_id,
  total_model_calls, compute_multiplier)`, `record_round_start(run_id,
  round_num)`, `record_round_complete(run_id, round_num,
  meta_review_guidance)`, `last_completed_round(run_id) -> int`,
  `save_candidate(run_id, candidate)`, `save_critique(run_id, critique)`,
  `save_debate(run_id, round_num, verdict, elo_before, elo_after)`,
  `save_final_selection(run_id, ranked_candidates)`, `get_pool(run_id) ->
  list[Candidate]`, `close()`.

Own schema — not the vendored repo's — designed for the same
idempotent/resumable-run property: a run's status and its last-completed
round are checked before re-running a round, so an interrupted run can
pick up where it left off (spec §6).

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_store.py
from orchestrator.models import Candidate, DebateVerdict, RunConfig
from orchestrator.store import RunStore


def _config(**overrides):
    defaults = dict(topic="t", domain="Computer Science", subfield="t", run_id="run-1")
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_create_run_and_get_status(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    assert store.get_run_status(config.run_id) == "in_progress"


def test_create_run_is_idempotent(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    store.create_run(config)  # should not raise or duplicate
    assert store.get_run_status(config.run_id) == "in_progress"


def test_round_lifecycle_and_resumability(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    assert store.last_completed_round(config.run_id) == 0
    store.record_round_start(config.run_id, 1)
    store.record_round_complete(config.run_id, 1, "guidance for round 2")
    assert store.last_completed_round(config.run_id) == 1

    store.record_round_start(config.run_id, 2)  # round 2 never completes
    assert store.last_completed_round(config.run_id) == 1  # resume point stays at 1


def test_save_candidate_and_get_pool(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)

    c1 = Candidate(text="idea one", round_num=1, source_strategy="divergent", elo=1210.0)
    c2 = Candidate(text="idea two", round_num=1, source_strategy="divergent", parent_ids=[c1.candidate_id])
    store.save_candidate(config.run_id, c1)
    store.save_candidate(config.run_id, c2)

    pool = store.get_pool(config.run_id)
    assert len(pool) == 2
    by_id = {c.candidate_id: c for c in pool}
    assert by_id[c1.candidate_id].elo == 1210.0
    assert by_id[c2.candidate_id].parent_ids == [c1.candidate_id]


def test_save_debate_records_elo_history(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    verdict = DebateVerdict(candidate_a_id="a", candidate_b_id="b", winner_id="a", reasoning="A is stronger")
    store.save_debate(config.run_id, 1, verdict, (1200.0, 1200.0), (1216.0, 1184.0))
    row = store._conn.execute("SELECT winner_id, elo_a_after, elo_b_after FROM debates").fetchone()
    assert row == ("a", 1216.0, 1184.0)


def test_save_final_selection_and_mark_completed(tmp_path):
    store = RunStore(tmp_path / "test.db")
    config = _config()
    store.create_run(config)
    c1 = Candidate(text="idea one", round_num=1, source_strategy="divergent", elo=1300.0)
    store.save_final_selection(config.run_id, [c1])
    store.mark_run_completed(config.run_id, total_model_calls=42, compute_multiplier=4.2)

    row = store._conn.execute(
        "SELECT candidate_id FROM final_selection WHERE run_id = ? AND rank = 1", (config.run_id,)
    ).fetchone()
    assert row[0] == c1.candidate_id
    assert store.get_run_status(config.run_id) == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/store.py
"""SQLite persistence for orchestrator runs (spec §6). Own schema — not
the vendored repo's — designed for the same idempotent/resumable-run
property: a run's status and its last-completed round are checked before
re-running a round, so an interrupted run can pick up where it left off.
"""
import json
import sqlite3
from pathlib import Path

from orchestrator.models import Candidate, CritiqueResult, DebateVerdict, RunConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topic TEXT, domain TEXT, subfield TEXT,
    config_json TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress',
    total_model_calls INTEGER DEFAULT 0,
    compute_multiplier REAL
);
CREATE TABLE IF NOT EXISTS rounds (
    run_id TEXT, round_num INTEGER,
    status TEXT NOT NULL DEFAULT 'in_progress',
    meta_review_guidance TEXT,
    PRIMARY KEY (run_id, round_num)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT, round_num INTEGER, text TEXT,
    source_strategy TEXT, parent_ids TEXT, elo REAL
);
CREATE TABLE IF NOT EXISTS critiques (
    candidate_id TEXT, run_id TEXT, passed INTEGER, issues TEXT
);
CREATE TABLE IF NOT EXISTS debates (
    run_id TEXT, round_num INTEGER,
    candidate_a_id TEXT, candidate_b_id TEXT, winner_id TEXT, reasoning TEXT,
    elo_a_before REAL, elo_b_before REAL, elo_a_after REAL, elo_b_after REAL
);
CREATE TABLE IF NOT EXISTS final_selection (
    run_id TEXT, rank INTEGER, candidate_id TEXT
);
"""


class RunStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_run(self, config: RunConfig) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, topic, domain, subfield, config_json) VALUES (?, ?, ?, ?, ?)",
            (config.run_id, config.topic, config.domain, config.subfield, json.dumps(config.__dict__)),
        )
        self._conn.commit()

    def get_run_status(self, run_id: str) -> str | None:
        row = self._conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    def mark_run_completed(self, run_id: str, total_model_calls: int, compute_multiplier: float) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'completed', total_model_calls = ?, compute_multiplier = ? WHERE run_id = ?",
            (total_model_calls, compute_multiplier, run_id),
        )
        self._conn.commit()

    def mark_run_truncated(self, run_id: str, total_model_calls: int, compute_multiplier: float) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'truncated_call_budget', total_model_calls = ?, compute_multiplier = ? WHERE run_id = ?",
            (total_model_calls, compute_multiplier, run_id),
        )
        self._conn.commit()

    def record_round_start(self, run_id: str, round_num: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO rounds (run_id, round_num, status) VALUES (?, ?, 'in_progress')",
            (run_id, round_num),
        )
        self._conn.commit()

    def record_round_complete(self, run_id: str, round_num: int, meta_review_guidance: str) -> None:
        self._conn.execute(
            "UPDATE rounds SET status = 'completed', meta_review_guidance = ? WHERE run_id = ? AND round_num = ?",
            (meta_review_guidance, run_id, round_num),
        )
        self._conn.commit()

    def last_completed_round(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(round_num) FROM rounds WHERE run_id = ? AND status = 'completed'", (run_id,)
        ).fetchone()
        return row[0] if row and row[0] is not None else 0

    def save_candidate(self, run_id: str, candidate: Candidate) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO candidates (candidate_id, run_id, round_num, text, source_strategy, parent_ids, elo) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id, run_id, candidate.round_num, candidate.text,
                candidate.source_strategy, json.dumps(candidate.parent_ids), candidate.elo,
            ),
        )
        self._conn.commit()

    def save_critique(self, run_id: str, critique: CritiqueResult) -> None:
        self._conn.execute(
            "INSERT INTO critiques (candidate_id, run_id, passed, issues) VALUES (?, ?, ?, ?)",
            (critique.candidate_id, run_id, int(critique.passed), json.dumps(critique.issues)),
        )
        self._conn.commit()

    def save_debate(
        self, run_id: str, round_num: int, verdict: DebateVerdict,
        elo_before: tuple, elo_after: tuple,
    ) -> None:
        self._conn.execute(
            "INSERT INTO debates (run_id, round_num, candidate_a_id, candidate_b_id, winner_id, reasoning, "
            "elo_a_before, elo_b_before, elo_a_after, elo_b_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, round_num, verdict.candidate_a_id, verdict.candidate_b_id, verdict.winner_id,
                verdict.reasoning, elo_before[0], elo_before[1], elo_after[0], elo_after[1],
            ),
        )
        self._conn.commit()

    def save_final_selection(self, run_id: str, ranked_candidates: list) -> None:
        self._conn.execute("DELETE FROM final_selection WHERE run_id = ?", (run_id,))
        for rank, candidate in enumerate(ranked_candidates, start=1):
            self._conn.execute(
                "INSERT INTO final_selection (run_id, rank, candidate_id) VALUES (?, ?, ?)",
                (run_id, rank, candidate.candidate_id),
            )
        self._conn.commit()

    def get_pool(self, run_id: str) -> list:
        rows = self._conn.execute(
            "SELECT candidate_id, round_num, text, source_strategy, parent_ids, elo FROM candidates WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            Candidate(
                candidate_id=row[0], round_num=row[1], text=row[2], source_strategy=row[3],
                parent_ids=json.loads(row[4]), elo=row[5],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/store.py tests/orchestrator/test_store.py
git commit -m "Add RunStore: SQLite persistence for orchestrator runs"
```

---

## Task 8: Proximity service

**Files:**
- Create: `src/orchestrator/proximity.py`
- Test: `tests/orchestrator/test_proximity.py`

**Interfaces:**
- Consumes: `orchestrator.embeddings.EmbeddingBackend, cosine_similarity`;
  `orchestrator.models.Candidate`.
- Produces: `Proximity(embedding_backend)` with `compute_embeddings(candidates)`
  (mutates `.embedding` in place, batches into one `embed()` call, skips
  candidates that already have an embedding), `is_near_duplicate(candidate,
  others, threshold) -> bool`, `pick_debate_pairs(pool, n_pairs,
  duplicate_threshold=0.92) -> list[tuple[Candidate, Candidate]]`,
  `select_diverse_topk(ranked_pool, k, threshold) -> list[Candidate]`.

`pick_debate_pairs` walks the Elo-sorted pool and, for each unpaired
candidate, pairs it with the nearest-Elo unpaired candidate that is *not* a
near-duplicate — this satisfies both of Proximity's debate-related jobs
from spec §5.2 at once: comparable-strength pairing (informative, not an
obvious blowout) and skipping near-duplicate pairs (not wasted on a
near-arbitrary winner between two near-identical ideas).

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_proximity.py
from unittest.mock import Mock

from orchestrator.models import Candidate
from orchestrator.proximity import Proximity


def _candidate(text, elo=1200.0, embedding=None):
    return Candidate(text=text, round_num=1, source_strategy="divergent", elo=elo, embedding=embedding)


def test_compute_embeddings_only_calls_embed_for_missing_ones():
    already_has = _candidate("has one", embedding=[0.0, 1.0])
    needs_one = _candidate("needs one")
    embedding_backend = Mock()
    embedding_backend.embed.return_value = [[1.0, 0.0]]

    proximity = Proximity(embedding_backend)
    proximity.compute_embeddings([already_has, needs_one])

    embedding_backend.embed.assert_called_once_with(["needs one"])
    assert needs_one.embedding == [1.0, 0.0]
    assert already_has.embedding == [0.0, 1.0]


def test_is_near_duplicate_true_above_threshold():
    a = _candidate("idea A", embedding=[1.0, 0.0])
    b = _candidate("idea B", embedding=[1.0, 0.0001])
    proximity = Proximity(Mock())
    assert proximity.is_near_duplicate(a, [b], threshold=0.99) is True


def test_is_near_duplicate_false_below_threshold():
    a = _candidate("idea A", embedding=[1.0, 0.0])
    b = _candidate("idea B", embedding=[0.0, 1.0])
    proximity = Proximity(Mock())
    assert proximity.is_near_duplicate(a, [b], threshold=0.5) is False


def test_pick_debate_pairs_pairs_adjacent_elo_and_skips_duplicates():
    a = _candidate("A", elo=1300.0, embedding=[1.0, 0.0])
    b = _candidate("B", elo=1290.0, embedding=[1.0, 0.0001])  # near-duplicate of A
    c = _candidate("C", elo=1280.0, embedding=[0.0, 1.0])
    proximity = Proximity(Mock())

    pairs = proximity.pick_debate_pairs([a, b, c], n_pairs=1, duplicate_threshold=0.99)

    assert len(pairs) == 1
    ids_in_pair = {pairs[0][0].candidate_id, pairs[0][1].candidate_id}
    assert b.candidate_id not in ids_in_pair or a.candidate_id not in ids_in_pair  # A and B never paired together


def test_select_diverse_topk_skips_near_duplicates_and_stops_at_k():
    a = _candidate("A", elo=1300.0, embedding=[1.0, 0.0])
    b = _candidate("B", elo=1290.0, embedding=[1.0, 0.0001])  # near-dup of A
    c = _candidate("C", elo=1280.0, embedding=[0.0, 1.0])
    proximity = Proximity(Mock())

    selected = proximity.select_diverse_topk([a, b, c], k=2, threshold=0.99)

    assert selected == [a, c]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_proximity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.proximity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/proximity.py
"""Proximity/dedup service (spec §5.2, §5.3, §5.4): picks informative
debate pairs, prevents wasted debates between near-duplicate ideas, and
powers diversity-aware final selection and memory-store deduplication.
"""
from orchestrator.embeddings import cosine_similarity


class Proximity:
    def __init__(self, embedding_backend):
        self.embedding_backend = embedding_backend

    def compute_embeddings(self, candidates: list) -> None:
        missing = [c for c in candidates if c.embedding is None]
        if not missing:
            return
        vectors = self.embedding_backend.embed([c.text for c in missing])
        for candidate, vector in zip(missing, vectors):
            candidate.embedding = vector

    def is_near_duplicate(self, candidate, others: list, threshold: float) -> bool:
        for other in others:
            if getattr(other, "candidate_id", None) == candidate.candidate_id:
                continue
            if other.embedding is None or candidate.embedding is None:
                continue
            if cosine_similarity(candidate.embedding, other.embedding) >= threshold:
                return True
        return False

    def pick_debate_pairs(self, pool: list, n_pairs: int, duplicate_threshold: float = 0.92) -> list:
        by_elo = sorted(pool, key=lambda c: c.elo, reverse=True)
        pairs = []
        used = set()
        for i, candidate in enumerate(by_elo):
            if len(pairs) >= n_pairs:
                break
            if candidate.candidate_id in used:
                continue
            for other in by_elo[i + 1:]:
                if other.candidate_id in used:
                    continue
                if (
                    candidate.embedding is not None and other.embedding is not None
                    and cosine_similarity(candidate.embedding, other.embedding) >= duplicate_threshold
                ):
                    continue
                pairs.append((candidate, other))
                used.add(candidate.candidate_id)
                used.add(other.candidate_id)
                break
        return pairs

    def select_diverse_topk(self, ranked_pool: list, k: int, threshold: float) -> list:
        by_elo = sorted(ranked_pool, key=lambda c: c.elo, reverse=True)
        selected = []
        for candidate in by_elo:
            if len(selected) >= k:
                break
            if candidate.embedding is not None and self.is_near_duplicate(candidate, selected, threshold):
                continue
            selected.append(candidate)
        return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_proximity.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/proximity.py tests/orchestrator/test_proximity.py
git commit -m "Add Proximity: debate pairing, dedup, diverse top-k selection"
```

---

## Task 9: MemoryStore

**Files:**
- Create: `src/orchestrator/memory.py`
- Test: `tests/orchestrator/test_memory.py`

**Interfaces:**
- Consumes: stdlib `sqlite3`, `json`; `orchestrator.models.Candidate,
  MemoryEntry`; a `Proximity`-shaped object (duck-typed, just needs
  `.is_near_duplicate`).
- Produces: `MemoryStore(db_path)` with `query(domain, subfield) ->
  list[MemoryEntry]`, `write(entry, proximity, threshold=0.92) -> bool`
  (returns `False` and skips the write if `entry` is a near-duplicate of an
  existing entry for that `(domain, subfield)`; `True` if written),
  `close()`.

A separate table from `RunStore`'s (spec §6: "separate table/collection"),
its own connection to the same SQLite file — cross-run memory is a
distinct concern from a single run's artifacts.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_memory.py
from unittest.mock import Mock

from orchestrator.memory import MemoryStore
from orchestrator.models import MemoryEntry


def test_query_returns_entries_for_domain_and_subfield(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    entry = MemoryEntry(domain="Computer Science", subfield="gnn expressiveness", text="idea", elo=1250.0, run_id="r1")
    store.write(entry, proximity=Mock(is_near_duplicate=Mock(return_value=False)))

    results = store.query("Computer Science", "gnn expressiveness")
    assert len(results) == 1
    assert results[0].text == "idea"

    assert store.query("Computer Science", "other subfield") == []


def test_write_skips_near_duplicate(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    existing = MemoryEntry(
        domain="Computer Science", subfield="t", text="existing idea", elo=1200.0, run_id="r1", embedding=[1.0, 0.0]
    )
    store.write(existing, proximity=Mock(is_near_duplicate=Mock(return_value=False)))

    proximity = Mock(is_near_duplicate=Mock(return_value=True))
    new_entry = MemoryEntry(
        domain="Computer Science", subfield="t", text="near-duplicate idea", elo=1210.0, run_id="r2", embedding=[1.0, 0.0001]
    )
    written = store.write(new_entry, proximity)

    assert written is False
    assert len(store.query("Computer Science", "t")) == 1


def test_write_stores_a_non_duplicate_entry(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    entry = MemoryEntry(domain="Computer Science", subfield="t", text="idea", elo=1200.0, run_id="r1", embedding=[1.0, 0.0])
    written = store.write(entry, proximity=Mock(is_near_duplicate=Mock(return_value=False)))
    assert written is True
    assert len(store.query("Computer Science", "t")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/memory.py
"""Cross-run persistent memory (spec §5.4, §6): keyed by (domain,
subfield), deduplicated on write using the same Proximity mechanism as the
live pool so the store doesn't degrade into near-duplicate clutter over
time (a gap EvoScientist's memory design has, per docs/research).
"""
import json
import sqlite3
from pathlib import Path

from orchestrator.models import Candidate, MemoryEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    memory_id TEXT PRIMARY KEY,
    domain TEXT, subfield TEXT, text TEXT, elo REAL, run_id TEXT,
    embedding TEXT
);
"""


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def query(self, domain: str, subfield: str) -> list:
        rows = self._conn.execute(
            "SELECT memory_id, domain, subfield, text, elo, run_id, embedding FROM memory "
            "WHERE domain = ? AND subfield = ?",
            (domain, subfield),
        ).fetchall()
        return [
            MemoryEntry(
                memory_id=row[0], domain=row[1], subfield=row[2], text=row[3], elo=row[4],
                run_id=row[5], embedding=json.loads(row[6]) if row[6] else None,
            )
            for row in rows
        ]

    def write(self, entry: MemoryEntry, proximity, threshold: float = 0.92) -> bool:
        existing = self.query(entry.domain, entry.subfield)
        if entry.embedding is not None and existing:
            as_candidates = [
                Candidate(text=e.text, round_num=0, source_strategy="memory", embedding=e.embedding)
                for e in existing
            ]
            probe = Candidate(text=entry.text, round_num=0, source_strategy="memory", embedding=entry.embedding)
            if proximity.is_near_duplicate(probe, as_candidates, threshold):
                return False

        self._conn.execute(
            "INSERT OR REPLACE INTO memory (memory_id, domain, subfield, text, elo, run_id, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.memory_id, entry.domain, entry.subfield, entry.text, entry.elo, entry.run_id,
                json.dumps(entry.embedding) if entry.embedding is not None else None,
            ),
        )
        self._conn.commit()
        return True

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_memory.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/memory.py tests/orchestrator/test_memory.py
git commit -m "Add MemoryStore: cross-run memory with dedup-on-write"
```

---

## Task 10: Retrieval (standalone Semantic Scholar wrapper)

**Files:**
- Create: `src/orchestrator/retrieval.py`
- Test: `tests/orchestrator/test_retrieval.py`

**Interfaces:**
- Consumes: `requests`; `config.orchestrator.SEMANTIC_SCHOLAR_API_KEY`.
- Produces: `PaperHit(title, abstract, year, citation_count, url)`,
  `search_literature(query, limit=5, api_key=SEMANTIC_SCHOLAR_API_KEY) ->
  list[PaperHit]`, `format_papers_for_prompt(hits) -> str`.

The orchestrator's own literature-grounding mechanism for Generation and
Evolution's `ground_in_literature` strategy — standalone, independent of
`vendor/AgentIdeaBench`'s retrieval code, which belongs to the benchmark
harness only (see plan header).

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_retrieval.py
from unittest.mock import Mock, patch

from orchestrator.retrieval import format_papers_for_prompt, search_literature


def test_search_literature_parses_hits():
    fake_response = Mock(status_code=200)
    fake_response.json.return_value = {
        "data": [
            {
                "title": "Contrastive Pretraining for Molecules", "abstract": "We introduce...",
                "year": 2023, "citationCount": 42, "url": "https://example.com/paper",
            }
        ]
    }
    with patch("orchestrator.retrieval.requests.get", return_value=fake_response) as mock_get:
        hits = search_literature("contrastive molecular pretraining", limit=5, api_key="key")

    assert len(hits) == 1
    assert hits[0].title == "Contrastive Pretraining for Molecules"
    assert hits[0].citation_count == 42
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["query"] == "contrastive molecular pretraining"


def test_search_literature_returns_empty_list_on_error():
    fake_response = Mock(status_code=500)
    with patch("orchestrator.retrieval.requests.get", return_value=fake_response):
        hits = search_literature("some query")
    assert hits == []


def test_format_papers_for_prompt_reports_no_papers_found():
    assert "no relevant papers" in format_papers_for_prompt([]).lower()


def test_format_papers_for_prompt_includes_titles():
    from orchestrator.retrieval import PaperHit
    hit = PaperHit(title="A Paper", abstract="short abstract text here", year=2024, citation_count=10, url="u")
    block = format_papers_for_prompt([hit])
    assert "A Paper" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.retrieval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/retrieval.py
"""Standalone Semantic Scholar search wrapper for the orchestrator's
retrieval-grounded Generation and Evolution stages (spec §5.2) —
independent of vendor/AgentIdeaBench's retrieval code, which belongs to
the benchmark harness only.
"""
from dataclasses import dataclass

import requests

from config.orchestrator import SEMANTIC_SCHOLAR_API_KEY

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,citationCount,url"


@dataclass
class PaperHit:
    title: str
    abstract: str
    year: int
    citation_count: int
    url: str


def search_literature(query: str, limit: int = 5, api_key: str = SEMANTIC_SCHOLAR_API_KEY) -> list:
    headers = {"x-api-key": api_key} if api_key else {}
    response = requests.get(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        params={"query": query, "limit": limit, "fields": FIELDS},
        headers=headers,
        timeout=15.0,
    )
    if response.status_code != 200:
        return []
    data = response.json()
    hits = []
    for paper in data.get("data", []):
        hits.append(PaperHit(
            title=paper.get("title") or "",
            abstract=paper.get("abstract") or "",
            year=paper.get("year"),
            citation_count=paper.get("citationCount") or 0,
            url=paper.get("url") or "",
        ))
    return hits


def format_papers_for_prompt(hits: list) -> str:
    if not hits:
        return "No relevant papers found."
    lines = ["PRIOR LITERATURE:"]
    for hit in hits:
        lines.append(f"- {hit.title} ({hit.year}, {hit.citation_count} citations): {hit.abstract[:300]}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_retrieval.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/retrieval.py tests/orchestrator/test_retrieval.py
git commit -m "Add standalone Semantic Scholar retrieval wrapper"
```

---

## Task 11: Generation stage

**Files:**
- Create: `src/orchestrator/generation.py`
- Test: `tests/orchestrator/test_generation.py`

**Interfaces:**
- Consumes: `orchestrator.backend.ModelBackend`;
  `orchestrator.json_utils.call_with_json_retry`;
  `orchestrator.models.Candidate, RunConfig`; `orchestrator.memory.MemoryStore`;
  `orchestrator.proximity.Proximity`; `orchestrator.retrieval.search_literature,
  format_papers_for_prompt`.
- Produces: `async def generate_round(config, backend, memory, proximity,
  round_num, meta_review_guidance="") -> list[Candidate]`.

Divergent(N)→convergent(k) (spec §5.2): N independent, concurrently-run
generation calls, each given a distinct directional hint so they explore
different angles rather than listing variations on one theme; convergent
narrowing happens by dropping any candidate that's a near-duplicate of
something already in cross-run memory for this (domain, subfield) — "checked
against the memory store first" from spec §5.2.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_generation.py
import json
from unittest.mock import Mock, patch

import pytest

from orchestrator.generation import generate_round
from orchestrator.models import RunConfig


class _FakeBackend:
    def __init__(self, ideas):
        self.ideas = list(ideas)
        self.call_count = 0

    def call(self, role, system_prompt, user_prompt):
        idea = self.ideas[self.call_count % len(self.ideas)]
        self.call_count += 1
        return json.dumps({"idea": idea})


@pytest.mark.asyncio
async def test_generate_round_produces_n_divergent_candidates():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t", n_divergent_per_round=3)
    backend = _FakeBackend(["idea A", "idea B", "idea C"])
    memory = Mock()
    memory.query.return_value = []
    proximity = Mock()
    proximity.compute_embeddings = Mock()
    proximity.is_near_duplicate = Mock(return_value=False)

    with patch("orchestrator.generation.search_literature", return_value=[]):
        result = await generate_round(config, backend, memory, proximity, round_num=1)

    assert len(result) == 3
    assert {c.text for c in result} == {"idea A", "idea B", "idea C"}
    assert all(c.source_strategy == "divergent" for c in result)
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_generate_round_drops_memory_near_duplicates():
    config = RunConfig(topic="t", domain="Computer Science", subfield="t", n_divergent_per_round=2)
    backend = _FakeBackend(["idea A", "idea B"])
    memory = Mock()
    memory.query.return_value = []
    proximity = Mock()
    proximity.compute_embeddings = Mock()
    proximity.is_near_duplicate = Mock(side_effect=[True, False])  # first candidate is a dup, second isn't

    with patch("orchestrator.generation.search_literature", return_value=[]):
        result = await generate_round(config, backend, memory, proximity, round_num=1)

    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.generation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/generation.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_generation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/generation.py tests/orchestrator/test_generation.py
git commit -m "Add Generation stage: divergent(N) -> convergent(k)"
```

---

## Task 12: Reflection stage

**Files:**
- Create: `src/orchestrator/reflection.py`
- Test: `tests/orchestrator/test_reflection.py`

**Interfaces:**
- Consumes: `orchestrator.json_utils.call_with_json_retry`;
  `orchestrator.models.Candidate, CritiqueResult`.
- Produces: `async def reflect_round(candidates, backend) ->
  list[CritiqueResult]`, `def filter_passed(candidates, critiques) ->
  list[Candidate]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_reflection.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_reflection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.reflection'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/reflection.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_reflection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/reflection.py tests/orchestrator/test_reflection.py
git commit -m "Add Reflection stage: cheap critique gate"
```

---

## Task 13: Ranking stage (debate + Elo)

**Files:**
- Create: `src/orchestrator/ranking.py`
- Test: `tests/orchestrator/test_ranking.py`

**Interfaces:**
- Consumes: `orchestrator.json_utils.call_with_json_retry`;
  `orchestrator.models.Candidate, DebateVerdict`; a `Proximity`-shaped
  object (duck-typed, needs `.pick_debate_pairs`).
- Produces: `elo_update(rating_a, rating_b, winner, k_factor=32.0) ->
  tuple[float, float]`, `async def rank_round(pool, backend, proximity,
  n_pairs) -> list[DebateVerdict]` (mutates candidates' `.elo` in place).

Pairwise debate + Elo, not absolute scoring (spec §5.2, §9) — the
single highest-leverage design choice per the research: Agent Laboratory
measured a 2.3-point LLM-self-judge inflation vs. human raters when using
absolute scoring instead (`docs/research/design-recommendations.md` §3).

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_ranking.py
import json
from unittest.mock import Mock

import pytest

from orchestrator.models import Candidate
from orchestrator.ranking import elo_update, rank_round


def test_elo_update_equal_ratings_winner_gains_half_k():
    new_a, new_b = elo_update(1200.0, 1200.0, "A", k_factor=32.0)
    assert abs(new_a - 1216.0) < 1e-9
    assert abs(new_b - 1184.0) < 1e-9


def test_elo_update_upset_produces_bigger_swing_than_expected_win():
    # B (rated much lower) beats A (rated much higher): an upset.
    upset_a, upset_b = elo_update(1400.0, 1000.0, "B", k_factor=32.0)
    # A (rated much higher) beats B: an expected win.
    expected_a, expected_b = elo_update(1400.0, 1000.0, "A", k_factor=32.0)
    upset_swing = abs(upset_b - 1000.0)
    expected_swing = abs(expected_a - 1400.0)
    assert upset_swing > expected_swing


@pytest.mark.asyncio
async def test_rank_round_applies_verdicts_to_candidate_elo():
    a = Candidate(text="A", round_num=1, source_strategy="divergent", elo=1200.0)
    b = Candidate(text="B", round_num=1, source_strategy="divergent", elo=1200.0)

    backend = Mock()
    backend.call.return_value = json.dumps({"winner": "A", "reasoning": "A is more feasible"})

    proximity = Mock()
    proximity.pick_debate_pairs.return_value = [(a, b)]

    verdicts = await rank_round([a, b], backend, proximity, n_pairs=1)

    assert len(verdicts) == 1
    assert verdicts[0].winner_id == a.candidate_id
    assert a.elo > 1200.0
    assert b.elo < 1200.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_ranking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.ranking'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/ranking.py
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
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    actual_a = 1.0 if winner == "A" else 0.0
    actual_b = 1.0 - actual_a
    new_a = rating_a + k_factor * (actual_a - expected_a)
    new_b = rating_b + k_factor * (actual_b - expected_b)
    return new_a, new_b


def _debate_one(backend, candidate_a, candidate_b) -> DebateVerdict:
    user_prompt = f"Hypothesis A: {candidate_a.text}\n\nHypothesis B: {candidate_b.text}"
    parsed = call_with_json_retry(backend, "ranking", _SYSTEM_PROMPT, user_prompt)
    winner_id = candidate_a.candidate_id if parsed["winner"] == "A" else candidate_b.candidate_id
    return DebateVerdict(
        candidate_a_id=candidate_a.candidate_id, candidate_b_id=candidate_b.candidate_id,
        winner_id=winner_id, reasoning=parsed.get("reasoning", ""),
    )


def _apply_verdict(candidate_a, candidate_b, verdict: DebateVerdict) -> None:
    winner = "A" if verdict.winner_id == candidate_a.candidate_id else "B"
    new_a, new_b = elo_update(candidate_a.elo, candidate_b.elo, winner)
    candidate_a.elo, candidate_b.elo = new_a, new_b


async def rank_round(pool: list, backend, proximity, n_pairs: int) -> list:
    pairs = proximity.pick_debate_pairs(pool, n_pairs)
    verdicts = list(await asyncio.gather(*[
        asyncio.to_thread(_debate_one, backend, a, b) for a, b in pairs
    ]))
    for (candidate_a, candidate_b), verdict in zip(pairs, verdicts):
        _apply_verdict(candidate_a, candidate_b, verdict)
    return verdicts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_ranking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/ranking.py tests/orchestrator/test_ranking.py
git commit -m "Add Ranking stage: pairwise debate + Elo"
```

---

## Task 14: Evolution stage

**Files:**
- Create: `src/orchestrator/evolution.py`
- Test: `tests/orchestrator/test_evolution.py`

**Interfaces:**
- Consumes: `orchestrator.json_utils.call_with_json_retry`;
  `orchestrator.models.Candidate`; `orchestrator.retrieval.search_literature,
  format_papers_for_prompt`.
- Produces: `async def evolve_round(pool, backend, round_num, topic) ->
  list[Candidate]`.

Five named strategies (spec §5.2, §9): combine, simplify,
ground_in_literature, analogical, and deliberate_divergence — the last one
runs **every round, unconditionally**, never gated (see Global
Constraints). `combine` only runs when the pool has at least 2 candidates
to combine.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_evolution.py
import json
from unittest.mock import patch

import pytest

from orchestrator.evolution import evolve_round
from orchestrator.models import Candidate


class _FakeBackend:
    def __init__(self, ideas):
        self.ideas = list(ideas)
        self.call_count = 0

    def call(self, role, system_prompt, user_prompt):
        idea = self.ideas[self.call_count]
        self.call_count += 1
        return json.dumps({"idea": idea})


@pytest.mark.asyncio
async def test_evolve_round_always_includes_deliberate_divergence():
    pool = [
        Candidate(text="idea A", round_num=1, source_strategy="divergent", elo=1300.0),
        Candidate(text="idea B", round_num=1, source_strategy="divergent", elo=1250.0),
    ]
    backend = _FakeBackend(["diverged", "combined", "simplified", "grounded", "analogized"])

    with patch("orchestrator.evolution.search_literature", return_value=[]):
        result = await evolve_round(pool, backend, round_num=2, topic="t")

    strategies = {c.source_strategy for c in result}
    assert "deliberate_divergence" in strategies
    assert len(result) == 5  # divergence + combine + simplify + ground + analogical


@pytest.mark.asyncio
async def test_evolve_round_skips_combine_when_pool_has_one_candidate():
    pool = [Candidate(text="idea A", round_num=1, source_strategy="divergent", elo=1300.0)]
    backend = _FakeBackend(["diverged", "simplified", "grounded", "analogized"])

    with patch("orchestrator.evolution.search_literature", return_value=[]):
        result = await evolve_round(pool, backend, round_num=2, topic="t")

    strategies = {c.source_strategy for c in result}
    assert "combine" not in strategies
    assert len(result) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_evolution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.evolution'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/evolution.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_evolution.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/evolution.py tests/orchestrator/test_evolution.py
git commit -m "Add Evolution stage: 5 named strategies, mandatory divergence"
```

---

## Task 15: Meta-review stage

**Files:**
- Create: `src/orchestrator/meta_review.py`
- Test: `tests/orchestrator/test_meta_review.py`

**Interfaces:**
- Consumes: `orchestrator.models.Candidate, CritiqueResult, DebateVerdict`.
- Produces: `def synthesize_meta_review(backend, candidates, critiques,
  debates) -> str` — plain freeform text, no JSON parsing (this output
  isn't structured data, just guidance text for the next round).

This is the round-over-round learning mechanism (spec §5.2), distinct from
cross-run memory: its output is threaded into the *next* round's
Generation call via `meta_review_guidance` — wired in Task 16's Supervisor
loop, not dropped after being computed.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_meta_review.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_meta_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.meta_review'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/meta_review.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_meta_review.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/meta_review.py tests/orchestrator/test_meta_review.py
git commit -m "Add Meta-review stage"
```

---

## Task 16: Supervisor (round loop, call budget, final selection)

**Files:**
- Create: `src/orchestrator/supervisor.py`
- Test: `tests/orchestrator/test_supervisor.py`

**Interfaces:**
- Consumes: everything from Tasks 2–15 — `RunConfig`, `RunStore`,
  `MemoryStore`, `Proximity`, `generate_round`, `reflect_round`,
  `filter_passed`, `rank_round`, `evolve_round`, `synthesize_meta_review`,
  `MemoryEntry`, `RunSummary`, `Candidate`.
- Produces: `CallBudgetExceeded(Exception)`, `CountingBackend(inner,
  max_calls)`, `async def run_supervisor(config, backend, embedding_backend,
  store, memory) -> RunSummary`.

Ties every stage together per spec §5.2/§5.3/§7: drives the round loop,
threads `meta_review_guidance` from one round into the next round's
Generation call (the actual round-over-round learning mechanism — this is
the one thing that must not become a stub that computes a summary and
discards it), enforces the hard model-call cap via `CountingBackend`, caps
the working pool at `config.pool_size` after each round's debates, and
performs the diversity-aware final selection (K=1 or K>1) via Proximity.
Idempotent/resumable: a run already marked `"completed"` returns
immediately without re-running; an interrupted run resumes from
`store.last_completed_round(...) + 1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_supervisor.py
from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrator.memory import MemoryStore
from orchestrator.models import Candidate, DebateVerdict, RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import CallBudgetExceeded, CountingBackend, run_supervisor


def _config(**overrides):
    defaults = dict(
        topic="t", domain="Computer Science", subfield="t", rounds=2,
        pool_size=10, n_divergent_per_round=2, debate_pairs_per_round=1, k_final=1,
        max_model_calls=100, run_id="run-1",
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _fake_embedding_backend():
    backend = Mock()
    backend.embed.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    return backend


def _candidate(text, elo=1200.0):
    return Candidate(text=text, round_num=1, source_strategy="divergent", elo=elo, embedding=[1.0, 0.0])


@pytest.mark.asyncio
async def test_run_supervisor_completes_and_threads_meta_review_guidance(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config()

    generate_calls = []

    async def fake_generate_round(cfg, backend, mem, prox, round_num, meta_review_guidance=""):
        generate_calls.append(meta_review_guidance)
        return [_candidate(f"idea round {round_num}")]

    async def fake_reflect_round(pool, backend):
        return []

    def fake_filter_passed(pool, critiques):
        return pool

    async def fake_rank_round(pool, backend, proximity, n_pairs):
        return []

    async def fake_evolve_round(pool, backend, round_num, topic):
        return []

    with patch("orchestrator.supervisor.generate_round", fake_generate_round), \
         patch("orchestrator.supervisor.reflect_round", fake_reflect_round), \
         patch("orchestrator.supervisor.filter_passed", fake_filter_passed), \
         patch("orchestrator.supervisor.rank_round", fake_rank_round), \
         patch("orchestrator.supervisor.evolve_round", fake_evolve_round), \
         patch("orchestrator.supervisor.synthesize_meta_review", return_value="guidance from this round"):
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    assert summary.status == "completed"
    assert generate_calls == ["", "guidance from this round"]  # round 2 received round 1's guidance
    assert store.get_run_status(config.run_id) == "completed"


@pytest.mark.asyncio
async def test_run_supervisor_truncates_when_call_budget_exceeded(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config(max_model_calls=0)

    async def fake_generate_round(cfg, backend, mem, prox, round_num, meta_review_guidance=""):
        backend.call("generation", "sys", "usr")  # triggers CallBudgetExceeded immediately
        return []

    with patch("orchestrator.supervisor.generate_round", fake_generate_round):
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    assert summary.status == "truncated_call_budget"
    assert store.get_run_status(config.run_id) == "truncated_call_budget"


@pytest.mark.asyncio
async def test_run_supervisor_skips_already_completed_run(tmp_path):
    store = RunStore(tmp_path / "test.db")
    memory = MemoryStore(tmp_path / "test.db")
    config = _config(rounds=1)
    store.create_run(config)
    store.save_candidate(config.run_id, _candidate("existing idea"))
    store.mark_run_completed(config.run_id, total_model_calls=5, compute_multiplier=0.5)

    with patch("orchestrator.supervisor.generate_round", AsyncMock()) as fake_generate:
        summary = await run_supervisor(config, Mock(), _fake_embedding_backend(), store, memory)

    fake_generate.assert_not_called()
    assert summary.status == "completed"
    assert len(summary.final_candidates) == 1


def test_counting_backend_raises_past_budget():
    inner = Mock()
    inner.call.return_value = "response"
    counting = CountingBackend(inner, max_calls=2)

    counting.call("role", "sys", "usr")
    counting.call("role", "sys", "usr")
    with pytest.raises(CallBudgetExceeded):
        counting.call("role", "sys", "usr")
    assert inner.call.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_supervisor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.supervisor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/supervisor.py
"""Supervisor (spec §5.2): holds run config, drives the round loop,
enforces the hard call-cap (spec §7), and performs final selection
(spec §5.3) — K=1 (benchmark mode) or diversity-aware K>1 (real use).
"""
from orchestrator.evolution import evolve_round
from orchestrator.generation import generate_round
from orchestrator.meta_review import synthesize_meta_review
from orchestrator.models import MemoryEntry, RunSummary
from orchestrator.proximity import Proximity
from orchestrator.ranking import rank_round
from orchestrator.reflection import filter_passed, reflect_round

AGENTIDEABENCH_ACTIVE_BUDGET = 10


class CallBudgetExceeded(Exception):
    pass


class CountingBackend:
    def __init__(self, inner, max_calls: int):
        self.inner = inner
        self.max_calls = max_calls
        self.count = 0

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        if self.count >= self.max_calls:
            raise CallBudgetExceeded(f"Model call budget of {self.max_calls} exceeded")
        self.count += 1
        return self.inner.call(role, system_prompt, user_prompt)


def _trim_pool(pool: list, pool_size: int) -> list:
    if len(pool) <= pool_size:
        return pool
    return sorted(pool, key=lambda c: c.elo, reverse=True)[:pool_size]


async def run_supervisor(config, backend, embedding_backend, store, memory) -> RunSummary:
    store.create_run(config)
    proximity = Proximity(embedding_backend)

    if store.get_run_status(config.run_id) == "completed":
        pool = _trim_pool(store.get_pool(config.run_id), config.pool_size)
        proximity.compute_embeddings(pool)
        final_candidates = proximity.select_diverse_topk(pool, config.k_final, config.diverse_selection_similarity_threshold)
        return RunSummary(run_id=config.run_id, total_model_calls=0, compute_multiplier=0.0, final_candidates=final_candidates, status="completed")

    counting_backend = CountingBackend(backend, config.max_model_calls)
    start_round = store.last_completed_round(config.run_id) + 1
    pool = _trim_pool(store.get_pool(config.run_id), config.pool_size) if start_round > 1 else []
    meta_review_guidance = ""
    truncated = False

    try:
        for round_num in range(start_round, config.rounds + 1):
            store.record_round_start(config.run_id, round_num)

            new_candidates = await generate_round(config, counting_backend, memory, proximity, round_num, meta_review_guidance)
            pool.extend(new_candidates)

            critiques = await reflect_round(pool, counting_backend)
            pool = filter_passed(pool, critiques)

            elo_before = {c.candidate_id: c.elo for c in pool}
            debates = await rank_round(pool, counting_backend, proximity, config.debate_pairs_per_round)
            by_id = {c.candidate_id: c for c in pool}
            for verdict in debates:
                store.save_debate(
                    config.run_id, round_num, verdict,
                    (elo_before[verdict.candidate_a_id], elo_before[verdict.candidate_b_id]),
                    (by_id[verdict.candidate_a_id].elo, by_id[verdict.candidate_b_id].elo),
                )

            pool = _trim_pool(pool, config.pool_size)

            evolved = await evolve_round(pool, counting_backend, round_num, config.topic)
            pool.extend(evolved)

            meta_review_guidance = synthesize_meta_review(counting_backend, pool, critiques, debates)

            for candidate in pool:
                store.save_candidate(config.run_id, candidate)
            for critique in critiques:
                store.save_critique(config.run_id, critique)
            store.record_round_complete(config.run_id, round_num, meta_review_guidance)
    except CallBudgetExceeded:
        truncated = True

    proximity.compute_embeddings(pool)
    final_candidates = proximity.select_diverse_topk(pool, config.k_final, config.diverse_selection_similarity_threshold)
    store.save_final_selection(config.run_id, final_candidates)

    for candidate in final_candidates:
        memory.write(
            MemoryEntry(
                domain=config.domain, subfield=config.subfield, text=candidate.text,
                elo=candidate.elo, run_id=config.run_id, embedding=candidate.embedding,
            ),
            proximity, config.duplicate_similarity_threshold,
        )

    compute_multiplier = counting_backend.count / AGENTIDEABENCH_ACTIVE_BUDGET
    if truncated:
        store.mark_run_truncated(config.run_id, counting_backend.count, compute_multiplier)
    else:
        store.mark_run_completed(config.run_id, counting_backend.count, compute_multiplier)

    return RunSummary(
        run_id=config.run_id, total_model_calls=counting_backend.count, compute_multiplier=compute_multiplier,
        final_candidates=final_candidates, status="truncated_call_budget" if truncated else "completed",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_supervisor.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/supervisor.py tests/orchestrator/test_supervisor.py
git commit -m "Add Supervisor: round loop, call budget, final selection"
```

---

## Task 17: CLI entry point + manual smoke test scripts

**Files:**
- Create: `src/orchestrator/cli.py`
- Create: `scripts/orchestrator_smoke_openrouter.py`
- Create: `scripts/orchestrator_smoke_claude_code.py`
- Test: `tests/orchestrator/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 4–16.
- Produces: `build_backend(kind: str)`, `main()` (argparse entry point,
  runnable as `python -m orchestrator.cli`).

The two manual scripts cover spec §8's testing-plan items 2 and 3
("Orchestrator smoke test, OpenRouter backend" / "...Claude Code backend")
— they are the only place orchestrator code touches `benchmark`, and only
to score the produced hypothesis as a boundary check, matching Plan 1
Task 8's pattern of a manual, non-pytest verification script.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_cli.py
from unittest.mock import Mock, patch

from orchestrator.cli import build_backend, main


def test_build_backend_openrouter():
    from orchestrator.backend import OpenRouterBackend
    backend = build_backend("openrouter")
    assert isinstance(backend, OpenRouterBackend)


def test_build_backend_claude_code():
    from orchestrator.claude_code_backend import ClaudeCodeBackend
    backend = build_backend("claude_code")
    assert isinstance(backend, ClaudeCodeBackend)


def test_main_wires_args_into_run_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["orchestrator", "--topic", "t", "--domain", "Computer Science", "--subfield", "t", "--rounds", "2", "--k", "3"],
    )
    fake_summary = Mock(run_id="r1", status="completed", total_model_calls=10, compute_multiplier=1.0, final_candidates=[])

    with patch("orchestrator.cli.build_backend", return_value=Mock()), \
         patch("orchestrator.cli.OpenRouterEmbeddingBackendFake", create=True), \
         patch("orchestrator.cli.RunStore", return_value=Mock()), \
         patch("orchestrator.cli.MemoryStore", return_value=Mock()), \
         patch("orchestrator.cli.asyncio.run", return_value=fake_summary) as mock_run:
        main()

    assert mock_run.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/orchestrator/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestrator/cli.py
"""CLI entry point: python -m orchestrator.cli --topic "..." --domain "..." \
    --subfield "..." --backend openrouter --rounds 3 --k 1
"""
import argparse
import asyncio

from config.orchestrator import DB_PATH, DEFAULT_GENERATION_MODEL, OPENROUTER_API_KEY
from orchestrator.backend import OpenRouterBackend
from orchestrator.claude_code_backend import ClaudeCodeBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


def build_backend(kind: str):
    if kind == "claude_code":
        return ClaudeCodeBackend()
    return OpenRouterBackend(model=DEFAULT_GENERATION_MODEL, api_key=OPENROUTER_API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ideation orchestrator on a topic.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--subfield", required=True)
    parser.add_argument("--backend", choices=["openrouter", "claude_code"], default="openrouter")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    config = RunConfig(
        topic=args.topic, domain=args.domain, subfield=args.subfield,
        backend_kind=args.backend, rounds=args.rounds, k_final=args.k,
    )
    backend = build_backend(args.backend)
    embedding_backend = SentenceTransformerBackend()
    store = RunStore(DB_PATH)
    memory = MemoryStore(DB_PATH)

    summary = asyncio.run(run_supervisor(config, backend, embedding_backend, store, memory))

    print(
        f"Run {summary.run_id}: {summary.status}, {summary.total_model_calls} calls "
        f"({summary.compute_multiplier:.1f}x AgentIdeaBench's Active budget)"
    )
    for rank, candidate in enumerate(summary.final_candidates, start=1):
        print(f"\n[{rank}] (elo {candidate.elo:.0f}) {candidate.text}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/orchestrator/test_cli.py -v`
Expected: PASS (3 tests). Note: the test patches
`orchestrator.cli.OpenRouterEmbeddingBackendFake` with `create=True` purely
as a no-op placeholder patch target so the test file's `with` block stays
uniform — `main()` itself constructs `SentenceTransformerBackend()`
directly and does not need this patch to pass; if it causes confusion,
simplify by removing that line, since `SentenceTransformerBackend`'s model
loads lazily (Task 6) and is never triggered in this test (`asyncio.run` is
mocked, so `run_supervisor` — and therefore any embedding calls — never
actually executes).

- [ ] **Step 5: Write the OpenRouter manual smoke script**

```python
# scripts/orchestrator_smoke_openrouter.py
"""Manual smoke test (spec §8 item 2): OpenRouter backend, one topic, small
round budget. Confirms a valid hypothesis is produced, then scores it
through the benchmark harness (Plan 1) as a boundary integration check —
this is the only place orchestrator code touches `benchmark`.

Run: python scripts/orchestrator_smoke_openrouter.py
Requires OPENROUTER_API_KEY set in the environment. Makes real OpenRouter
API calls and, if `benchmark` is available, 3 more calls to score the
result — costs a small amount of real money.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.orchestrator import DEFAULT_GENERATION_MODEL, OPENROUTER_API_KEY
from orchestrator.backend import OpenRouterBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


async def main() -> None:
    config = RunConfig(
        topic="graph neural network expressiveness", domain="Computer Science",
        subfield="graph neural network expressiveness", backend_kind="openrouter",
        rounds=1, pool_size=6, n_divergent_per_round=3, debate_pairs_per_round=1, k_final=1,
        max_model_calls=30,
    )
    backend = OpenRouterBackend(model=DEFAULT_GENERATION_MODEL, api_key=OPENROUTER_API_KEY)
    embedding_backend = SentenceTransformerBackend()
    db_path = Path("/tmp/orchestrator_smoke_openrouter.db")
    store = RunStore(db_path)
    memory = MemoryStore(db_path)

    summary = await run_supervisor(config, backend, embedding_backend, store, memory)
    print(f"Status: {summary.status}, calls: {summary.total_model_calls}")
    for candidate in summary.final_candidates:
        print(f"\nHypothesis (elo {candidate.elo:.0f}): {candidate.text}")

    try:
        from benchmark.models import Hypothesis
        from benchmark.scorer import score_hypothesis
    except ImportError:
        print("\n(benchmark package not available — skipping scoring step)")
        return

    for candidate in summary.final_candidates:
        hyp = Hypothesis(
            topic=config.topic, domain=config.domain, subfield=config.subfield,
            text=candidate.text, source="our_swarm",
        )
        result = score_hypothesis(hyp)
        print(f"\nBenchmark score for this hypothesis: {result.weighted_total:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Write the Claude Code manual smoke script**

```python
# scripts/orchestrator_smoke_claude_code.py
"""Manual smoke test (spec §8 item 3): same as
orchestrator_smoke_openrouter.py, with backend swapped to ClaudeCodeBackend
— proves the pluggable-backend design actually works with just a backend
swap (spec §5.1). Requires the `claude` CLI installed and authenticated;
also requires OPENROUTER_API_KEY for the embedding-similarity service
(Proximity always uses local sentence-transformers, no API key needed
there — see Task 6 — but the topic/domain match the OpenRouter smoke
script's for a fair comparison of the two backends' output on request).

Run: python scripts/orchestrator_smoke_claude_code.py
Runs `claude -p ...` subprocesses that consume the user's Claude Code plan.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.claude_code_backend import ClaudeCodeBackend
from orchestrator.embeddings import SentenceTransformerBackend
from orchestrator.memory import MemoryStore
from orchestrator.models import RunConfig
from orchestrator.store import RunStore
from orchestrator.supervisor import run_supervisor


async def main() -> None:
    config = RunConfig(
        topic="graph neural network expressiveness", domain="Computer Science",
        subfield="graph neural network expressiveness", backend_kind="claude_code",
        rounds=1, pool_size=6, n_divergent_per_round=3, debate_pairs_per_round=1, k_final=1,
        max_model_calls=30,
    )
    backend = ClaudeCodeBackend()
    embedding_backend = SentenceTransformerBackend()
    db_path = Path("/tmp/orchestrator_smoke_claude_code.db")
    store = RunStore(db_path)
    memory = MemoryStore(db_path)

    summary = await run_supervisor(config, backend, embedding_backend, store, memory)
    print(f"Status: {summary.status}, calls: {summary.total_model_calls}")
    for candidate in summary.final_candidates:
        print(f"\nHypothesis (elo {candidate.elo:.0f}): {candidate.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 7: Run both manual scripts and inspect the output**

Run: `python scripts/orchestrator_smoke_openrouter.py`
Run: `python scripts/orchestrator_smoke_claude_code.py`

Expected for each: a "completed" status, a non-empty final hypothesis, and
(OpenRouter script only, if `benchmark` is importable) a benchmark score
printed. If either script errors out, the failure is more informative than
a green pytest suite here — this is real end-to-end wiring across every
stage plus a real backend, exactly the class of bug mocked unit tests
can't catch.

- [ ] **Step 8: Commit**

```bash
git add src/orchestrator/cli.py tests/orchestrator/test_cli.py scripts/orchestrator_smoke_openrouter.py scripts/orchestrator_smoke_claude_code.py
git commit -m "Add orchestrator CLI and manual smoke test scripts"
```

---

## Explicitly deferred (not silently dropped)

- **Baseline ladder / compute-matched best-of-N baseline (spec §8 item 4,
  design-recommendations.md §6)** — comparing bare LLM / LLM+RAG /
  single-round swarm / full swarm against AgentIdeaBench's published
  numbers needs its own eval-harness plan (Plan 3) that calls
  `benchmark.scorer.score_hypothesis` on each tier's output. This plan
  provides the reusable primitive Plan 1's own "Explicitly deferred"
  section expected: `generation._generate_one` (Task 11) already produces
  one independent single-shot hypothesis with no debate/evolution — Plan 3
  can call it N times directly, or drive `run_supervisor` with
  `rounds=1` and no evolution for the "single-round swarm" tier.
- **Diversity sanity check (LiveIdeaBench-style per-role probing,
  design-recommendations.md §6)** — not one of spec §5's explicit
  components; not built here.
- **Human-anchoring spot-check process (design-recommendations.md §6)** —
  a methodology/process, not code; nothing to implement.
- Everything already excluded by the spec's own v1 scope (§10): dynamic
  stopping instead of a fixed round budget, the full 100-subfield topic
  set, memory clustering/consolidation beyond dedup-on-write, and
  per-role backend mixing. Not this plan's job to add.

## Definition of done

All 17 tasks complete, `pytest tests/orchestrator/ -v` passes fully (mocked,
no network, no subprocess, no real embedding model load), and both manual
smoke scripts (Task 17, Steps 5–6) have been run at least once with real
credentials — OpenRouter backend and Claude Code backend — each producing a
"completed" run with a non-empty final hypothesis. At that point the
orchestrator is ready for the baseline-ladder eval harness (Plan 3, written
separately) and for real ideation runs.
