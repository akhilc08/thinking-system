# Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin Python wrapper that scores any research hypothesis using
AgentIdeaBench's real, published scoring pipeline (`LIT8D_SYSTEM` rubric,
per-hypothesis literature evidence, 3 fixed critics, drop-highest-per-dimension
aggregation), so results are directly comparable to their public leaderboard.

**Architecture:** Every scoring primitive is imported from the vendored
`vendor/AgentIdeaBench` repo, never reimplemented. Our own code is limited to:
sys.path/import wiring, a Semantic Scholar reference-text refetcher (their
release ships paper IDs only, not text), an aggregation function matching
their exact drop-highest-per-dimension + weighted-mean rule, and a baseline
loader for their published leaderboard numbers.

**Tech Stack:** Python 3.9+, pytest, `requests` (already a vendored dependency),
`pandas` (for reading the released `.csv.gz` files).

**Spec:** [`docs/superpowers/specs/2026-09-16-multiagent-ideation-system-design.md`](../specs/2026-09-16-multiagent-ideation-system-design.md) — implements §4 (Benchmark harness) of that spec.

## Global Constraints

- Files under 500 lines (project CLAUDE.md).
- TDD London School (mock-first): tests mock the vendored functions and all
  network calls; nothing in this plan's automated test suite makes a real
  HTTP call. The one exception is Task 8, a manual integration script, not
  part of the pytest suite.
- Source in `/src`, tests in `/tests`, config in `/config` (project CLAUDE.md).
- Never reimplement a vendored scoring primitive — import it. If a needed
  vendored function's behavior is unclear, read the actual source at the
  cited path/line range before writing the wrapper — do not guess.
- **Critical:** the rubric that produced AgentIdeaBench's published leaderboard
  is `LIT8D_SYSTEM` from `experiments/e13_litverify_rubric8.py`, **not** the
  default `SYSTEM_PROMPT` in `evaluation/absolute_scorer.py`. Every task that
  touches scoring must import from `experiments/`, never fall back to the
  `evaluation/` default.

---

## Task 1: Project scaffolding + vendor bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore` (append `.cache/`, `.env`, `__pycache__/`, `*.egg-info/`)
- Create: `src/benchmark/__init__.py`
- Create: `src/benchmark/vendor_bootstrap.py`
- Create: `config/benchmark.py`
- Test: `tests/benchmark/test_vendor_bootstrap.py`

**Interfaces:**
- Produces: `benchmark.vendor_bootstrap.LIT8D_SYSTEM: str`,
  `benchmark.vendor_bootstrap.score_one(idea, critic, domain, system_prompt, references) -> tuple[dict | None, str]`,
  `benchmark.vendor_bootstrap.trimmed_mean(values: list[float]) -> float | None`,
  `benchmark.vendor_bootstrap.build_evidence_for_item(item_id, grp, idea_text, cutoff, self_pid, self_title) -> tuple[list, list, int]`,
  `benchmark.vendor_bootstrap.format_evidence_block(evidence, cutoff) -> str`,
  `benchmark.vendor_bootstrap.SCORING_DIMS: list[str]`,
  `benchmark.vendor_bootstrap.SCORING_WEIGHTS: dict[str, float]`,
  `benchmark.vendor_bootstrap.ss_get(url, params, api_key, delay, backoffs=None) -> dict | None`,
  `benchmark.vendor_bootstrap.SS_BASE: str`,
  `benchmark.vendor_bootstrap.vendor_config`: the vendored `config` module (for `CRITIC_MODELS`, `OPENROUTER_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`).

- [ ] **Step 1: Set up project scaffolding**

Create `pyproject.toml`:

```toml
[project]
name = "thinking-system"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "requests",
    "pandas",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Append to `.gitignore`:

```
.cache/
.env
__pycache__/
*.egg-info/
*.pyc
```

Install dev deps: `pip install -e ".[dev]"`

- [ ] **Step 2: Copy and fill the vendored `.env`**

```bash
cp vendor/AgentIdeaBench/.env.example vendor/AgentIdeaBench/.env
```

Edit `vendor/AgentIdeaBench/.env` and fill in real values for
`OPENROUTER_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` (from the user). This file
is gitignored inside the vendored repo already — verify with
`git check-ignore vendor/AgentIdeaBench/.env` (should print the path, meaning
it's ignored).

- [ ] **Step 3: Write the failing test**

```python
# tests/benchmark/test_vendor_bootstrap.py
from benchmark import vendor_bootstrap as vb


def test_lit8d_system_is_the_evidence_grounded_rubric_not_the_simple_default():
    # Guards against silently importing evaluation/absolute_scorer's simpler
    # SYSTEM_PROMPT instead of the real production rubric.
    assert "EVIDENCE-BASED ORIGINALITY" in vb.LIT8D_SYSTEM
    assert "IDEA-LEVEL SCALE RECALIBRATION" in vb.LIT8D_SYSTEM


def test_scoring_dims_and_weights_match_published_values():
    assert vb.SCORING_DIMS == [
        "originality", "feasibility", "clarity", "impact", "specificity",
    ]
    assert vb.SCORING_WEIGHTS == {
        "originality": 2.0,
        "feasibility": 1.0,
        "clarity": 0.5,
        "impact": 1.5,
        "specificity": 0.5,
    }


def test_trimmed_mean_drops_the_highest_value():
    assert vb.trimmed_mean([3, 5, 9]) == 4.0  # drops 9, mean of [3, 5]


def test_vendor_callables_are_importable_and_callable():
    assert callable(vb.score_one)
    assert callable(vb.build_evidence_for_item)
    assert callable(vb.format_evidence_block)
    assert callable(vb.ss_get)
    assert isinstance(vb.SS_BASE, str) and vb.SS_BASE.startswith("https://")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/benchmark/test_vendor_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`

- [ ] **Step 5: Write minimal implementation**

```python
# config/benchmark.py
"""Project-level config for the benchmark harness (not the vendored repo's own config)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AgentIdeaBench"
CACHE_DIR = PROJECT_ROOT / ".cache" / "semantic_scholar"
```

```python
# src/benchmark/__init__.py
```

```python
# src/benchmark/vendor_bootstrap.py
"""
Single place that wires up imports from vendor/AgentIdeaBench.

Every other module in this package imports vendored scoring primitives from
HERE, never reaches into vendor/AgentIdeaBench directly. This is deliberate:
it's the one file responsible for making sure we use the real production
rubric (LIT8D_SYSTEM from experiments/e13_litverify_rubric8.py) rather than
the simpler default in evaluation/absolute_scorer.py.
"""
import sys

from config.benchmark import VENDOR_ROOT

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import config as vendor_config  # noqa: E402  (vendored repo's own config.py)
from experiments.e13_litverify_rubric8 import (  # noqa: E402
    LIT8D_SYSTEM,
    build_evidence_for_item,
    format_evidence_block,
)
from experiments.e10_idea_anchor_calibration import (  # noqa: E402
    score_one,
    trimmed_mean,
)
from utils.constants import SCORING_DIMS, SCORING_WEIGHTS  # noqa: E402
from data_collection.fetch_ss_search import _get as ss_get, SS_BASE  # noqa: E402

__all__ = [
    "vendor_config", "LIT8D_SYSTEM", "build_evidence_for_item",
    "format_evidence_block", "score_one", "trimmed_mean", "SCORING_DIMS",
    "SCORING_WEIGHTS", "ss_get", "SS_BASE",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/benchmark/test_vendor_bootstrap.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore config/benchmark.py src/benchmark/__init__.py src/benchmark/vendor_bootstrap.py tests/benchmark/test_vendor_bootstrap.py
git commit -m "Add project scaffolding and vendor bootstrap for AgentIdeaBench reuse"
```

---

## Task 2: Data models

**Files:**
- Create: `src/benchmark/models.py`
- Test: `tests/benchmark/test_models.py`

**Interfaces:**
- Consumes: nothing (pure data classes).
- Produces: `Hypothesis(topic, domain, subfield, text, source)`,
  `DimensionScore(name: str, score: int, reasoning: str)`,
  `ScoreResult(hypothesis: Hypothesis, per_critic: dict[str, dict[str, DimensionScore]], per_dimension_trimmed: dict[str, float], weighted_total: float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_models.py
from benchmark.models import DimensionScore, Hypothesis, ScoreResult


def test_hypothesis_holds_identifying_fields():
    h = Hypothesis(
        topic="graph neural network expressiveness",
        domain="Computer Science",
        subfield="graph neural network expressiveness",
        text="We hypothesize that...",
        source="our_swarm",
    )
    assert h.domain == "Computer Science"
    assert h.source == "our_swarm"


def test_score_result_holds_aggregation_output():
    h = Hypothesis("t", "Computer Science", "t", "text", "our_swarm")
    result = ScoreResult(
        hypothesis=h,
        per_critic={
            "qwen/qwen3.6-plus": {
                "originality": DimensionScore("originality", 7, "reason"),
            },
        },
        per_dimension_trimmed={"originality": 6.5},
        weighted_total=6.2,
    )
    assert result.weighted_total == 6.2
    assert result.per_dimension_trimmed["originality"] == 6.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark/models.py
"""Plain data holders for the benchmark harness. No behavior lives here."""
from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    topic: str
    domain: str
    subfield: str
    text: str
    source: str  # e.g. "our_swarm", "agentideabench_static", "agentideabench_active"


@dataclass
class DimensionScore:
    name: str
    score: int
    reasoning: str


@dataclass
class ScoreResult:
    hypothesis: Hypothesis
    per_critic: dict = field(default_factory=dict)       # critic_model -> {dim: DimensionScore}
    per_dimension_trimmed: dict = field(default_factory=dict)  # dim -> float
    weighted_total: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/models.py tests/benchmark/test_models.py
git commit -m "Add data models for benchmark harness"
```

---

## Task 3: Score aggregation (pure function, no network)

**Files:**
- Create: `src/benchmark/scorer.py`
- Test: `tests/benchmark/test_scorer.py`

**Interfaces:**
- Consumes: `benchmark.vendor_bootstrap.trimmed_mean`, `SCORING_DIMS`, `SCORING_WEIGHTS`.
- Produces: `aggregate_dimension_scores(per_critic_scores: dict[str, dict[str, int]]) -> tuple[dict[str, float], float]`
  — returns `(per_dimension_trimmed, weighted_total)`. This must exactly match
  the aggregation used to build AgentIdeaBench's published `lit8d_scores_3seed`
  table: **per dimension**, collect each critic's score, drop the single
  highest, average the rest, then combine dimensions with `SCORING_WEIGHTS`
  and divide by the sum of the weights (5.5) — confirmed from
  `experiments/e16_v4_full_lit8d.py::analyze()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_scorer.py
from benchmark.scorer import aggregate_dimension_scores


def test_aggregate_drops_highest_score_per_dimension_then_weights():
    per_critic_scores = {
        "qwen/qwen3.6-plus": {
            "originality": 9, "feasibility": 6, "clarity": 7, "impact": 8, "specificity": 6,
        },
        "moonshotai/kimi-k2.6": {
            "originality": 5, "feasibility": 7, "clarity": 6, "impact": 5, "specificity": 5,
        },
        "z-ai/glm-5.1": {
            "originality": 6, "feasibility": 5, "clarity": 5, "impact": 6, "specificity": 4,
        },
    }
    per_dim, weighted = aggregate_dimension_scores(per_critic_scores)

    # originality: [9, 5, 6] -> drop 9 -> mean([5, 6]) = 5.5
    assert per_dim["originality"] == 5.5
    # feasibility: [6, 7, 5] -> drop 7 -> mean([6, 5]) = 5.5
    assert per_dim["feasibility"] == 5.5
    # clarity: [7, 6, 5] -> drop 7 -> mean([6, 5]) = 5.5
    assert per_dim["clarity"] == 5.5
    # impact: [8, 5, 6] -> drop 8 -> mean([5, 6]) = 5.5
    assert per_dim["impact"] == 5.5
    # specificity: [6, 5, 4] -> drop 6 -> mean([5, 4]) = 4.5
    assert per_dim["specificity"] == 4.5

    # weighted = (2*5.5 + 1.0*5.5 + 0.5*5.5 + 1.5*5.5 + 0.5*4.5) / 5.5
    expected = (2.0 * 5.5 + 1.0 * 5.5 + 0.5 * 5.5 + 1.5 * 5.5 + 0.5 * 4.5) / 5.5
    assert abs(weighted - expected) < 1e-9


def test_aggregate_handles_two_critics_by_using_lower_of_the_two():
    # trimmed_mean with len < 2 items after nothing to drop still applies
    # vendor's own rule: len(vs) < 2 returns vs[0]; with exactly 2 values,
    # trimmed_mean still drops the highest and returns the single remaining one.
    per_critic_scores = {
        "qwen/qwen3.6-plus": {"originality": 8, "feasibility": 5, "clarity": 5, "impact": 5, "specificity": 5},
        "moonshotai/kimi-k2.6": {"originality": 4, "feasibility": 5, "clarity": 5, "impact": 5, "specificity": 5},
    }
    per_dim, _ = aggregate_dimension_scores(per_critic_scores)
    assert per_dim["originality"] == 4  # drop 8, left with [4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.scorer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark/scorer.py
"""Scores a hypothesis using AgentIdeaBench's real critic pipeline.

Aggregation logic mirrors experiments/e16_v4_full_lit8d.py::analyze() exactly:
per dimension, drop the single highest of the critics' scores, average the
rest, then combine dimensions with SCORING_WEIGHTS / sum(SCORING_WEIGHTS).
"""
from benchmark.vendor_bootstrap import SCORING_DIMS, SCORING_WEIGHTS, trimmed_mean


def aggregate_dimension_scores(
    per_critic_scores: dict[str, dict[str, int]],
) -> tuple[dict[str, float], float]:
    per_dimension_trimmed: dict[str, float] = {}
    for dim in SCORING_DIMS:
        values = [scores[dim] for scores in per_critic_scores.values()]
        per_dimension_trimmed[dim] = trimmed_mean(values)

    weight_sum = sum(SCORING_WEIGHTS.values())
    weighted_total = sum(
        per_dimension_trimmed[dim] * SCORING_WEIGHTS[dim] for dim in SCORING_DIMS
    ) / weight_sum

    return per_dimension_trimmed, weighted_total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_scorer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/scorer.py tests/benchmark/test_scorer.py
git commit -m "Add score aggregation matching AgentIdeaBench's published methodology"
```

---

## Task 4: Reference fetcher (Semantic Scholar refetch + disk cache)

**Files:**
- Create: `src/benchmark/reference_fetcher.py`
- Test: `tests/benchmark/test_reference_fetcher.py`

**Interfaces:**
- Consumes: `benchmark.vendor_bootstrap.ss_get`, `SS_BASE`, `vendor_config.SEMANTIC_SCHOLAR_API_KEY`.
- Produces: `fetch_paper(ss_paper_id: str) -> dict | None` — returns
  `{"title": str, "abstract": str, "year": int | None}` or `None` if not found.
  Caches to disk at `config.benchmark.CACHE_DIR / f"{ss_paper_id}.json"` so
  repeat runs don't re-hit the API.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_reference_fetcher.py
import json
from unittest.mock import patch

from benchmark.reference_fetcher import fetch_paper


def test_fetch_paper_returns_title_abstract_year(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    fake_response = {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture...",
        "year": 2017,
    }
    with patch("benchmark.reference_fetcher.ss_get", return_value=fake_response) as mock_get:
        result = fetch_paper("204e3073870fae3d05bcbc2f6a8e263d9b72e776")

    assert result == fake_response
    mock_get.assert_called_once()


def test_fetch_paper_returns_none_when_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    with patch("benchmark.reference_fetcher.ss_get", return_value=None):
        result = fetch_paper("unknown_id")
    assert result is None


def test_fetch_paper_uses_disk_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.reference_fetcher.CACHE_DIR", tmp_path)
    fake_response = {"title": "Cached Paper", "abstract": "abstract text", "year": 2020}
    with patch("benchmark.reference_fetcher.ss_get", return_value=fake_response) as mock_get:
        fetch_paper("some_id")   # first call: hits ss_get, writes cache
        fetch_paper("some_id")   # second call: should read from cache, not call ss_get again
    assert mock_get.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_reference_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.reference_fetcher'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark/reference_fetcher.py
"""Refetches paper title/abstract/year by Semantic Scholar paper ID.

AgentIdeaBench's release ships only ss_paper_id values (abstracts are
publisher content their API license doesn't let them redistribute) — this
module refetches the text ourselves, under our own Semantic Scholar API
agreement, and caches it to disk since paper text never changes.
"""
import json

from benchmark.vendor_bootstrap import SS_BASE, ss_get, vendor_config
from config.benchmark import CACHE_DIR

FIELDS = "title,abstract,year"


def fetch_paper(ss_paper_id: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ss_paper_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    result = ss_get(
        f"{SS_BASE}/paper/{ss_paper_id}",
        {"fields": FIELDS},
        vendor_config.SEMANTIC_SCHOLAR_API_KEY,
        0.2,
    )
    if not result:
        return None

    paper = {
        "title": result.get("title") or "",
        "abstract": result.get("abstract") or "",
        "year": result.get("year"),
    }
    cache_path.write_text(json.dumps(paper))
    return paper
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_reference_fetcher.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/reference_fetcher.py tests/benchmark/test_reference_fetcher.py
git commit -m "Add Semantic Scholar reference fetcher with disk cache"
```

---

## Task 5: Evidence builder wrapper

**Files:**
- Create: `src/benchmark/evidence.py`
- Test: `tests/benchmark/test_evidence.py`

**Interfaces:**
- Consumes: `benchmark.vendor_bootstrap.build_evidence_for_item`, `format_evidence_block`.
- Produces: `build_evidence(item_id: str, idea_text: str, cutoff_date: str) -> str`
  — a formatted "PRIOR-ART SEARCH RESULTS" block, ready to pass as the
  `references` argument to `score_one`. Wraps the vendored two-step
  extract-queries-then-search-then-format flow behind one call, always with
  `grp="v4"` (i.e. not an anchor/control item) and no self-exclusion, since our
  hypotheses are never one of AgentIdeaBench's own reference papers.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_evidence.py
from unittest.mock import patch

from benchmark.evidence import build_evidence


def test_build_evidence_formats_retrieved_prior_art():
    fake_queries = ["contrastive learning molecular property prediction"]
    fake_hits = [{
        "paperId": "abc123",
        "title": "Contrastive Pretraining for Molecules",
        "year": 2023,
        "publicationDate": "2023-04-01",
        "citationCount": 42,
        "abstract": "We introduce a contrastive pretraining method...",
        "query": fake_queries[0],
    }]
    with patch(
        "benchmark.evidence.build_evidence_for_item",
        return_value=(fake_queries, fake_hits, 0),
    ) as mock_build:
        block = build_evidence(
            item_id="our_swarm|graph-nn-expressiveness",
            idea_text="We hypothesize that...",
            cutoff_date="2026-05-31",
        )

    mock_build.assert_called_once_with(
        "our_swarm|graph-nn-expressiveness", "v4", "We hypothesize that...",
        "2026-05-31", None, None,
    )
    assert "Contrastive Pretraining for Molecules" in block
    assert "2026-05-31" in block


def test_build_evidence_reports_no_papers_found():
    with patch("benchmark.evidence.build_evidence_for_item", return_value=([], [], 0)):
        block = build_evidence("item", "idea text", "2026-05-31")
    assert "no relevant papers" in block.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.evidence'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark/evidence.py
"""Grounded-originality evidence, built the same way AgentIdeaBench builds it
for its own published lit8d scores: extract search queries from the
hypothesis, run a date-filtered Semantic Scholar search, format the hits.
"""
from benchmark.vendor_bootstrap import build_evidence_for_item, format_evidence_block


def build_evidence(item_id: str, idea_text: str, cutoff_date: str) -> str:
    _queries, hits, _n_self = build_evidence_for_item(
        item_id, "v4", idea_text, cutoff_date, None, None,
    )
    return format_evidence_block(hits, cutoff_date)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_evidence.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/evidence.py tests/benchmark/test_evidence.py
git commit -m "Add grounded-evidence builder wrapper"
```

---

## Task 6: Hypothesis scorer (calls the 3 real critics)

**Files:**
- Modify: `src/benchmark/scorer.py`
- Modify: `tests/benchmark/test_scorer.py`

**Interfaces:**
- Consumes: `benchmark.vendor_bootstrap.score_one`, `LIT8D_SYSTEM`, `vendor_config.CRITIC_MODELS`; `benchmark.evidence.build_evidence`; `benchmark.scorer.aggregate_dimension_scores` (Task 3); `benchmark.models.Hypothesis`, `ScoreResult`, `DimensionScore`.
- Produces: `score_hypothesis(hypothesis: Hypothesis, cutoff_date: str = "2026-05-31") -> ScoreResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_scorer.py  (append to the file from Task 3)
from unittest.mock import patch

from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis


def _fake_scores(originality):
    return {
        "originality": {"score": originality, "reasoning": "r"},
        "feasibility": {"score": 6, "reasoning": "r"},
        "clarity": {"score": 6, "reasoning": "r"},
        "impact": {"score": 6, "reasoning": "r"},
        "specificity": {"score": 6, "reasoning": "r"},
    }


def test_score_hypothesis_calls_all_three_critics_and_aggregates():
    hyp = Hypothesis("t", "Computer Science", "t", "We hypothesize that...", "our_swarm")

    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (_fake_scores(5), "raw2"),
            (_fake_scores(6), "raw3"),
        ]
        result = score_hypothesis(hyp)

    assert mock_score_one.call_count == 3
    # originality [9, 5, 6] -> drop 9 -> mean([5, 6]) = 5.5
    assert result.per_dimension_trimmed["originality"] == 5.5
    assert set(result.per_critic.keys()) == {
        "qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1",
    }
    assert result.hypothesis is hyp


def test_score_hypothesis_skips_a_critic_whose_call_failed():
    hyp = Hypothesis("t", "Computer Science", "t", "text", "our_swarm")
    with patch("benchmark.scorer.vendor_config") as mock_cfg, \
         patch("benchmark.scorer.build_evidence", return_value="PRIOR-ART: none"), \
         patch("benchmark.scorer.score_one") as mock_score_one:
        mock_cfg.CRITIC_MODELS = ["qwen/qwen3.6-plus", "moonshotai/kimi-k2.6", "z-ai/glm-5.1"]
        mock_score_one.side_effect = [
            (_fake_scores(9), "raw1"),
            (None, "ERR: parse failure"),
            (_fake_scores(6), "raw3"),
        ]
        result = score_hypothesis(hyp)

    assert len(result.per_critic) == 2
    # originality [9, 6], len < 2 rule doesn't apply since we have 2 values ->
    # trimmed_mean([9, 6]) drops 9, returns [6]
    assert result.per_dimension_trimmed["originality"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_scorer.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_hypothesis'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/benchmark/scorer.py`:

```python
from benchmark.evidence import build_evidence
from benchmark.models import DimensionScore, ScoreResult
from benchmark.vendor_bootstrap import LIT8D_SYSTEM, score_one, vendor_config


def score_hypothesis(hypothesis, cutoff_date: str = "2026-05-31") -> ScoreResult:
    item_id = f"{hypothesis.source}|{hypothesis.subfield[:40]}"
    references = build_evidence(item_id, hypothesis.text, cutoff_date)

    per_critic: dict[str, dict[str, DimensionScore]] = {}
    for critic in vendor_config.CRITIC_MODELS:
        raw_scores, _raw_response = score_one(
            hypothesis.text, critic, hypothesis.domain, LIT8D_SYSTEM, references,
        )
        if raw_scores is None:
            continue
        per_critic[critic] = {
            dim: DimensionScore(dim, v["score"], v["reasoning"])
            for dim, v in raw_scores.items()
        }

    per_critic_int_scores = {
        critic: {dim: ds.score for dim, ds in dims.items()}
        for critic, dims in per_critic.items()
    }
    per_dimension_trimmed, weighted_total = aggregate_dimension_scores(per_critic_int_scores)

    return ScoreResult(
        hypothesis=hypothesis,
        per_critic=per_critic,
        per_dimension_trimmed=per_dimension_trimmed,
        weighted_total=weighted_total,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_scorer.py -v`
Expected: PASS (4 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/scorer.py tests/benchmark/test_scorer.py
git commit -m "Add score_hypothesis: calls the 3 real critics and aggregates"
```

---

## Task 7: Baseline loader

**Files:**
- Create: `src/benchmark/baseline_loader.py`
- Test: `tests/benchmark/test_baseline_loader.py`

**Interfaces:**
- Consumes: `pandas`; reads `vendor/AgentIdeaBench/release_data/core/lit8d_scores_3seed.csv.gz`.
- Produces: `load_published_score(idea_model: str, track: str, subdomain: str) -> float | None`
  — returns the same weighted-total number our own `score_hypothesis` produces,
  computed from the released per-critic rows for that (model, track,
  subfield), or `None` if no rows exist for that combination.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmark/test_baseline_loader.py
import pandas as pd

from benchmark.baseline_loader import load_published_score


def _fake_scores_df():
    # Mirrors release_data/core/lit8d_scores_3seed.csv.gz columns.
    rows = []
    for critic, o, f, c, i, s in [
        ("qwen/qwen3.6-plus", 9, 6, 7, 8, 6),
        ("moonshotai/kimi-k2.6", 5, 7, 6, 5, 5),
        ("z-ai/glm-5.1", 6, 5, 5, 6, 4),
    ]:
        rows.append({
            "idea_model": "glm/glm-5.1", "track": "C", "subdomain": "graph nn expressiveness",
            "critic_model": critic,
            "score_originality": o, "score_feasibility": f, "score_clarity": c,
            "score_impact": i, "score_specificity": s,
        })
    return pd.DataFrame(rows)


def test_load_published_score_matches_our_own_aggregation(monkeypatch):
    monkeypatch.setattr(
        "benchmark.baseline_loader._load_scores_dataframe",
        lambda: _fake_scores_df(),
    )
    score = load_published_score("glm/glm-5.1", "C", "graph nn expressiveness")
    # Same numbers as test_score_hypothesis_calls_all_three_critics_and_aggregates:
    # originality [9,5,6] -> drop 9 -> mean([5,6]) = 5.5, etc.
    assert score is not None
    assert 5.0 < score < 6.0


def test_load_published_score_returns_none_when_not_found(monkeypatch):
    monkeypatch.setattr(
        "benchmark.baseline_loader._load_scores_dataframe",
        lambda: _fake_scores_df(),
    )
    score = load_published_score("nonexistent/model", "C", "graph nn expressiveness")
    assert score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmark/test_baseline_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.baseline_loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark/baseline_loader.py
"""Reads AgentIdeaBench's published per-critic scores and aggregates them the
same way we aggregate our own hypotheses, so a comparison is apples-to-apples.
"""
import pandas as pd

from benchmark.scorer import aggregate_dimension_scores
from benchmark.vendor_bootstrap import SCORING_DIMS
from config.benchmark import VENDOR_ROOT

_SCORES_PATH = VENDOR_ROOT / "release_data" / "core" / "lit8d_scores_3seed.csv.gz"
_DIM_COLUMNS = {dim: f"score_{dim}" for dim in SCORING_DIMS}


def _load_scores_dataframe() -> pd.DataFrame:
    return pd.read_csv(_SCORES_PATH)


def load_published_score(idea_model: str, track: str, subdomain: str) -> float | None:
    df = _load_scores_dataframe()
    rows = df[
        (df["idea_model"] == idea_model)
        & (df["track"] == track)
        & (df["subdomain"] == subdomain)
    ]
    if rows.empty:
        return None

    per_critic_scores = {}
    for _, row in rows.iterrows():
        per_critic_scores[row["critic_model"]] = {
            dim: row[col] for dim, col in _DIM_COLUMNS.items()
        }

    _per_dim, weighted_total = aggregate_dimension_scores(per_critic_scores)
    return weighted_total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmark/test_baseline_loader.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/baseline_loader.py tests/benchmark/test_baseline_loader.py
git commit -m "Add baseline loader for AgentIdeaBench's published leaderboard scores"
```

---

## Task 8: Manual parity check script (not part of the automated test suite)

**Files:**
- Create: `scripts/benchmark_parity_check.py`

**Interfaces:**
- Consumes: `benchmark.scorer.score_hypothesis`, `benchmark.baseline_loader.load_published_score`,
  `benchmark.models.Hypothesis`; reads `release_data/core/subdomain_ideas.csv.gz`
  for a real released hypothesis to re-score.

This task makes real network calls (OpenRouter + Semantic Scholar) and costs
real money — it is a manual verification script, run by a human, not part of
`pytest`. Its only purpose: confirm our wrapper's score for a hypothesis
AgentIdeaBench already scored lands close to their published number, proving
the wrapper is wired correctly end-to-end.

- [ ] **Step 1: Write the script**

```python
# scripts/benchmark_parity_check.py
"""
Manual parity check: pick one already-released hypothesis, re-score it
through our wrapper, compare against AgentIdeaBench's own published score.

Run: python scripts/benchmark_parity_check.py

Requires vendor/AgentIdeaBench/.env to have real OPENROUTER_API_KEY and
SEMANTIC_SCHOLAR_API_KEY values (see Task 1, Step 2). Costs a small number of
real OpenRouter API calls (3 critics x 1 hypothesis).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from benchmark.baseline_loader import load_published_score
from benchmark.models import Hypothesis
from benchmark.scorer import score_hypothesis
from config.benchmark import VENDOR_ROOT

IDEAS_PATH = VENDOR_ROOT / "release_data" / "core" / "subdomain_ideas.csv.gz"


def main():
    ideas = pd.read_csv(IDEAS_PATH)
    # Pick the first Active-track (track "C") row with non-empty text.
    row = ideas[(ideas["track"] == "C") & (ideas["idea_text"].str.strip() != "")].iloc[0]

    published = load_published_score(row["idea_model"], row["track"], row["subdomain"])
    if published is None:
        print(f"No published score found for {row['idea_model']} / {row['subdomain']}")
        return

    hyp = Hypothesis(
        topic=row["subdomain"],
        domain=row["domain"],
        subfield=row["subdomain"],
        text=row["idea_text"],
        source="parity_check",
    )
    result = score_hypothesis(hyp)

    print(f"Subfield: {row['subdomain']} ({row['domain']})")
    print(f"Original model: {row['idea_model']}, track {row['track']}")
    print(f"Published weighted score: {published:.2f}")
    print(f"Our wrapper's weighted score: {result.weighted_total:.2f}")
    print(f"Difference: {abs(published - result.weighted_total):.2f}")
    print("\nPer-dimension (published critics may differ from ours by random selection):")
    for dim, value in result.per_dimension_trimmed.items():
        print(f"  {dim:14s}: {value:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it manually and inspect the output**

Run: `python scripts/benchmark_parity_check.py`

Expected: prints a published score and our wrapper's score for the same
hypothesis, within a few points of each other (exact match isn't expected —
critics are stochastic and our evidence-retrieval query extraction may pull
slightly different prior-art hits than the original run did — but a wildly
different number, e.g. off by more than ~2 points on the 1-10 scale, signals a
wiring bug worth investigating before moving on to the orchestrator).

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_parity_check.py
git commit -m "Add manual parity check script for the benchmark harness"
```

---

## Explicitly deferred (not silently dropped)

Two pieces of spec §4 are not covered by this plan:

- **Compute-matched best-of-N baseline** — needs something that generates
  independent single-shot hypotheses to compare against. That's the
  orchestrator's job (Plan 2). Once Plan 2 exists, this baseline is a thin
  addition: call the orchestrator's single-shot generation function N times
  with no debate/evolution, score each with `score_hypothesis`, keep the best.
- **Recall-only baseline lookup** — `load_published_score` (Task 7) only reads
  `release_data/core/lit8d_scores_3seed.csv.gz`, which covers Static (track
  `B`) and Active (track `C`) only. Recall-only scores live in a separate file
  (`release_data/appendix/e32_recall_scores.csv.gz`) with a different schema
  (no `track` column — it's its own condition, not `B`/`C`). Add a second
  loader function for it if/when a comparison against Recall-only is needed;
  not required for the harness's core job of scoring our own hypotheses.

## Definition of done

All 8 tasks complete, `pytest tests/benchmark/ -v` passes fully (mocked, no
network), and `scripts/benchmark_parity_check.py` has been run at least once
manually with real API keys, producing a score within a few points of
AgentIdeaBench's published number for the same hypothesis. At that point the
harness is ready to score output from the orchestrator (Plan 2, written
separately once this one is implemented).
