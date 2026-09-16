# Multi-agent research ideation system — design

**Date:** 2026-09-16
**Status:** approved for implementation planning
**Related research:** [`docs/research/`](../../research/) (benchmarks.md, existing-systems.md,
literature-review.md, design-recommendations.md)

## 1. Goal and scope

Build a system with two cooperating parts:

1. **A benchmark harness** that scores any generated research hypothesis using the
   real, published AgentIdeaBench scoring pipeline, so results are directly
   comparable to their public leaderboard.
2. **An orchestrator**: a multi-agent swarm that takes a research topic and
   produces either one hypothesis (for benchmark testing) or the top-K diverse
   hypotheses (for real use), by running rounds of generation, debate-based
   ranking, and evolution.

**In scope:** the ideation phase only — researching a topic, generating candidate
ideas, critiquing and ranking them, evolving the strongest ones, returning a
final ranked/diverse set of hypotheses.

**Out of scope:** running real experiments, executing code, writing full papers,
anything past producing a hypothesis. (This mirrors what Agent Laboratory,
AI-Researcher, and AI co-scientist all do in addition to ideation — we
deliberately stop before that.)

## 2. Why this design (brief; full detail in `docs/research/`)

- **Benchmark choice:** AgentIdeaBench, because it's the only existing benchmark
  built specifically to isolate the value of agentic literature retrieval
  (Static vs. Active), has the strongest validation of the four surveyed
  (Fleiss κ up to 0.97, human agreement up to 93% on clear-cut pairs), and
  ships a fully reusable scoring pipeline.
- **Architecture:** closest to Google's AI co-scientist (Generation → Reflection
  → Ranking via debate+Elo → Evolution → Proximity → Meta-review), because it's
  the only surveyed system whose scope is essentially generate-critique-rank-evolve
  already. We improve on it and on the more recently found EvoScientist
  (arXiv:2603.08127) in specific, verified ways — see §9.

## 3. Architecture overview

Two subsystems in one repo:

```
benchmark/      — wraps vendor/AgentIdeaBench's real scoring pipeline.
                  Always OpenRouter, always the 3 fixed critics
                  (qwen/qwen3.6-plus, moonshotai/kimi-k2.6, z-ai/glm-5.1),
                  always the real LIT8D_SYSTEM rubric.

orchestrator/   — the multi-agent swarm. Pluggable backend
                  (OpenRouter or the user's Claude Code plan).
                  Produces one or many final hypotheses per topic.
```

Data flow:

```
topic (an AgentIdeaBench subfield, or custom)
  → orchestrator: N rounds of generate / reflect / rank / evolve
  → final pool, read out as either 1 hypothesis (benchmark mode)
    or top-K diverse hypotheses (real-use mode)
  → benchmark/: score each hypothesis via the reused LIT8D_SYSTEM pipeline
  → compare against AgentIdeaBench's published Static / Active / Recall-only
    rows for that same subfield, and against our own compute-matched baseline
```

## 4. Benchmark harness (`benchmark/`)

All pieces below are **imported directly from `vendor/AgentIdeaBench`**, not
reimplemented — this is the "completely reuse the benchmark" requirement.

| Component | Reuses | Purpose |
|---|---|---|
| Reference fetcher | Semantic Scholar API, using AgentIdeaBench's released `ss_paper_id` values (`release_data/derived/static_refs.csv.gz`) | AgentIdeaBench doesn't ship abstract text (license reasons) — refetch it ourselves under our own API key |
| Evidence builder | `build_evidence_for_item`, `format_evidence_block` (`experiments/e13_litverify_rubric8.py`) | Per-hypothesis Semantic Scholar prior-art retrieval, date-filtered, self-excluded — the real mechanism behind AgentIdeaBench's grounded Originality score |
| Scorer | `score_one`, `LIT8D_SYSTEM`, `trimmed_mean` (`experiments/e10_idea_anchor_calibration.py`, `experiments/e13_litverify_rubric8.py`), `SCORING_DIMS`/`SCORING_WEIGHTS` (`utils/constants.py`) | Runs the exact production rubric (not the simpler `evaluation/absolute_scorer.py` default) against the 3 fixed critics, aggregates with drop-highest-of-three + weighted mean |
| Baseline loader | `release_data/core/lit8d_scores_3seed.csv.gz`, `LEADERBOARD.md` | Published Static / Active / Recall-only scores per subfield, for comparison — no re-running required |
| Compute-matched baseline | our own addition | Best-of-N independent single-shot generations at the same total call budget as a full swarm run, scored the same way — isolates whether the swarm's architecture (not just extra compute) is responsible for any gain |

**Critical detail, confirmed by reading the vendored source directly:** the
rubric that produced the published leaderboard is `LIT8D_SYSTEM`
(`experiments/e13_litverify_rubric8.py`), a five-block-stacked prompt, **not**
the default `SYSTEM_PROMPT` in `evaluation/absolute_scorer.py`. Using the wrong
one silently produces scores that look plausible but aren't actually comparable
to the published numbers. The harness must import from `experiments/`, not just
`evaluation/`.

## 5. Orchestrator (`orchestrator/`)

### 5.1 Model backend abstraction

```python
class ModelBackend(Protocol):
    def call(self, role: str, system_prompt: str, user_prompt: str) -> str: ...
```

- `OpenRouterBackend` — HTTP calls to OpenRouter, model configurable per role.
- `ClaudeCodeBackend` — shells out to the `claude` CLI in headless mode
  (`claude -p "<prompt>" --output-format json`), one subprocess per call, so
  swarm runs consume the user's Claude Code plan instead of metered tokens.

All orchestrator logic is written once against this interface. The backend
choice is a single config flag, per-run (and could be per-role later, but not
in v1). This choice **only applies to the swarm's own internal agents** — the
benchmark scoring critics in §4 always use OpenRouter with the 3 fixed models,
regardless of this setting.

### 5.2 Components

- **Supervisor** — holds run config, drives the round loop over an async task
  queue.
- **Generation** — divergent(N)→convergent(k) per round: produces several
  conceptually distinct candidates (not variations on one theme), each
  retrieval-grounded, each checked against the memory store first (§5.4) to
  avoid re-proposing something already tried and to build on what worked.
- **Reflection** — cheap per-candidate critique gate (coherence, minimal
  feasibility, concreteness) before a candidate enters the ranked pool. Not
  the final quality signal — just a filter against obviously broken output.
- **Ranking** — pairwise debate + Elo, not absolute scoring. Two ideas from
  the pool are judged head-to-head; the winner's Elo rises, the loser's falls.
  Pairs are chosen by Proximity (below) so debates are informative (comparable
  strength) rather than wasted (obviously-better vs. obviously-worse).
- **Evolution** — five named strategies applied to the top of the ranked pool
  each round: combine, simplify, ground-further-in-literature, analogical
  reasoning, and **deliberate divergence** — the last one is mandatory every
  round, not conditional, since nothing else in the system reliably produces
  novelty (confirmed independently by AgentIdeaBench's finding that retrieval
  improves feasibility/clarity but not originality, and by the "Diversity
  Collapse in Multi-Agent LLM Systems" paper).
- **Proximity/dedup** — background embedding-similarity service over the
  current pool. Three jobs: (1) pick informative debate pairs, (2) prevent
  wasted debates between near-duplicate ideas, (3) power the diversity-aware
  final selection (§5.3) and memory-store deduplication (§5.4).
- **Meta-review** — end-of-round synthesis of recurring critique patterns,
  fed into next round's Generation as extra context. This is the actual
  round-over-round learning mechanism, distinct from cross-run memory.

### 5.3 Final selection (single output vs. top-K)

The pool is never collapsed to one lineage — it's ranked via Elo throughout.
Reading results out at the end is a parameter, not a different code path:

- **K=1 (benchmark mode):** take the single highest-Elo idea.
- **K>1 (real use):** walk down the Elo-ranked pool; add each idea to the
  result set only if it isn't near-duplicate (via Proximity's similarity
  metric) to anything already selected; stop at K or when the pool runs out.

Pool size (default 20–30) and the mandatory-divergence Evolution strategy are
load-bearing for this — a pool that's collapsed to 5 similar ideas by the last
round has nothing distinct left to return for K=10.

### 5.4 Persistent memory

- Keyed by (domain, subfield). Stores past final hypotheses plus their scores
  (sourced from Ranking's Elo/debate outcome, never from agent self-report —
  this specifically avoids the reward-hacking failure mode AgentRxiv
  documented).
- Checked by Generation before producing new candidates for a topic.
- Deduplicated on write using the same Proximity similarity mechanism as the
  live pool, so the memory store doesn't degrade into near-duplicate clutter
  over time (a gap found in EvoScientist's memory design, which has no
  pruning/curation at all).

## 6. Storage

- **Run artifacts** (SQLite, mirroring the vendored repo's own pattern for
  idempotent/resumable runs): per (topic, run) — every round's candidate pool,
  Elo history, evolution operations applied, final selection, benchmark
  scores.
- **Memory store**: separate table/collection, embedding-searchable, as in
  §5.4.

## 7. Error handling

- Structured-output agent calls (debate verdicts, evolution outputs) reuse
  AgentIdeaBench's own multi-layer JSON-parse-with-retry pattern.
- Backend failures (API timeout, CLI subprocess failure) retry with capped
  backoff; failures are logged and marked, never crash the whole run.
- Hard cap on total model calls per run, reported explicitly as a "compute
  multiplier" relative to AgentIdeaBench's 10-call Active budget, so any
  reported win stays honest about how much extra compute the swarm used.

## 8. Testing plan

1. **Harness parity check** — score one of AgentIdeaBench's own released
   hypotheses through our wrapper; confirm it lands close to their published
   score for that row.
2. **Orchestrator smoke test, OpenRouter backend** — one topic, small fixed
   round budget, confirm a valid hypothesis is produced and scores.
3. **Orchestrator smoke test, Claude Code backend** — same, backend swapped.
4. **Baseline ladder**, on a pilot topic set: bare LLM / LLM+RAG / single-round
   swarm / full swarm (debate+evolution+memory), compared against
   AgentIdeaBench's published Static/Active/Recall-only numbers for those
   subfields, plus our own compute-matched best-of-N baseline.

## 9. Gaps this design addresses, confirmed against existing systems

| Gap | Who has it | Our fix |
|---|---|---|
| Novelty is ungrounded LLM opinion | AI-Researcher, EvoScientist | Reuse AgentIdeaBench's real evidence-retrieval pipeline |
| Ranking is absolute LLM scoring, not comparison | (measured failure mode: Agent Laboratory's 6.1 vs. 3.8 human/LLM gap) | Pairwise debate + Elo throughout |
| Elo tournament has no debate/critique exchange | EvoScientist | Keep debate-style argumentation (as in Google's co-scientist) |
| No diversity safeguard; pool can collapse | EvoScientist; validated independently by "Diversity Collapse in Multi-Agent LLM Systems" (ACL 2026 Findings) | Proximity/dedup + mandatory deliberate-divergence evolution every round |
| Memory grows unbounded, no curation | EvoScientist | Dedup memory writes via the same Proximity mechanism |
| Single-pass per task; no intra-task iteration | EvoScientist (evolves only across tasks via memory, not within one) | Multiple rounds of debate+evolve within a single topic, not just cross-run memory |
| Hard-coded to specific model backends | EvoScientist (Gemini+Claude+Ollama, no abstraction) | `ModelBackend` interface, pluggable OpenRouter/Claude Code |
| Evaluated only on self-constructed queries | EvoScientist (30 author-curated queries), AI-Researcher (own Scientist-Bench) | Benchmarked against AgentIdeaBench's published, human-validated data |
| Always outputs a single winner | Every surveyed system | Diversity-aware top-K final selection |

## 10. v1 scope (explicit cuts)

- Fixed round budget (e.g. 3), not dynamic stopping.
- Small pilot topic set (5–10 of AgentIdeaBench's 100 subfields), not the
  full set — controls OpenRouter cost while validating.
- Simple memory store (single table + embedding similarity), no clustering or
  advanced consolidation beyond the dedup-on-write rule in §5.4.
- Per-run backend choice only (OpenRouter or Claude Code), not per-role mixing.

## 11. Open risks / things to watch

- **Claude Code plan usage limits** — a single swarm run involves dozens of
  agent calls; running many pilot topics in Claude-Code-backend mode could
  consume plan quota faster than expected.
- **Domain reliability is uneven** — AgentIdeaBench's own critic discrimination
  is strong in CS/physics, weaker in biology/chemistry/medicine. Any
  cross-domain comparison should report this rather than presenting one
  blended number.
- **LLM-judge-vs-human gap** — confirmed repeatedly across every system
  surveyed (Agent Laboratory: 2.3-point gap; AI-Researcher: 13.6%–81.8% swing
  depending on judge; Ideation Arena: best automated judge only 72.56% aligned
  with human preference). Periodic small-sample human spot-checks are still
  warranted even though day-to-day iteration relies on the automated critics.
- **EvoScientist's code availability** was only surfaced via search snippets
  during research, not independently verified as public/inspectable — if we
  ever want to directly compare against it, that needs a direct check first.
