# Handoff — orchestrator / benchmark-harness

Written 2026-09-24 at the end of a long session. Read this before doing anything else.

## What this project is

`orchestrator/` is a standalone multi-agent ideation swarm (spec:
`docs/superpowers/specs/2026-09-16-multiagent-ideation-system-design.md`,
plan: `docs/superpowers/plans/2026-09-16-orchestrator.md`). It generates
research hypotheses on a topic, has them debate each other (pairwise + Elo,
not absolute scoring), evolves the best ones, and selects a final
diversity-aware top-k. It's deliberately independent of `benchmark-harness/`
(a sibling package, built by a different session, that wraps AgentIdeaBench's
real critic-scoring pipeline) — the two talk to each other only through a
JSON data contract (an orchestrator run's `--export` file), never through
Python imports.

**Current pipeline**: Generate → Debate (Elo) → pool-trim → Evolve →
Meta-review, looped for N rounds. There is **no Reflection stage** — it was
removed (commit `3267547`) after real data showed it never filtered a single
candidate across 1287 real generations.

**Idea format**: every idea is 8 structured fields (title, problem,
hypothesis, mechanism, method, prediction, falsification, novelty_delta —
see `orchestrator/src/orchestrator/idea_schema.py`), not free text. This was
a deliberate fix for ideas reading as "a whole bunch of nothing."

## Where things stand right now

A long autonomous optimization loop (user-requested) tried 4 candidate
changes, validated each on a fixed 4-subfield subset, kept/reverted based on
real data, then confirmed the winners at full scale (39 real AgentIdeaBench
subfields). **Final validated result**: our swarm's final-3 picks score
**7.56 avg** vs. published Claude Sonnet 5's **6.65 avg** (Active track) on
the *same* real subfields — **37/39 wins**, paired t=10.19 (extremely
significant), at **38% lower cost per subfield** than before the
optimization pass (56 vs ~90 model calls). Full history:

| Step | Change | Verdict | Commit |
|---|---|---|---|
| Exp1 | Removed reflection gate; replaced `simplify` (0% win rate) with a 2nd `combine` pass (34% win rate) | kept | `3267547`, `df9e322` |
| Exp2 | `debate_pairs_per_round` 8→12 | kept | `7ec4bb1` |
| Exp3 | Literature search limit 5→8 | **reverted** (net -0.356 on validation set) | — |
| Exp4 | `pool_size` 4→6 (in the sweep script, not the CLI default) | kept | `c3bdeb7` |

All of this is committed on `main` (not pushed to `origin` — check with the
user before pushing). The only uncommitted file is `.env.example`, which is
**hard-blocked** by this session's permission settings (Read/Write/Edit and
even Bash commands mentioning that path get denied) — if the user wants
orchestrator-specific env vars documented there, they need to do it
themselves. The suggested block to add:
```
# orchestrator/ only — optional, all have working defaults, uncomment to override.
# CLAUDE_CLI_PATH=claude
# ORCHESTRATOR_GENERATION_MODEL=anthropic/claude-sonnet-5
# ORCHESTRATOR_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Key files

- `orchestrator/src/orchestrator/` — the swarm itself. `supervisor.py` is
  the round loop; `models.py` has `RunConfig` (all the tunable knobs);
  `idea_schema.py` defines the 8-field structure; `ranking.py` +
  `proximity.py` do debate pairing (adjacent-Elo by default, or
  `round_robin_debates=True` for every-pair-once, which is quadratic and
  expensive — see the earlier token-cost discussion in conversation history
  if resuming that thread).
- `orchestrator/src/orchestrator/store.py` — SQLite persistence
  (`.data/runs.db`, gitignored). Tracks **every** candidate ever produced,
  including pruned ones, with a `status` field
  (`active`/`pruned_pool_trim`/`selected_final`/`not_selected_final`) and
  per-round Elo snapshots. Has a `_migrate()` method that auto-adds missing
  columns to an existing DB file — needed once already when a column was
  added after `.data/runs.db` already existed.
- `orchestrator/src/orchestrator/report.py` — assembles a full run's
  history into one JSON-serializable dict for export/dashboarding.
- `orchestrator/scripts/build_dashboard.py` — turns a run's `--export` JSON
  into a self-contained HTML dashboard (Elo chart, round-by-round
  candidate/debate log, filterable browser of every idea). Two are
  published as Claude Artifacts already (links were shared with the user
  earlier in conversation, not repeated here — ask the user or check
  `Artifact action:"list"` if you need them).
- `orchestrator/scripts/single_call_baseline.py` — baseline #1 from the
  design doc's 4-tier ladder: one raw LLM call, no swarm at all.
- `benchmark-harness/scripts/score_orchestrator_run.py` — scores an
  orchestrator run's ideas with AgentIdeaBench's real critic pipeline
  (3 critics, drop-highest, weighted dims). Supports `--which final|all`,
  `--offset`/`--limit` (for batching a big scoring job), `--out` (appends
  results as JSON keyed by candidate_id).
- `scripts/run_subfield_sweep.py` (repo root, **not** inside either
  package) — the cross-package driver used for all the real-subfield
  validation runs. Coordinates both venvs by path via `subprocess`; imports
  neither package itself. Key flags: `--start`/`--count` (slice of the
  real 40 subfields) or `--subfields "name1,name2"` (exact ones), `--tag`
  (namespaces output under `orchestrator/runs/<tag>/`), `--pool-size`
  (default 6, passed through to the orchestrator CLI). Idempotent/resumable
  — re-running the same tag skips subfields already marked `"done"` in
  that tag's `summary.json` and retries failed ones.
- `orchestrator/runs/all_40_subfields.json` — the real AgentIdeaBench
  subfield list (40 rows, `domain`/`subdomain`), pulled from
  `benchmark-harness/vendor/AgentIdeaBench/release_data/core/lit8d_scores_3seed.csv.gz`.
  Domain strings are `"CS"`/`"Biology"`/`"Chemistry"`/`"Medicine"`/`"Physics"`
  (not `"Computer Science"` — that mismatch bit an earlier run).
- `orchestrator/runs/sweep2/` — the final validated 39-subfield result
  (post-optimization). `orchestrator/runs/sweep/` — the original
  pre-optimization 39-subfield result (V0 baseline, 7.36 avg). Both have a
  `summary.json` with per-subfield `our_final3_avg`,
  `published_static_sonnet5`, `published_active_sonnet5`,
  `total_model_calls`, `status`.

## Two separate venvs — don't mix them up

- `orchestrator/.venv` — has `requests`, `numpy`, `sentence-transformers`.
  Run orchestrator code with `cd orchestrator && source .venv/bin/activate`,
  or `PYTHONPATH=src:.` if invoking `python -m orchestrator.cli` from
  outside pytest (pytest's `pythonpath` setting isn't picked up by a bare
  `python` invocation).
- `benchmark-harness/.venv` — has `openai`, `pandas`, `python-dotenv`. Same
  `PYTHONPATH=src:.` caveat applies for its scripts.
- `scripts/run_subfield_sweep.py` calls both venvs' Python binaries
  directly by path (`ORCH_PY`, `BENCH_PY` constants) via `subprocess`, so
  you don't need to activate anything to use it — just
  `python3 scripts/run_subfield_sweep.py ...` from the repo root with
  whatever system Python.

## Secrets

`OPENROUTER_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` are present and valid in
the repo-root `.env` (confirmed present/non-empty, never read or printed —
that file is permission-blocked for direct Read/Write/Edit/Bash-mention in
this session). `benchmark-harness/project_config/benchmark.py` loads it
automatically via `load_dotenv` the moment anything imports
`benchmark.vendor_bootstrap` — this is normal, intended app behavior, not a
workaround. OpenRouter scoring is real billed money, ~$0.02/idea scored —
trivial per-call but be deliberate about `--which all` on a large batch.
Claude Code generation (the `claude_code` backend) is free but consumes the
account's 5-hour usage window — this has been hit at least once, causing
`Claude CLI exited 1` failures with **empty stderr**. That's the signature
to recognize; the sweep script auto-retries once, and a second full pass
usually clears it once the window resets.

## Known open items (not yet done)

1. **`antiviral resistance infectious disease mechanism`** has failed
   generation **3 times** across two separate full sweeps (both the
   original `sweep` and `sweep2`) — same subfield both times. This looks
   less like random usage-limit noise and more like something specific to
   that subfield's content or prompt shape. Worth a manual single-subfield
   run with stderr actually captured (the sweep script truncates it) to
   root-cause before assuming it's just bad luck.
2. **Compute-matched baseline ablation not run.** The single highest-value
   next experiment for publication rigor: generate N independent
   single-calls (same N as the swarm's total idea count) and compare their
   best-of-N against the swarm's best-of-N, to isolate "is the architecture
   smarter" from "did it just get more tries." `orchestrator/scripts/single_call_baseline.py`
   exists for generating one baseline idea; would need a small loop wrapper
   to do it N times and take the max.
3. **`pool_size` default mismatch**: the sweep script's default is now 6
   (validated), but `orchestrator/src/orchestrator/cli.py`'s own
   `--pool-size` default and `RunConfig.pool_size`'s default are both still
   24. Never resolved whether real (non-benchmark) runs should adopt 6 too
   — flag this to the user rather than silently changing it.
4. **No human validation** of the critic scores — this whole result is LLM
   critics judging LLM-generated ideas. Every design note in this repo
   (`docs/research/design-recommendations.md`) flags this as a real risk,
   not a formality.
5. **Structured-schema-as-rubric-confound** untested — the 8-field format
   may score well partly because it directly maps onto what the rubric
   checks for (specificity, falsifiability), independent of actual idea
   quality. No ablation isolating this yet.
6. **Not pushed to `origin`** — all commits are local to this machine's
   `main` branch. Confirm with the user before pushing.
7. `.env.example` update still pending (see above — permission-blocked,
   needs the user to do it manually).

## If you're picking this up to write the paper

The result is genuinely solid enough for a workshop paper now (see
conversation history for the full reasoning): real n=39 statistical
significance, a real ablation (items 1-4 in the experiment table above)
showing which components matter, and the gains came *with* a cost
reduction, which pre-empts the "it's just more compute" objection. Item #2
above (compute-matched baseline) is still the one gap worth closing before
calling it done — it's the first thing a reviewer would ask for.
