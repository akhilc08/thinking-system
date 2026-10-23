# thinking-system

A multi-agent swarm for the ideation phase of research. You give it a topic. It searches the literature, generates candidate hypotheses, makes them debate each other, evolves the strongest ones, and returns a diversity-aware top-k. It does not run experiments or write papers. The output is a short list of concrete, falsifiable research ideas.

The repo also includes a benchmark harness that scores the swarm's ideas with the critic pipeline from [AgentIdeaBench](https://github.com/HKUST-KnowComp/AgentIdeaBench). That makes it possible to compare against published model scores on the same subfields.

## How it works

Each run loops for N rounds:

1. **Generate**: new hypotheses, grounded in a Semantic Scholar search on the topic.
2. **Debate**: an LLM judge compares ideas pairwise and updates each idea's Elo rating. There is no absolute scoring. By default, ideas with nearby Elo get paired.
3. **Trim**: only the top `pool_size` survivors are kept.
4. **Evolve**: the leaders get refined and combined into new candidates.
5. **Meta-review**: notes from the round's debates feed into the next round.

At the end it picks the top-k by Elo and drops near-duplicates using sentence embeddings.

Every idea uses the same 8 fields: title, problem, hypothesis, mechanism, method, prediction, falsification, novelty delta (see `orchestrator/src/orchestrator/idea_schema.py`). Free-text ideas tended to sound plausible while saying very little. The fixed fields force each one to commit to something testable.

Every candidate goes into SQLite (`orchestrator/.data/runs.db`), pruned ones included, along with per-round Elo snapshots and debate logs. A run can be exported as JSON and turned into a standalone HTML dashboard.

## Results

The swarm ran on 39 of AgentIdeaBench's 40 subfields, across CS, biology, chemistry, medicine and physics. One subfield failed generation repeatedly. Each subfield's final 3 ideas were scored with AgentIdeaBench's critic rubric (3 critics, highest score dropped, weighted dimensions).

| | Avg score |
|---|---|
| This swarm (final-3 avg, Claude Sonnet 5 backend) | **7.56** |
| Claude Sonnet 5, Active track (published) | 6.65 |
| Claude Sonnet 5, Static track (published) | 6.04 |

The swarm beat the published Active-track score on 37 of 39 subfields (paired t = 10.19), averaging about 56 model calls per subfield. Per-subfield numbers are in `orchestrator/runs/sweep2/summary.json`.

Caveats:

- LLM critics scored LLM-generated ideas. Nobody has checked these scores against human judgment.
- No compute-matched baseline has been run yet: best-of-N single calls with the same N. Part of the gain could come from more attempts rather than a better architecture.
- The 8-field format lines up closely with what the rubric rewards (specificity, falsifiability). Some of the gain may come from the format and not from better ideas. No ablation tests this yet.

## Repo layout

```
orchestrator/          the swarm (standalone package)
  src/orchestrator/    supervisor loop, generation, debate/Elo, evolution, store
  scripts/             dashboard builder, single-call baseline, smoke tests
  runs/                subfield list + validated sweep results
benchmark-harness/     scoring wrapper around AgentIdeaBench's critic pipeline
  vendor/AgentIdeaBench/  vendored upstream copy (MIT), see VENDORED.md
scripts/
  run_subfield_sweep.py   runs + scores a batch of benchmark subfields end to end
docs/
  research/            background survey: benchmarks, existing systems, design notes
  design/              design spec and implementation plans
```

`orchestrator` and `benchmark-harness` never import each other. The only interface between them is the JSON file an orchestrator run writes with `--export`.

## Setup

Each package has its own virtualenv. The orchestrator needs Python 3.9+ and the harness needs 3.10+.

```bash
cd orchestrator
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

cd ../benchmark-harness
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Copy `.env.example` to `.env` at the repo root and fill in:

- `OPENROUTER_API_KEY`: used for the `openrouter` generation backend and for scoring.
- `SEMANTIC_SCHOLAR_API_KEY`: used for literature search.

Optional overrides: `CLAUDE_CLI_PATH` (default `claude`), `ORCHESTRATOR_GENERATION_MODEL` (default `anthropic/claude-sonnet-5`), `ORCHESTRATOR_EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`).

There are two generation backends:

- `openrouter` calls models through the OpenRouter API, billed per token.
- `claude_code` shells out to the Claude Code CLI and draws on your Claude subscription's usage window. If you hit the limit, you'll see `Claude CLI exited 1` with an empty stderr.

## Usage

Run the swarm on a topic:

```bash
cd orchestrator
PYTHONPATH=src:. .venv/bin/python -m orchestrator.cli \
  --topic "protein language models for enzyme design" \
  --domain Biology --subfield "protein engineering" \
  --backend claude_code --rounds 3 --k 3 --pool-size 6 \
  --export runs/my_run.json
```

Useful flags: `--brief` adds constraints like a deadline or budget, which the generator and judge see but literature search doesn't. `--max-calls` is a hard budget cap. `--round-robin` debates every pair each round instead of only adjacent-Elo pairs, which costs quadratically more.

Build a dashboard from an export:

```bash
.venv/bin/python scripts/build_dashboard.py runs/my_run.json runs/my_run.html
```

Score a run's final ideas with the AgentIdeaBench critics (billed, about $0.02 per idea):

```bash
cd ../benchmark-harness
PYTHONPATH=src:. .venv/bin/python scripts/score_orchestrator_run.py \
  ../orchestrator/runs/my_run.json --which final --out ../orchestrator/runs/my_scores.json
```

Reproduce the benchmark sweep from the repo root. It's resumable, so re-running a tag skips subfields that already finished:

```bash
python3 scripts/run_subfield_sweep.py --start 0 --count 40 --tag sweep2
```

## Tests

```bash
cd orchestrator && .venv/bin/python -m pytest
cd benchmark-harness && .venv/bin/python -m pytest
```

## License

`benchmark-harness/vendor/AgentIdeaBench` is a copy of AgentIdeaBench, used under its MIT license (`LICENSE` and `LICENSE-DATA` in that directory).
