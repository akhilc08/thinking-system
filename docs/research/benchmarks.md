# Benchmarks for evaluating research-idea generation

Four benchmarks, all with public code/data as of Sept 2026. Ordered from
least to most relevant to a multi-agent, retrieval-using ideation swarm.

## AI Idea Bench 2025 (AIIB)

**What it measures:** whether a generated idea converges toward what a real
subsequent paper actually did — not abstract creativity.

- **Dataset**: 3,495 target papers from ICLR/CVPR/ECCV/NeurIPS/ICML/NAACL
  /EMNLP/ACL 2024, all published after Oct 2023 specifically to dodge LLM
  training-data leakage. Each paired with its 5 most-cited "inspiring" prior
  works, standing in for the citation context a real researcher would've had.
- **Tasks**: I2T (idea→topic matching), **I2I** (idea→idea — the main task:
  generate an idea from the prior-work context, score 0–5 similarity to what
  the real paper actually did), IMCQ (cheap multiple-choice proxy).
- **Scoring**: Quality (similarity to real outcome), Novelty (distance from
  literature, citation-weighted), Feasibility (grounded in cited-work
  influence). Hybrid ground-truth + LLM-judge, not pure LLM-judge.
- **Fit**: good if the system's job is "given a slice of literature, converge
  on directions real researchers independently found valuable." Bad fit for
  divergent/exploratory ideation — the metric structurally rewards
  mainstream-adjacent convergence over genuinely novel gaps.
- **Code**: `github.com/yansheng-qiu/AI_Idea_Bench_2025`, dataset on
  Hugging Face.

## LiveIdeaBench

**What it measures:** raw divergent-thinking ability, deliberately stripped
of context — input is a single keyword, output capped at 200 words.

- **Dataset**: 1,180 keywords from live analytics data (snapshot Dec 2024),
  across 18 domains, refreshed monthly — a genuinely non-static benchmark.
- **Scoring**: no ground truth at all. The top 10 of 20 evaluated models
  double as judges (excluded from judging their own output), ≥3 judgments
  per idea, scored on Originality / Feasibility / Fluency / Flexibility
  (borrowed from Guilford's psychometric divergent-thinking model).
- **Key finding**: creativity and general intelligence are only weakly
  correlated (r=0.409, not significant) — a weak general-purpose model
  (qwq-32b-preview) placed 3rd on creativity. Domain specialization emerged
  (Gemini→chemistry, Claude→medicine, o1-preview→physics). Safety refusals on
  certain keywords directly tanked scores for those models — a confound to
  watch for if evaluating safety-tuned models.
- **Fit**: cheap sanity check that individual agent "roles" aren't collapsing
  into homogeneous, safe outputs. Not useful for evaluating literature-
  grounded pipeline output — it deliberately removes context.
- **Code**: `x66ccff/liveideabench`, live leaderboard at liveideabench.com,
  CC BY 4.0.

## AgentIdeaBench — closest match to a research-and-propose swarm

Purpose-built to isolate the value of *agentic literature retrieval* —
exactly the "spawn agents to research first, then propose" pattern.

- **Design**: matched **Static** (given curated papers) vs. **Active**
  (given only a subfield name + a Semantic Scholar tool, 10-call budget, must
  retrieve its own evidence) conditions, identical scoring rubric across
  both — isolating "who chose the literature" as the only variable.
- **Dataset**: 100 narrow subfields across CS / physics / biology /
  chemistry / medicine, target papers postdating training cutoffs.
- **Notable finding**: models left to search on their own retrieve very
  different literature than a curator would — median pub year 2023 vs. 2020,
  median citations 16 vs. 2,207. Unguided retrieval agents will gravitate to
  recent, low-citation papers instead of canonical ones — worth correcting
  for explicitly (e.g. weight retrieval toward highly-cited work) rather than
  leaving to chance.
- **Scoring rigor** (best-validated of the four): 3 open-weight critics,
  trimmed mean, Originality-weighted (2:1.5:1:0.5:0.5 across
  Originality/Feasibility/Clarity/Impact/Specificity). Fleiss κ=0.97 (binary)
  / 0.81 (ternary) inter-critic agreement; human-critic agreement 77%
  (κ=0.55) overall, 96% (κ=0.93) on clear-cut cases. Sanity check: award-
  winning papers score 7.71, above every evaluated model's Active mean (best
  model: 6.33) — rubric isn't saturated.
- **Headline result — design around this**: Active mode **quadruples score
  variance** vs. Static (re-separating a field that looks artificially
  compressed under static context). Whether retrieval helps is
  capability-gated (r=+0.69 between base model capability and Active-mode
  gain — strong models gain, weak models are actively *hurt* by search
  tools). Decomposed by dimension: retrieval boosts Feasibility (+1.21),
  Clarity (+0.63), Specificity (+0.58) — **Originality barely moves
  (−0.14, not significant)**.
  > Literature retrieval makes ideas more grounded and feasible.
  > It does **not**, by itself, make ideas more novel.
  If novelty is a target metric, retrieval alone won't deliver it — a
  separate mechanism (adversarial critique, forced divergence, deliberate
  "go against the pack" step) is required.
- **Stated limitation**: authors explicitly haven't run a real domain-expert
  novelty study — everything is an LLM-critic proxy. Domain validity is
  uneven (CS/physics show strong discrimination; bio/chem/medicine much
  weaker).
- **Code**: `github.com/HKUST-KnowComp/AgentIdeaBench`, releases hypotheses,
  critic scores, retrieval IDs, bibliographic metadata; CC BY 4.0.

## Ideation Arena — the credibility benchmark

- **Design**: pairwise battles, not absolute scores — two systems' ideas
  shown to a human expert, preference feeds an Elo rating. Evaluates 14
  frontier LLMs **plus 5 named agentic research architectures** — one of few
  benchmarks treating agent pipelines as first-class subjects, not just raw
  LLMs.
- **Scale**: 6,000+ double-blind comparisons from 105 active CS researchers —
  the most human-grounded of the four.
- **Companion sub-benchmark (Ideation Arena Eval)**: can an LLM judge replace
  the human panel? Best automated judge reaches only **72.56% agreement**
  with human preference — even the best current judge is wrong roughly 1 in
  4 times. Load-bearing number against fully automating evaluation.
- **Fit**: not reproducible on demand (105 domain experts), but the target to
  calibrate against once a system is mature — most directly comparable to
  "is my agent pipeline actually better than the base LLM," since some of
  their 5 tested agent architectures helped and others didn't.
- **Code**: `github.com/foss12138/Research-Ideation-Arena`, CC BY 4.0,
  includes code, data, and leaderboard infrastructure.

## Comparison table

| | AIIB | LiveIdeaBench | AgentIdeaBench | Ideation Arena |
|---|---|---|---|---|
| Format | Idea vs. real outcome | Keyword → free idea | Static vs. Active retrieval | Pairwise human battle → Elo |
| Judge | Ground-truth + citation metrics | LLM panel | 3 LLM critics, human-validated | Human experts (LLM judge only 72.6% aligned) |
| Context given | Curated prior papers | Almost none | Curated vs. self-retrieved | Shared, identical across systems |
| Agent-pipeline aware? | No | No | Yes — its whole point | Yes — 5 named architectures |
| Best use for us | Convergence-with-literature test | Per-agent diversity sanity check | **Primary dev-loop eval** | Eventual external credibility target |

**Recommended eval strategy**: build the dev loop on **AgentIdeaBench's
Static-vs-Active protocol** (best validation, most structurally similar to
what we're building), use **LiveIdeaBench** as a cheap per-agent-role
diversity check, and treat **Ideation Arena** as the external target to
eventually benchmark against — not something to run in-house.
