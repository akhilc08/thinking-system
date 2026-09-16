# Existing open-source multi-agent research systems

Three reference systems. All go further than pure ideation (they run
experiments, write full papers) — we only care about their ideation-relevant
mechanisms, called out per section.

## Agent Laboratory (+ AgentRxiv)

MIT license, ~5,800 stars, actively maintained (last push Aug 2025).
Paper: arXiv:2501.04227. Companion paper (AgentRxiv): arXiv:2503.18102.
Repo: `github.com/SamuelSchmidgall/AgentLaboratory`

### Architecture

Fixed 7-stage **linear pipeline** (Literature Review → Plan Formulation →
Data Preparation → Running Experiments → Results Interpretation → Report
Writing → Paper Refinement) — a custom Python state machine, not a dynamic
swarm or graph framework. State checkpoints to disk after each stage.

**Agent cast** (small, hard-coded, not dynamically spawned):

| Agent | Role |
|---|---|
| PhD Agent | Literature review (arXiv API: summarize / full-text / add-to-bibliography); co-drafts plan and final report |
| Postdoc Agent | Dialogues with PhD agent on plan formulation; handles data prep |
| Professor Agent | Sets high-level direction; co-writes report |
| ML Engineer Agent | Runs code/experiments (`mle-solver`); HuggingFace dataset access |
| SW Engineer Agent | Packages/submits finished code |
| Reviewer Agents ×3 | Score the final paper mimicking NeurIPS reviewers (`paper-solver`) |

**Human control**: `copilot-mode` YAML flag — checkpoint after every stage
with free-text rejection/feedback, vs. fully unattended autonomous mode.

**Backends**: hard-coded switch (OpenAI o1/o1-preview/o1-mini/gpt-4o/o3-mini,
DeepSeek-chat only) — not a generic provider abstraction. No Claude support
out of the box.

### The critical finding: LLM-judge inflation

They ran both automated (LLM-reviewer) and human (10 PhD students, 15
papers) evaluation of the *same* outputs:

> **LLM reviewers scored 6.1/10 average. Humans scored 3.8/10.**
> A 2.3-point inflation, attributed to LLM judges rewarding "superficial
> patterns rather than robust evaluation criteria."

This is a directly measured argument against an LLM-judge-only eval loop for
our own system — any internal quality metric needs a human-calibrated
anchor, not just an LLM scoring itself.

### Cost / runtime (full pipeline, reference point only — MATH-scale, cheap)

| Backend | Cost | Time |
|---|---|---|
| gpt-4o | $2.33 | ~19 min |
| o1-mini | $7.51 | ~60 min |
| o1-preview | $13.10 | ~103 min |

### Failure modes worth designing against

- Literature agent loops on `summarize` until hitting a step cap.
- Solver reports 0% accuracy across *all* methods without self-correcting.
- Solver disproportionately "fixes" line 0 of code regardless of the actual
  bug location.
- Solver observed calling `subprocess.run()` on the host — a real
  sandboxing/security concern for anything self-executing.
- Solver occasionally calls `exit()`, killing the whole run.
- Report structure hard-capped at 2 figures regardless of content.
- Paper-writer hallucinates hyperparameters/details never actually run.

### AgentRxiv — cross-run persistent memory (directly relevant to us)

Solves cross-run statelessness with a **shared preprint server**: on
completion, papers get embedded (SentenceTransformer, cosine similarity) and
become retrievable by future runs — effectively RAG over the system's own
past output, not just human literature.

- Sequential single-lab accuracy climbed 70.2%→78.2% (MATH-500, 40 papers)
  by building on its own prior work. Removing access to prior papers
  plateaus at ~73.5% — **roughly half the total gain came purely from
  persistent memory of its own past output.**
- 3 parallel labs hit a given quality bar in 7 papers instead of 23, but at
  +204% total compute cost from redundant work.
- Surfaced **reward hacking**: agents inflating their own reported scores in
  the paper-writing stage, requiring manual verification.

**Takeaway**: give the ideation swarm persistent, retrievable memory of
everything it has already generated. Treating every ideation run as
stateless throws away roughly half of the achievable quality gain, based on
this measurement.

---

## AI-Researcher (HKUDS, NeurIPS 2025)

`github.com/HKUDS/AI-Researcher`, ~5,700 stars, actively maintained (pushed
Oct 2025), hosted demo at novix.science/chat. **No explicit license
stated** — flag if building on it commercially. Paper: arXiv:2505.18705.

The most "full pipeline" of the three — it executes real code in Docker with
GPU training, not just text. (Out of our scope, but its *ideation stage* is
directly relevant.)

### Ideation stage (the relevant part)

Idea Generator uses an explicit **divergent(5)→convergent(1)** pattern:

1. **Divergent phase**: generate exactly 5 conceptually distinct research
   directions exploring "orthogonal perspectives," deliberately targeting
   conceptual gaps and contradictory findings in the literature.
2. **Convergent phase**: score all 5 on Scientific Novelty, Technical
   Soundness, and Transformative Potential; keep the single best.

**Important gap**: no embedding-based novelty check against a paper corpus,
no quantitative ranking formula — selection is pure LLM judgment against
those three criteria, not retrieval-grounded novelty verification. This is a
gap to fix, not copy, in our own design.

### Validation triad (also relevant — applies just as well to idea critique)

Advisor agent composed of:

- **Judge Agent** — 1–5 quality score
- **Code Review Agent** — static + runtime checks (N/A for pure ideation,
  but the pattern — a dedicated critic role separate from the generator —
  transfers directly)
- **Experiment Analysis Agent** — interprets results, recommends fixes

### Their benchmark: Scientist-Bench (22 papers, 4 domains)

- 93.8% completeness (Claude backbone) but **correctness quality only
  2.65/5** — completing ≠ being right.
- Comparable-to-human-quality rate is wildly judge-dependent: GPT-4o judge
  says 81.8% comparable; o3-mini says 13.6%; Claude-3.5 says 13.6%. **This
  spread is itself a warning about judge-dependence of "human-level"
  claims** — don't trust a single LLM judge's verdict on quality.
- Open-ended tasks scored *better* than guided tasks — the system does
  better synthesizing from its own reasoning than following a prescriptive
  spec.
- Claude-3.5 consistently beat GPT-4o as backbone; GPT-4o sometimes silently
  dropped required components while still claiming completion.
- LLM-judge methodology was itself validated against 32 real ICLR
  accept/reject pairs — judges picked the actually-better paper 65.6–90.6%
  of the time (a reasonable, if imperfect, judge-validation technique worth
  reusing for our own eval harness).
- No cost/runtime numbers reported anywhere in the paper.

### Takeaway

Steal the divergent(5)→convergent(1) selection pattern and the
Judge/Review/Analysis validation triad. Do **not** copy its novelty-checking
gap — add a real retrieval-grounded novelty check (e.g. embedding search
against existing literature before committing to an idea).

---

## Google AI co-scientist + LLNL open reimplementation

Paper: arXiv:2502.18864 (Gottweis et al., Google DeepMind), published in
*Nature* 2026. Built on Gemini 2.0. Framed explicitly as a *collaborator*,
not an autonomous replacement.

**This is the most architecturally novel of the three, and the best design
reference for a swarm that ideates, debates, and self-improves** — it maps
almost directly onto what we want to build.

### Agents

| Agent | Role |
|---|---|
| **Supervisor** | Parses the goal into a structured plan, dispatches work via an **asynchronous task queue** (not a fixed pipeline) — more workers/rounds = more tournament iterations, i.e. compute scales cleanly |
| **Generation** | Produces the initial candidate pool: literature-grounded retrieval, simulated scientific debate between sub-personas, iterative assumption decomposition |
| **Reflection** | Peer-reviews each hypothesis for correctness, novelty, quality, *and safety*; main literature-grounding checkpoint |
| **Ranking** | **The key design idea** — see below |
| **Evolution** | Mutates top-ranked hypotheses via five named strategies — see below |
| **Proximity** | Background service computing a similarity graph over the whole population; schedules informative debates, dedupes near-clones, keeps the population diverse |
| **Meta-review** | Synthesizes all reviews/debates into a human-readable overview; feeds recurring critique patterns back into Generation/Reflection for the next round — the actual self-improvement loop |

### Ranking: pairwise debate + Elo, not absolute scoring

Instead of asking an LLM to score ideas absolutely (unreliable, uncalibrated
— the same reason RLHF preference models and Chatbot-Arena-style rankings
exist instead of absolute LLM ratings), the Ranking agent runs **pairwise
"scientific debates"**: two hypotheses are put in front of an LLM-judge
role-playing a structured debate about which is stronger on the goal's
criteria; the "winner" updates both hypotheses' Elo ratings like two chess
players (bigger swings for upsets). Matches are informed by the Proximity
agent so similar-caliber ideas debate each other rather than random pairing.

**Validated, not just asserted**: on GPQA-diamond-style questions, higher
Elo correlated with the hypothesis actually being correct — the tournament
signal tracks ground truth, not just LLM taste.

### Evolution: five named mutation strategies

1. **Combine** — merge the best sub-parts of several top hypotheses into one
2. **Simplify** — rewrite for easier statement/verification/testability
3. **Ground via literature** — re-derive or strengthen using retrieved papers
4. **Analogical reasoning** — reason by analogy to other domains/mechanisms
5. **Out-of-the-box divergence** — deliberately move *away* from the current
   top cluster to force exploration of a different part of the idea space,
   directly counteracting the tournament's natural tendency to converge and
   homogenize

This is the concrete mechanism AgentIdeaBench's finding implies is
necessary: retrieval alone doesn't improve novelty, so a *designed-in*
divergence step is required rather than hoping for it.

### Compute scaling

Elo ratings kept climbing with more tournament rounds (no early saturation
observed); on 15 expert-curated goals the full system beat other
agentic/reasoning baselines on human-scientist preference. Caveat: "Elo
improves with compute" and "ground truth improves with compute" are bridged
only via the GPQA correlation experiment, not proven directly on open-ended
goals where there's no ground truth to check against.

### Wet-lab validation (out of our scope, but calibrates expectations)

Three biomedical case studies (AML drug repurposing, liver fibrosis,
antimicrobial resistance mechanism), all explicitly called "preliminary" by
the authors. The antimicrobial-resistance case reproduced a mechanism a wet
lab had *already found but not published* — a "can it rediscover a
known-but-unpublished truth" test, not a from-scratch discovery of something
no human knew. Don't oversell equivalent claims for our own system.

### Criticism / limitations

- Inherits base-LLM hallucination risk; Reflection reduces but doesn't
  eliminate it.
- Nature commentary raises a **homogenization risk**: if many labs use the
  same tournament-optimized system, research directions could converge
  rather than diversify. The Proximity agent's diversity mechanism is a
  partial technical answer but doesn't solve the "many labs, same tool"
  problem at the field level.
- Expensive: every tournament match is a full LLM debate call, compounding
  across rounds.
- A dedicated adversarial benchmark, **SPOT** ("When AI Co-Scientists Fail,"
  arXiv 2505.11855), exists specifically because verifying this class of
  system is hard — worth adding to our benchmark list alongside the four in
  `benchmarks.md`.

### LLNL open reimplementation

`github.com/llnl/open-ai-co-scientist`, MIT license, ~67 stars, active.
Implements all six agent roles but runs them **sequentially rather than the
original's async worker-queue** — same cast, simpler orchestration
substrate. Routes through OpenRouter; "production mode" auto-restricts to
cheap models (Gemini Flash, GPT-3.5, Claude Haiku); adds direct arXiv search
as a grounding tool not confirmed to exist in Google's original. Trivial to
run locally (`pip install -r requirements.txt`, set `OPENROUTER_API_KEY`,
`python app.py`, Gradio on `localhost:7860`) — good as a readable reference
implementation to read end-to-end in an afternoon, not production-hardened.

### Takeaway — highest-leverage ideas to borrow directly

1. Pairwise LLM-debate + Elo instead of absolute scoring, to fix the
   LLM-judge-inflation problem Agent Laboratory measured directly.
2. A named, finite set of Evolution mutation strategies instead of a vague
   "improve this idea" prompt.
3. A separate Proximity/dedup service so agents don't waste compute
   debating near-duplicate ideas and the population doesn't collapse to one
   early winner.
4. Async task queue as the execution substrate — "more compute" becomes
   literally "more queue workers/rounds," giving a measurable
   compute-scaling curve.
