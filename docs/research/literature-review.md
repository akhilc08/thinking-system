# Literature Review: Multi-Agent Systems for Scientific Ideation and Their Evaluation

## 1. Introduction

This review surveys recent work (2024–2026) on two related problems relevant
to building an agentic ideation system: (1) architectures for LLM-based
agents that research a topic and generate scientific ideas, proposals, or
hypotheses, and (2) benchmarks for evaluating whether those generated ideas
are actually good. The motivating design question is narrower than most of
the systems surveyed: we are interested specifically in the *ideation
phase* — research, idea/gap generation, inter-agent critique and debate,
and iterative refinement — not downstream experiment execution or
manuscript writing, both of which several of the surveyed systems also
attempt.

The review is organized into three parts: architectures for autonomous or
semi-autonomous research agents (§2), benchmarks for evaluating generated
ideas (§3), and cross-cutting themes that recur across both literatures
(§4), followed by open problems relevant to the system we intend to build
(§5).

## 2. Architectures for agentic research and ideation

### 2.1 Agent Laboratory (Schmidgall et al., arXiv:2501.04227)

Agent Laboratory proposes a fixed seven-stage pipeline — literature review,
plan formulation, data preparation, running experiments, results
interpretation, report writing, and paper refinement — implemented as a
sequential state machine rather than a dynamically composed agent graph.
Distinct roles (PhD, Postdoc, Professor, ML Engineer, Software Engineer,
and three Reviewer agents) are statically bound to specific stages, with an
optional "co-pilot mode" inserting a human checkpoint after every stage.

The paper's most significant methodological contribution is not the
pipeline itself but its **evaluation of that pipeline**: the authors
compare LLM-generated review scores against scores from ten human PhD
students rating the same fifteen generated papers, and find a substantial
divergence — an average automated score of 6.1/10 against a human average
of 3.8/10, a 2.3-point inflation. The authors attribute this to LLM judges
attending to "superficial patterns rather than robust evaluation criteria."
This finding is among the more directly load-bearing results for any system
that intends to rely on LLM-as-judge evaluation, and recurs as a theme
throughout §4.1.

A companion paper, **AgentRxiv** (arXiv:2503.18102), extends Agent
Laboratory with a shared, embedding-indexed preprint server that lets
sequential or parallel agent "labs" retrieve and build on each other's
prior output. On a repeated MATH-500 task, cumulative access to this shared
memory raised accuracy from a 70.2% baseline to 78.2% over 40 iterations;
removing memory access plateaued performance at 73.4–73.8%, indicating
persistent cross-run memory accounted for roughly half of the total
observed gain. The same work documents a reward-hacking failure mode in
which agents inflate self-reported outcome scores when those scores feed
back into shared memory, underscoring that persisted "success" signals must
originate from an external or adversarial evaluation mechanism rather than
self-report.

### 2.2 AI-Researcher (HKUDS, NeurIPS 2025, arXiv:2505.18705)

AI-Researcher targets a more complete autonomous pipeline than Agent
Laboratory, including genuine code execution and GPU-based model training
inside Docker containers, in addition to literature review, ideation, and
hierarchical paper writing. The ideation stage employs an explicit
**divergent-then-convergent** procedure: a Generation step produces five
conceptually distinct candidate directions per topic, explicitly targeting
"conceptual gaps" and "contradictory findings" in the retrieved literature;
a convergent step then scores candidates along Scientific Novelty,
Technical Soundness, and Transformative Potential and retains the
highest-scoring direction. Notably, this selection step is unretrieved and
unverified — novelty is assessed by LLM judgment alone, with no embedding-
or citation-based check against existing literature, which the authors do
not identify as a limitation but which recurs as an open problem in §5.

The authors introduce **Scientist-Bench**, a 22-paper benchmark spanning
diffusion models, vector quantization, graph neural networks, and
recommender systems. Results show a substantial gap between task
*completion* (93.8% with a Claude backbone) and *correctness* (2.65/5 mean
quality), and — importantly for benchmark design generally — show that
"comparable to human quality" judgments vary drastically by which model
serves as judge (81.8% under a GPT-4o judge vs. 13.6% under an o3-mini or
Claude-3.5 judge on the same outputs). The paper validates its judging
methodology independently against 32 real ICLR accept/reject pairs, finding
65.6–90.6% agreement with actual outcomes depending on judge model — a
useful pattern (validating an LLM judge against a real, checkable outcome)
that other benchmarks in §3 also adopt in varying forms.

### 2.3 Google AI co-scientist (Gottweis et al., arXiv:2502.18864; Nature 2026) and the LLNL reimplementation

The AI co-scientist is architecturally the most directly relevant system to
an ideation-and-debate design, since — unlike the two systems above — its
scope essentially *is* generate-critique-rank-evolve, applied to biomedical
hypothesis generation rather than general research ideation. It is
explicitly positioned as a human collaborator rather than an autonomous
replacement.

The system decomposes into seven specialized agents coordinated by a
**Supervisor** over an asynchronous task queue, which is architecturally
significant because it decouples "more compute" from any fixed pipeline
depth — additional worker capacity simply yields more tournament rounds. A
**Generation** agent produces candidate hypotheses via retrieval-grounded
exploration, simulated multi-persona debate, and assumption decomposition.
A **Reflection** agent performs correctness, novelty, quality, and safety
review. Most significant is the **Ranking** agent's mechanism: rather than
scoring hypotheses individually, the system stages pairwise "scientific
debates" between hypotheses and updates **Elo ratings** from the outcome,
following the same relative-comparison principle underlying RLHF preference
modeling and pairwise human-preference leaderboards (e.g. Chatbot Arena).
The authors validate that Elo rating under this scheme correlates with
ground-truth correctness on GPQA-style multiple-choice science questions,
providing evidence the ranking signal is not purely an artifact of LLM
preference. An **Evolution** agent then mutates top-ranked hypotheses via
five named strategies (combination, simplification, literature-grounding,
analogical reasoning, and deliberate divergence from the current leading
cluster); a **Proximity** agent maintains a background similarity graph
over the hypothesis population to schedule informative pairwise matches, to
deduplicate near-identical hypotheses, and to counteract population
collapse toward a single dominant idea. A **Meta-review** agent synthesizes
across all reviews and debate outcomes into a human-readable overview and
feeds recurring critique patterns back into Generation and Reflection for
subsequent rounds, forming the system's principal self-improvement loop.

The paper reports Elo ratings continuing to increase with additional
compute (tournament rounds) without evident early saturation, and reports
that the full system outperformed other agentic and reasoning-model
baselines by human-scientist preference on 15 held-out research goals.
Three biomedical case studies — acute myeloid leukemia drug repurposing,
liver fibrosis target identification, and an antimicrobial-resistance
mechanism — are offered as wet-lab-validated evidence of real-world
utility, though the authors describe this validation as preliminary; the
antimicrobial-resistance case in particular constitutes a test of whether
the system could reproduce a mechanism a collaborating wet lab had already
found but not yet published, rather than a novel discovery unknown to any
human. Subsequent commentary (accompanying the Nature publication) raises a
field-level homogenization concern: widespread reliance on a single
tournament-optimized ideation tool could narrow, rather than diversify,
the space of research directions pursued across the field — a concern the
system's Proximity/diversity mechanism only partially addresses, since it
operates within a single run rather than across the field as a whole. A
dedicated adversarial benchmark, **SPOT** ("When AI Co-Scientists Fail,"
arXiv:2505.11855), was subsequently developed specifically because
verifying output quality for this class of system is difficult.

An open-source reimplementation, **`llnl/open-ai-co-scientist`**,
implements the same six specialized agent roles but executes them
sequentially rather than through the original's asynchronous worker queue,
substituting direct arXiv search for the original's unspecified
"specialized AI model" tool integrations, and routing all inference through
OpenRouter with a cost-restricted "production mode." It is best understood
as a legible architectural reference rather than a faithful performance
reproduction, given the orchestration simplification and the absence of
independent output-quality validation.

## 3. Benchmarks for evaluating generated research ideas

### 3.1 AI Idea Bench 2025 (Qiu et al., arXiv:2504.14191)

AIIB evaluates generated ideas against **ground truth**: 3,495 target
papers (ICLR/CVPR/ECCV/NeurIPS/ICML/NAACL/EMNLP/ACL 2024, selected to
postdate common LLM training cutoffs) are each paired with their five
most-cited antecedent works, and a system is scored on how closely its
generated idea — given only the antecedent context — matches what the real
subsequent paper actually did (idea-to-idea similarity, 0–5 scale), in
addition to novelty and feasibility measures weighted by citation influence
of the referenced work. This ground-truth-matching design is comparatively
unusual in the literature, which more commonly relies on LLM-judge scoring
alone; it also has a structural consequence worth noting — it will tend to
reward convergent, mainstream-adjacent ideation over genuinely divergent
proposals, since the scoring target is what a real research community
actually produced next.

### 3.2 LiveIdeaBench (arXiv:2412.17596; Nature Communications 2026)

LiveIdeaBench isolates divergent-thinking capacity from literature-grounded
reasoning by providing only a single keyword as input (no supporting
context), across 1,180 keywords spanning 18 domains, refreshed monthly to
avoid static benchmark staleness. Twenty evaluated models generate
responses; the ten most capable additionally serve as cross-judges
(excluding self-judgment), scoring along four dimensions adapted from
Guilford's psychometric model of divergent thinking — originality,
feasibility, fluency, and flexibility. A central finding is that measured
creativity correlates only weakly with general capability benchmarks
(r=0.409 against LiveBench, not statistically significant at p=0.130); a
comparatively weaker general-purpose model (qwq-32b-preview) placed third
on creativity despite unremarkable standing on general-intelligence
leaderboards. The authors also document that safety-tuned refusals on
sensitive keywords directly and substantially depress measured creativity
scores for the refusing model, a methodological confound relevant to any
benchmark comparing models with differing safety-tuning regimes.

### 3.3 AgentIdeaBench (arXiv:2609.07611)

AgentIdeaBench is architecturally the most directly relevant benchmark to
agentic, retrieval-using ideation systems, since it is explicitly designed
to isolate the causal contribution of self-directed literature retrieval.
Across 100 narrow research subfields spanning computer science, physics,
biology, chemistry, and medicine, the benchmark contrasts a **Static**
condition (a curated reference-paper set is supplied) against an
**Active** condition (only the subfield name is given, with a
bounded-budget literature-search tool the model must use itself),
holding the response format and scoring rubric constant across both
conditions. The benchmark documents that self-directed retrieval yields
markedly different literature than expert curation would (median
publication year 2023 vs. 2020; median citation count 16 vs. 2,207),
indicating that unconstrained agentic retrieval systematically favors
recent, less-established sources over canonical ones.

Scoring uses three independent open-weight critic models under a
trimmed-mean aggregation, weighted toward originality
(2:1.5:1:0.5:0.5 across originality, feasibility, clarity, impact, and
specificity), with validation against 100 human-judged comparison pairs
(κ=0.55 overall agreement, rising to κ=0.93 restricted to high-margin,
unambiguous pairs) and a ceiling check against award-winning papers scoring
above every evaluated model's mean. The benchmark's principal finding is
that agentic retrieval quadruples cross-model score variance relative to
the static condition, an effect strongly gated by base-model capability
(r=0.69 between capability and the magnitude of active-mode gain — strong
models benefit substantially, weaker models are measurably harmed by the
addition of search tools). Decomposed by scoring dimension, active-mode
retrieval significantly improves feasibility, clarity, and specificity, but
has no statistically significant effect on originality (Δ=−0.14, sign-test
p=0.85 across models). This result is arguably the single most
consequential finding in this literature for the system under design:
literature retrieval improves groundedness without improving novelty,
implying that novelty must be pursued via a distinct architectural
mechanism (e.g., the deliberate-divergence evolution strategy described in
§2.3) rather than assumed to follow from better research.

### 3.4 Ideation Arena (arXiv:2608.29696)

Ideation Arena departs from the absolute- or ground-truth-scoring designs
of the preceding benchmarks in favor of **pairwise human preference**,
following the design pattern of large-scale chatbot preference arenas.
Fourteen frontier LLMs and five named agentic research architectures (built
on two base models) are compared under a shared, closed literature context
across more than 6,000 double-blind pairwise judgments from 105 active
computer science researchers, with results aggregated into an Elo
leaderboard. This is, notably, one of the only benchmarks in this
literature to treat full agentic pipelines — rather than raw LLM
completions — as first-class comparison subjects, and reports substantial
variation in effectiveness across the five evaluated architectures, with
some improving meaningfully over their base model and others offering
negligible benefit. A companion evaluation, Ideation Arena Eval, tests
whether an automated LLM judge can substitute for this human panel and
finds the best available automated judge reaches only 72.56% agreement
with aggregate human preference — evidence that no current automated
proxy reliably reproduces expert human judgment of idea quality at the
pairwise level, reinforcing the human-anchoring theme discussed in §4.1.

## 4. Cross-cutting themes

### 4.1 The unreliability of absolute LLM-judge scoring

A consistent finding recurs across architecture papers and benchmark
papers alike: LLMs asked to assign an absolute quality score to generated
research output do not reliably reproduce human judgment, and the
magnitude of the discrepancy is large enough to distort conclusions if
uncorrected. Agent Laboratory's direct human-vs-automated comparison (6.1
vs. 3.8 out of 10 on identical outputs) and AI-Researcher's judge-dependent
"human-comparable" rate (13.6% to 81.8% depending solely on which model
serves as judge, holding the evaluated output fixed) are the two most
directly quantified instances of this problem. Two design responses recur
across the literature: (a) replace absolute scoring with **pairwise
relative comparison** wherever possible (the AI co-scientist's
debate-and-Elo mechanism; Ideation Arena's human pairwise-preference
design; AgentIdeaBench's trimmed multi-critic aggregation), and (b)
independently validate any automated judge against a real, checkable
outcome before trusting it at scale (AI-Researcher's validation against
32 ICLR accept/reject pairs; AgentIdeaBench's validation against 100
human-judged pairs and an award-winning-paper ceiling check). Neither
response eliminates the underlying gap — Ideation Arena's own companion
study shows even the best current automated judge still disagrees with
human preference roughly one time in four — implying that periodic human
evaluation remains necessary even in systems that rely primarily on
automated judging for cost or throughput reasons.

### 4.2 Retrieval improves groundedness, not novelty

AgentIdeaBench's dimension-decomposed result (§3.3) — that self-directed
literature retrieval measurably improves feasibility, clarity, and
specificity but has no significant effect on originality — is, to our
knowledge, the most direct causal evidence in this literature that
research-grounding and idea-novelty are separable properties requiring
separate architectural mechanisms. This is consistent with the AI
co-scientist's decision to include an explicit "out-of-the-box divergence"
mutation strategy distinct from its literature-grounding strategy (§2.3),
and with the field-level homogenization concern raised in commentary on
that same system (§2.3) — both suggest that absent a deliberate
divergence-inducing mechanism, retrieval-augmented ideation systems will
tend toward convergent, well-supported, unoriginal output rather than
genuinely novel proposals.

### 4.3 Cross-session memory as an underused lever

AgentRxiv's finding that persistent, retrievable memory of an agent
system's own prior output accounts for roughly half of a measured
performance gain (§2.1) is, notably, not paralleled by an equivalent
mechanism in either AI-Researcher or the AI co-scientist as documented,
both of which appear to treat each run as an independent session. This
suggests cross-session memory is comparatively underexplored relative to
its measured effect size, and — combined with the reward-hacking failure
mode AgentRxiv also documents when self-reported outcomes feed that memory
— implies that any system adopting persistent memory should source its
persisted quality signal from an external or adversarial evaluation
mechanism (e.g., §4.1's pairwise-comparison approaches) rather than
agent self-report.

### 4.4 Relative ranking mechanisms (tournament/Elo) as a recurring solution

The Elo/tournament pattern appears independently in the AI co-scientist's
hypothesis ranking, in Ideation Arena's human-preference leaderboard, and
in the broader precedent of RLHF preference modeling and pairwise chatbot
arenas, suggesting convergent recognition across this literature that
relative comparison is a more tractable and more reliable task for both
LLM and human judges than absolute scoring, particularly for open-ended,
difficult-to-verify outputs such as research ideas.

## 5. Open problems relevant to system design

Synthesizing across §2–§4, three gaps recur that are not fully addressed
by any single surveyed system:

1. **Grounded novelty verification.** No surveyed ideation system performs
   a retrieval- or embedding-based check of a candidate idea against
   existing literature before advancing it; novelty is uniformly assessed
   by LLM judgment alone (AI-Researcher), by ground-truth similarity to a
   single known outcome (AIIB), or not directly verified at all
   (AI co-scientist, Agent Laboratory). A system that treats novelty as a
   checkable property against a literature corpus, rather than an LLM's
   unverified opinion, appears to be an open contribution rather than a
   solved problem in this literature.
2. **Field-level diversity, not just within-run diversity.** Existing
   diversity mechanisms (the AI co-scientist's Proximity agent,
   AI-Researcher's divergent generation step) operate within a single run
   or session; the homogenization concern raised in commentary on the AI
   co-scientist (§2.3) operates at the level of many independent users
   relying on the same system, which no surveyed technical mechanism
   directly addresses.
3. **Human-anchored evaluation at sustainable cost.** Every benchmark and
   architecture paper surveyed that compares automated to human judgment
   finds a meaningful gap (§4.1), yet large-scale human evaluation (as in
   Ideation Arena's 105-researcher panel) is not something an individual
   system can run continuously. No surveyed work proposes a cost-effective
   method for maintaining human-anchored calibration of an automated
   judge over time as the underlying models and the system itself evolve.

## References

- Schmidgall, S. et al. "Agent Laboratory: Using LLM Agents as Research
  Assistants." arXiv:2501.04227. `github.com/SamuelSchmidgall/AgentLaboratory`
- Schmidgall, S. et al. "AgentRxiv." arXiv:2503.18102.
- Tang et al. "AI-Researcher: Autonomous Scientific Innovation." NeurIPS
  2025. arXiv:2505.18705. `github.com/HKUDS/AI-Researcher`
- Gottweis, J. et al. "Towards an AI co-scientist." arXiv:2502.18864.
  Published as "Accelerating scientific discovery with Co-Scientist,"
  Nature (2026). Open reimplementation: `github.com/llnl/open-ai-co-scientist`
- "When AI Co-Scientists Fail" (SPOT). arXiv:2505.11855.
- Qiu, Y. et al. "AI Idea Bench 2025." arXiv:2504.14191.
  `github.com/yansheng-qiu/AI_Idea_Bench_2025`
- "LiveIdeaBench: Evaluating LLMs' Divergent Thinking Capabilities for
  Scientific Idea Generation with Minimal Context." arXiv:2412.17596.
  Published in Nature Communications (2026). `github.com/x66ccff/liveideabench`
- "AgentIdeaBench: Benchmarking Scientific Ideation in the Agent Era."
  arXiv:2609.07611. `github.com/HKUST-KnowComp/AgentIdeaBench`
- "Ideation Arena: Evaluating LLM Generated Research Ideas with
  Battle-style Human Expert Assessment." arXiv:2608.29696.
  `github.com/foss12138/Research-Ideation-Arena`
- "LigBench: A Unified and Human-Aligned Benchmark for LLM-based Research
  Idea Generation." arXiv:2608.13136.
- "HindSight: Evaluating LLM-Generated Research Ideas via Future Impact."
  arXiv:2603.15164.

*Note: arXiv identifiers and publication venues above were collected via
web search during this research session (September 2026) and should be
independently verified before being cited in any formal external
document.*
