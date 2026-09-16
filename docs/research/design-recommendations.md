# Design recommendations: ideation-only research swarm

Scope, as stated: an agent system that (1) researches a topic, (2) generates
candidate ideas / identifies research gaps / drafts proposals, (3) has agents
debate and critique each other's ideas, (4) evolves/learns across rounds, and
(5) can be measured against baselines to show it's actually good. No
requirement to run real experiments or write full papers — this stops at a
ranked set of well-formed research proposals.

This maps cleanly onto pieces of all three reference systems, minus the
parts of each that are out of scope (code execution, Docker sandboxes,
paper writing). None of the three existing systems is "ideation-only" as a
product — we're extracting the ideation-relevant subset of each.

## 1. Core architecture (borrow: Google AI co-scientist's shape)

The AI co-scientist's agent-role decomposition is the best fit precisely
because it *is* an ideation-and-critique loop already — it just happens to
sit inside a bigger biomedical-research product. Adopt its shape:

```
topic/goal
   -> Generation agents   (produce candidate ideas, grounded in retrieval)
   -> Reflection agents    (critique each idea: correctness, novelty, feasibility)
   -> Ranking agents       (pairwise debate + Elo, NOT absolute scoring)
   -> Evolution agents     (mutate top-ranked ideas via named strategies)
   -> Proximity service    (dedup + diversity, runs continuously)
   -> Meta-review agent    (synthesize into a ranked, human-readable output)
   -> loop back to Generation, seeded by Meta-review's distilled critique
```

Everything runs off an async task queue (Supervisor pattern), so "more
compute" = "more rounds," and that gives a measurable scaling curve for free
— useful evidence when demonstrating the system is worth the cost of running
a swarm vs. a single LLM call.

## 2. Generation stage — borrow AI-Researcher's divergent→convergent shape, fix its gap

Use the **divergent(N)→convergent(1 or top-k)** pattern from AI-Researcher:
generate several conceptually distinct directions per round (not variations
on one theme), explicitly targeting "conceptual gaps" and "contradictory
findings" in retrieved literature, then filter.

**Fix AI-Researcher's known gap**: it filters by pure LLM judgment on
Novelty/Soundness/Transformative-Potential with no grounding. Add a real
grounded-novelty check — embedding search of the proposed idea against a
literature corpus (e.g. Semantic Scholar/arXiv embeddings) before it's
allowed to advance, similar to what AIIB and AgentIdeaBench use as their
scoring backbone. This turns "novelty" from an LLM's opinion into something
checkable against actual prior work.

**Also correct for AgentIdeaBench's retrieval-bias finding**: agents given
free rein to search retrieve recent, low-citation papers by default (median
citations 16 vs. 2,207 for curated sets). If grounding in *established*
literature matters for feasibility, bias retrieval toward citation-weighted
results, not just recency; if the goal is deliberately frontier/contrarian
ideation, this bias may be fine as-is — worth deciding explicitly rather than
inheriting it by accident.

## 3. Debate/critique stage — borrow Google's pairwise-Elo, not absolute scoring

This is the single highest-leverage design choice across everything
researched. **Do not ask an LLM to rate an idea 1–10.** Agent Laboratory
measured directly what happens when you do: LLM reviewers scored their own
system's papers at 6.1/10 average while human reviewers scored the *same
papers* at 3.8/10 — a 2.3-point inflation from self-judging on superficial
patterns.

Instead: put two candidate ideas in front of a debate-judge role, have it
argue which is stronger against the stated goal criteria, and update Elo
ratings from the outcome (winner up, loser down, bigger swings for upsets).
Google validated that Elo-under-this-scheme correlates with actual
correctness (via GPQA), not just LLM taste — this is a load-bearing claim,
worth replicating on our own eval set before trusting the ranking in
production.

Pair debates using a proximity/similarity signal so comparable ideas debate
each other (informative matches) rather than random pairing (wasted
compute).

## 4. Evolution/learning stage — named mutation strategies, not "improve this"

Adopt Google's five explicit strategies rather than a vague improvement
prompt:

1. Combine top ideas
2. Simplify for testability/clarity
3. Ground further in literature
4. Reason by analogy to other domains
5. **Deliberately diverge** from the current leading cluster

**Strategy 5 is not optional.** AgentIdeaBench found that literature
retrieval improves Feasibility/Clarity/Specificity but has essentially zero
effect on Originality (+/-0.14, not significant). If a metric goal is
"generate genuinely novel ideas," nothing in the pipeline will deliver that
except a step that explicitly forces exploration away from the
already-highly-ranked cluster. Without it, more rounds of "evolution" will
just produce fitter versions of the same handful of ideas — a
homogenization failure mode Google's own Nature commentary also flags as a
field-level risk, not just a single-run one.

## 5. Persistent memory across ideation sessions — borrow AgentRxiv's finding

Don't make each ideation run stateless. Agent Laboratory's AgentRxiv
extension measured that giving agents a retrievable memory of prior
generated output (embed + cosine-similarity search over past ideas/papers)
accounted for **roughly half of the total quality gain** they observed
across 40 sequential runs (70.2%→78.2%, dropping to ~73.5% with memory
removed). For an ideation-only system this is cheap to add — a vector store
of every idea the swarm has ever produced, with the Generation stage
retrieving against it before generating something new (both to build on
strong past directions and to avoid regenerating ideas already tried and
rejected).

Watch for the **reward-hacking failure mode** AgentRxiv also surfaced:
agents inflating their own self-reported scores when memory of past
"successful" runs is available. If ideas get a persisted quality score,
that score needs to come from the debate/Elo mechanism (Section 3), not
from agent self-report, or the system will learn to game its own memory.

## 6. Measuring "is this actually good" vs. baselines

This is the part to take most directly from `benchmarks.md`. Concretely:

**Primary dev-loop metric — AgentIdeaBench protocol.** Run the swarm in two
modes on the same set of topics: (a) Static — feed it a curated set of
papers, no search tool; (b) Active — give it only the topic and let it
retrieve its own literature via the actual pipeline being built. Score both
with the same rubric (Originality/Feasibility/Clarity/Impact/Specificity,
Originality-weighted). This directly answers "is the agentic/swarm part of
this system adding value over just prompting a model with curated context,"
which is the core claim worth being able to defend. Reuse or adapt
AgentIdeaBench's critic setup (3 independent LLM critics, trimmed mean) as
the scoring backend rather than inventing a new one — it's already
human-validated (κ=0.55–0.93 agreement with 100 human-judged CS pairs).

**Baselines to compare against, in increasing order of strength:**
1. Single LLM call, no retrieval, no swarm ("just ask GPT/Claude for an
   idea about X")
2. Single LLM call + retrieval (RAG), no debate/evolution
3. This swarm, single generation round, no debate/evolution/memory
4. This swarm, full pipeline (debate + evolution + persistent memory)

Comparing all four on the same topics under the same rubric isolates which
architectural piece (retrieval vs. debate vs. evolution vs. memory) is
actually responsible for any measured quality gain — important because
AgentIdeaBench already shows retrieval alone barely moves novelty, so it
would be easy to over-credit "the swarm" for a gain that's really just "the
retrieval."

**Diversity sanity check — LiveIdeaBench-style.** Periodically run
individual agent roles on bare keywords with minimal context and check
Originality/Fluency/Flexibility scores per role. Cheap early-warning signal
if a particular agent role (e.g. a Generation persona) has collapsed into
producing repetitive, safe outputs — catch this before it pollutes the
debate/evolution stages downstream.

**External credibility target — Ideation Arena.** Once the system is far
enough along to be worth showing outside our own eval loop, that's the
benchmark to submit to / compare against — it's the only one of the four
that treats full agentic pipelines as first-class subjects and scores them
against human expert preference at scale (105 researchers, 6,000+
comparisons). Their own finding that the best automated judge only hits
72.56% agreement with human judgment is the reason not to treat any
in-house LLM-judge score as a final answer — it's a dev-loop signal, not
proof of quality.

**Do not skip human-anchoring entirely.** Agent Laboratory's 2.3-point
LLM-vs-human gap and AI-Researcher's judge-dependent "comparable to human"
rate (13.6% to 81.8% depending on which judge you ask) both point the same
direction: any claim that "the swarm produces good ideas" needs at least a
small periodic human-rated sample as a check on the automated critics, even
if the bulk of day-to-day iteration runs on LLM judges for cost reasons.

## Summary: what to actually build, in priority order

1. Generation agents with retrieval, divergent(N)→convergent(k) selection,
   plus a real embedding-based novelty check against literature (not just
   LLM opinion).
2. Debate-based pairwise ranking with Elo — not absolute LLM scoring.
3. Named evolution/mutation strategies, including a mandatory
   forced-divergence strategy — not optional, this is the only lever that
   demonstrably affects novelty.
4. Persistent vector memory of all past generated ideas, feeding back into
   Generation.
5. Eval harness built on the AgentIdeaBench Static-vs-Active protocol,
   comparing against the 4-tier baseline ladder above, with periodic human
   spot-checks and an eventual Ideation Arena-style external comparison.
