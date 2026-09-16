# Research: multi-agent ideation systems

Background research for designing an agent system whose scope is the **ideation
phase only**: given a topic, spawn agents that research it, generate candidate
ideas / identify research gaps / draft proposals, critique and debate each
other, and evolve those ideas over rounds — with a way to measure whether the
output is actually good, not just plausible-sounding.

Explicitly out of scope for this system (based on what these reference systems
do that we don't want): running real experiments/code, writing full papers,
end-to-end autonomous publication.

## Documents

- [`benchmarks.md`](./benchmarks.md) — four benchmarks for evaluating
  LLM/agent-generated research ideas (AIIB, LiveIdeaBench, AgentIdeaBench,
  Ideation Arena). Use this to decide what to measure ourselves against.
- [`existing-systems.md`](./existing-systems.md) — three open-source
  reference systems (Agent Laboratory + AgentRxiv, AI-Researcher, Google's AI
  co-scientist + the LLNL reimplementation). Use this for architecture
  patterns to borrow or avoid.
- [`design-recommendations.md`](./design-recommendations.md) — synthesis:
  what to actually build, given our narrower ideation-only scope, and how to
  measure it against baselines.
