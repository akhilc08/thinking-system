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
    # How many critics actually returned a usable score. AgentIdeaBench's
    # published numbers are 3-critic scores; a 2-critic score here is a
    # degraded result (the drop-highest trim behaves differently with 2
    # values) and is not directly comparable. Recorded so a consumer can tell
    # the two apart instead of silently mixing them.
    n_critics: int = 0
