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
