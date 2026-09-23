"""Plain data holders for the orchestrator. No behavior lives here."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Candidate:
    text: str
    round_num: int
    source_strategy: str  # "divergent" | "combine" | "ground_in_literature" | "analogical" | "deliberate_divergence"
    parent_ids: list = field(default_factory=list)
    candidate_id: str = field(default_factory=new_id)
    elo: float = 1200.0
    embedding: list = None
    # Structured idea fields (orchestrator.idea_schema.FIELD_KEYS), when the
    # candidate was produced from the structured prompt contract. `text` is
    # always the human-readable rendering of these fields (or a plain string
    # for candidates with no structure, e.g. loaded from memory).
    structured: dict = field(default_factory=dict)


@dataclass
class CritiqueResult:
    candidate_id: str
    passed: bool
    issues: list = field(default_factory=list)


@dataclass
class DebateVerdict:
    candidate_a_id: str
    candidate_b_id: str
    winner_id: str
    reasoning: str


@dataclass
class RunConfig:
    topic: str
    domain: str
    subfield: str
    backend_kind: str = "openrouter"  # "openrouter" | "claude_code"
    rounds: int = 3
    pool_size: int = 24
    n_divergent_per_round: int = 6
    debate_pairs_per_round: int = 8
    k_final: int = 1
    max_model_calls: int = 200
    duplicate_similarity_threshold: float = 0.92
    diverse_selection_similarity_threshold: float = 0.85
    round_robin_debates: bool = False  # every non-duplicate pair debates once per round, instead of debate_pairs_per_round-capped adjacent-Elo pairing
    run_id: str = field(default_factory=new_id)


@dataclass
class MemoryEntry:
    domain: str
    subfield: str
    text: str
    elo: float
    run_id: str
    embedding: list = None
    memory_id: str = field(default_factory=new_id)


@dataclass
class RunSummary:
    run_id: str
    total_model_calls: int
    compute_multiplier: float
    final_candidates: list
    status: str  # "completed" | "truncated_call_budget"
