"""Embedding backend for Proximity's similarity computations (spec §5.2).
Local sentence-transformers model — self-contained, no extra API key,
matches what AgentRxiv used for the same purpose
(docs/research/existing-systems.md). Independent of ModelBackend: the
embedding backend always uses this, regardless of whether the swarm's
text-generation backend is OpenRouter or Claude Code (spec §5.1's backend
choice only covers text-generation agents).
"""
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from config.orchestrator import DEFAULT_EMBEDDING_MODEL


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().encode(texts).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
