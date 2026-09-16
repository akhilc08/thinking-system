"""Proximity/dedup service (spec §5.2, §5.3, §5.4): picks informative
debate pairs, prevents wasted debates between near-duplicate ideas, and
powers diversity-aware final selection and memory-store deduplication.
"""
from orchestrator.embeddings import cosine_similarity


class Proximity:
    def __init__(self, embedding_backend):
        self.embedding_backend = embedding_backend

    def compute_embeddings(self, candidates: list) -> None:
        missing = [c for c in candidates if c.embedding is None]
        if not missing:
            return
        vectors = self.embedding_backend.embed([c.text for c in missing])
        for candidate, vector in zip(missing, vectors):
            candidate.embedding = vector

    def is_near_duplicate(self, candidate, others: list, threshold: float) -> bool:
        for other in others:
            if getattr(other, "candidate_id", None) == candidate.candidate_id:
                continue
            if other.embedding is None or candidate.embedding is None:
                continue
            if cosine_similarity(candidate.embedding, other.embedding) >= threshold:
                return True
        return False

    def pick_debate_pairs(self, pool: list, n_pairs: int, duplicate_threshold: float = 0.92) -> list:
        by_elo = sorted(pool, key=lambda c: c.elo, reverse=True)
        pairs = []
        used = set()
        for i, candidate in enumerate(by_elo):
            if len(pairs) >= n_pairs:
                break
            if candidate.candidate_id in used:
                continue
            for other in by_elo[i + 1:]:
                if other.candidate_id in used:
                    continue
                if (
                    candidate.embedding is not None and other.embedding is not None
                    and cosine_similarity(candidate.embedding, other.embedding) >= duplicate_threshold
                ):
                    continue
                pairs.append((candidate, other))
                used.add(candidate.candidate_id)
                used.add(other.candidate_id)
                break
        return pairs

    def select_diverse_topk(self, ranked_pool: list, k: int, threshold: float) -> list:
        by_elo = sorted(ranked_pool, key=lambda c: c.elo, reverse=True)
        selected = []
        for candidate in by_elo:
            if len(selected) >= k:
                break
            if candidate.embedding is not None and self.is_near_duplicate(candidate, selected, threshold):
                continue
            selected.append(candidate)
        return selected
