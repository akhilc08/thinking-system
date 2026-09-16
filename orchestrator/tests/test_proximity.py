from unittest.mock import Mock

from orchestrator.models import Candidate
from orchestrator.proximity import Proximity


def _candidate(text, elo=1200.0, embedding=None):
    return Candidate(text=text, round_num=1, source_strategy="divergent", elo=elo, embedding=embedding)


def test_compute_embeddings_only_calls_embed_for_missing_ones():
    already_has = _candidate("has one", embedding=[0.0, 1.0])
    needs_one = _candidate("needs one")
    embedding_backend = Mock()
    embedding_backend.embed.return_value = [[1.0, 0.0]]

    proximity = Proximity(embedding_backend)
    proximity.compute_embeddings([already_has, needs_one])

    embedding_backend.embed.assert_called_once_with(["needs one"])
    assert needs_one.embedding == [1.0, 0.0]
    assert already_has.embedding == [0.0, 1.0]


def test_is_near_duplicate_true_above_threshold():
    a = _candidate("idea A", embedding=[1.0, 0.0])
    b = _candidate("idea B", embedding=[1.0, 0.0001])
    proximity = Proximity(Mock())
    assert proximity.is_near_duplicate(a, [b], threshold=0.99) is True


def test_is_near_duplicate_false_below_threshold():
    a = _candidate("idea A", embedding=[1.0, 0.0])
    b = _candidate("idea B", embedding=[0.0, 1.0])
    proximity = Proximity(Mock())
    assert proximity.is_near_duplicate(a, [b], threshold=0.5) is False


def test_pick_debate_pairs_pairs_adjacent_elo_and_skips_duplicates():
    a = _candidate("A", elo=1300.0, embedding=[1.0, 0.0])
    b = _candidate("B", elo=1290.0, embedding=[1.0, 0.0001])  # near-duplicate of A
    c = _candidate("C", elo=1280.0, embedding=[0.0, 1.0])
    proximity = Proximity(Mock())

    pairs = proximity.pick_debate_pairs([a, b, c], n_pairs=1, duplicate_threshold=0.99)

    assert len(pairs) == 1
    ids_in_pair = {pairs[0][0].candidate_id, pairs[0][1].candidate_id}
    assert b.candidate_id not in ids_in_pair or a.candidate_id not in ids_in_pair  # A and B never paired together


def test_select_diverse_topk_skips_near_duplicates_and_stops_at_k():
    a = _candidate("A", elo=1300.0, embedding=[1.0, 0.0])
    b = _candidate("B", elo=1290.0, embedding=[1.0, 0.0001])  # near-dup of A
    c = _candidate("C", elo=1280.0, embedding=[0.0, 1.0])
    proximity = Proximity(Mock())

    selected = proximity.select_diverse_topk([a, b, c], k=2, threshold=0.99)

    assert selected == [a, c]
