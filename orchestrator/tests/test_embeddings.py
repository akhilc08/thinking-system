import json
from unittest.mock import Mock, patch

import numpy as np

from orchestrator.embeddings import SentenceTransformerBackend, cosine_similarity


def test_cosine_similarity_of_identical_vectors_is_one():
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embed_lazily_loads_model_and_returns_vectors():
    backend = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
    fake_model = Mock()
    # Real sentence-transformers .encode() returns an ndarray, not a plain
    # list — mock it that way so this test reflects actual runtime shape.
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch("orchestrator.embeddings.SentenceTransformer", return_value=fake_model) as mock_ctor:
        result = backend.embed(["idea one", "idea two"])
        backend.embed(["idea three"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_ctor.assert_called_once_with("all-MiniLM-L6-v2")  # loaded once, reused on 2nd call


def test_embed_converts_ndarray_to_json_serializable_list():
    # Regression test: sentence-transformers' encode() defaults to
    # convert_to_numpy=True, returning an ndarray. embed() must convert it
    # to a plain list of lists of Python floats so downstream json.dumps()
    # (via MemoryStore.write) doesn't crash with
    # "TypeError: Object of type ndarray is not JSON serializable".
    backend = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
    fake_model = Mock()
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch("orchestrator.embeddings.SentenceTransformer", return_value=fake_model):
        result = backend.embed(["idea one", "idea two"])

    assert isinstance(result, list)
    assert not isinstance(result, np.ndarray)
    assert json.dumps(result)  # must not raise
