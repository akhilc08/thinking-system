from unittest.mock import Mock, patch

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
    fake_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    with patch("orchestrator.embeddings.SentenceTransformer", return_value=fake_model) as mock_ctor:
        result = backend.embed(["idea one", "idea two"])
        backend.embed(["idea three"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_ctor.assert_called_once_with("all-MiniLM-L6-v2")  # loaded once, reused on 2nd call
