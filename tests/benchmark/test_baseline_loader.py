import pandas as pd

from benchmark.baseline_loader import load_published_score


def _fake_scores_df():
    # Mirrors release_data/core/lit8d_scores_3seed.csv.gz columns.
    rows = []
    for critic, o, f, c, i, s in [
        ("qwen/qwen3.6-plus", 9, 6, 7, 8, 6),
        ("moonshotai/kimi-k2.6", 5, 7, 6, 5, 5),
        ("z-ai/glm-5.1", 6, 5, 5, 6, 4),
    ]:
        rows.append({
            "idea_model": "glm/glm-5.1", "track": "C", "subdomain": "graph nn expressiveness",
            "critic_model": critic,
            "score_originality": o, "score_feasibility": f, "score_clarity": c,
            "score_impact": i, "score_specificity": s,
        })
    return pd.DataFrame(rows)


def test_load_published_score_matches_our_own_aggregation(monkeypatch):
    monkeypatch.setattr(
        "benchmark.baseline_loader._load_scores_dataframe",
        lambda: _fake_scores_df(),
    )
    score = load_published_score("glm/glm-5.1", "C", "graph nn expressiveness")
    # Same numbers as test_score_hypothesis_calls_all_three_critics_and_aggregates:
    # originality [9,5,6] -> drop 9 -> mean([5,6]) = 5.5, etc.
    assert score is not None
    assert 5.0 < score < 6.0


def test_load_published_score_returns_none_when_not_found(monkeypatch):
    monkeypatch.setattr(
        "benchmark.baseline_loader._load_scores_dataframe",
        lambda: _fake_scores_df(),
    )
    score = load_published_score("nonexistent/model", "C", "graph nn expressiveness")
    assert score is None
