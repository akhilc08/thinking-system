import pandas as pd

from benchmark.baseline_loader import load_published_score
from benchmark.scorer import aggregate_dimension_scores


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
            "idea_index": 1,
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


def test_load_published_score_averages_across_multiple_ideas_not_blended_rows(monkeypatch):
    # A real (idea_model, track, subdomain) triple can have several released
    # hypotheses (idea_index 1, 2, 3), each independently scored by the same
    # 3 critics. This regression test guards against grouping by
    # critic_model alone, which would silently overwrite idea 1's row with
    # idea 2's row for the same critic and blend scores of two different
    # hypotheses as if they were 3 critics scoring one hypothesis.
    idea1_critics = {
        "qwen/qwen3.6-plus": {"originality": 9, "feasibility": 6, "clarity": 7, "impact": 8, "specificity": 6},
        "moonshotai/kimi-k2.6": {"originality": 5, "feasibility": 7, "clarity": 6, "impact": 5, "specificity": 5},
        "z-ai/glm-5.1": {"originality": 6, "feasibility": 5, "clarity": 5, "impact": 6, "specificity": 4},
    }
    idea2_critics = {
        "qwen/qwen3.6-plus": {"originality": 2, "feasibility": 2, "clarity": 2, "impact": 2, "specificity": 2},
        "moonshotai/kimi-k2.6": {"originality": 3, "feasibility": 3, "clarity": 3, "impact": 3, "specificity": 3},
        "z-ai/glm-5.1": {"originality": 4, "feasibility": 4, "clarity": 4, "impact": 4, "specificity": 4},
    }

    rows = []
    for idea_index, critics in [(1, idea1_critics), (2, idea2_critics)]:
        for critic, dims in critics.items():
            rows.append({
                "idea_model": "multi/idea-model", "track": "T", "subdomain": "multi idea subdomain",
                "idea_index": idea_index,
                "critic_model": critic,
                "score_originality": dims["originality"],
                "score_feasibility": dims["feasibility"],
                "score_clarity": dims["clarity"],
                "score_impact": dims["impact"],
                "score_specificity": dims["specificity"],
            })
    df = pd.DataFrame(rows)
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    score = load_published_score("multi/idea-model", "T", "multi idea subdomain")

    _, expected_idea1_total = aggregate_dimension_scores(idea1_critics)
    _, expected_idea2_total = aggregate_dimension_scores(idea2_critics)
    expected = (expected_idea1_total + expected_idea2_total) / 2
    assert score == expected

    # The old (buggy) behavior would build one per_critic_scores dict where
    # each critic's entry is whichever idea_index came last for that critic
    # in file iteration order (here, idea 2's rows, since they're appended
    # after idea 1's for each critic name). Confirm we don't get that value.
    blended_per_critic = dict(idea1_critics)
    blended_per_critic.update(idea2_critics)
    _, blended_total = aggregate_dimension_scores(blended_per_critic)
    assert score != blended_total
