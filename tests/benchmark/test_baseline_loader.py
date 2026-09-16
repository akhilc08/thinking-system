import math

import pandas as pd

from benchmark.baseline_loader import load_published_score
from benchmark.scorer import aggregate_dimension_scores

_CLEAN_CRITICS = {
    "qwen/qwen3.6-plus": {"originality": 9, "feasibility": 6, "clarity": 7, "impact": 8, "specificity": 6},
    "moonshotai/kimi-k2.6": {"originality": 5, "feasibility": 7, "clarity": 6, "impact": 5, "specificity": 5},
    "z-ai/glm-5.1": {"originality": 6, "feasibility": 5, "clarity": 5, "impact": 6, "specificity": 4},
}

_LOW_CRITICS = {
    "qwen/qwen3.6-plus": {"originality": 2, "feasibility": 2, "clarity": 2, "impact": 2, "specificity": 2},
    "moonshotai/kimi-k2.6": {"originality": 3, "feasibility": 3, "clarity": 3, "impact": 3, "specificity": 3},
    "z-ai/glm-5.1": {"originality": 4, "feasibility": 4, "clarity": 4, "impact": 4, "specificity": 4},
}


def _row(idea_model, track, subdomain, idea_index, critic, dims, error=None):
    """Build one row shaped like release_data/core/lit8d_scores_3seed.csv.gz."""
    return {
        "idea_model": idea_model,
        "track": track,
        "subdomain": subdomain,
        "idea_index": idea_index,
        "critic_model": critic,
        "score_originality": dims["originality"] if dims else float("nan"),
        "score_feasibility": dims["feasibility"] if dims else float("nan"),
        "score_clarity": dims["clarity"] if dims else float("nan"),
        "score_impact": dims["impact"] if dims else float("nan"),
        "score_specificity": dims["specificity"] if dims else float("nan"),
        "error": error,
    }


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


def test_load_published_score_excludes_errored_critic_rows(monkeypatch):
    # lit8d_scores_3seed.csv.gz has an `error` column; 515 of its 47,013 rows
    # are error="parse_fail" with NaN in every score_* column. Those NaNs must
    # be dropped before aggregation, or they propagate and the function
    # silently returns nan (a float, so no `is None` check catches it).
    # The vendor's own analyze() does the equivalent via
    # "WHERE variant=? AND scores_json IS NOT NULL".
    rows = [
        _row("m", "C", "sub", 1, critic, dims)
        for critic, dims in _CLEAN_CRITICS.items()
    ]
    rows.append(_row("m", "C", "sub", 1, "broken/critic", None, error="parse_fail"))
    df = pd.DataFrame(rows)
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    score = load_published_score("m", "C", "sub")

    assert score is not None
    assert not math.isnan(score)
    # Must equal the 3 clean critics' aggregation exactly -- i.e. the errored
    # row contributed nothing at all, rather than merely not crashing.
    _, expected = aggregate_dimension_scores(_CLEAN_CRITICS)
    assert score == expected


def test_load_published_score_returns_none_when_every_row_errored(monkeypatch):
    df = pd.DataFrame([
        _row("m", "C", "sub", 1, critic, None, error="parse_fail")
        for critic in _CLEAN_CRITICS
    ])
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    assert load_published_score("m", "C", "sub") is None


def test_load_published_score_skips_fully_errored_idea_and_averages_the_rest(monkeypatch):
    # idea_index 1 is usable, idea_index 2 is entirely parse_fail rows. The
    # returned average should be idea 1's total alone, not a NaN-poisoned mean.
    rows = [
        _row("m", "C", "sub", 1, critic, dims)
        for critic, dims in _CLEAN_CRITICS.items()
    ]
    rows += [
        _row("m", "C", "sub", 2, critic, None, error="parse_fail")
        for critic in _LOW_CRITICS
    ]
    df = pd.DataFrame(rows)
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    score = load_published_score("m", "C", "sub")

    _, expected = aggregate_dimension_scores(_CLEAN_CRITICS)
    assert score == expected


def test_load_published_score_with_idea_index_isolates_that_one_hypothesis(monkeypatch):
    # Comparing our single re-scored hypothesis against a mean over all
    # released hypotheses for the cell understates real per-idea variance.
    # With idea_index given, the result must be that one idea's aggregation.
    rows = []
    for idea_index, critics in [(1, _CLEAN_CRITICS), (2, _LOW_CRITICS)]:
        for critic, dims in critics.items():
            rows.append(_row("m", "C", "sub", idea_index, critic, dims))
    df = pd.DataFrame(rows)
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    _, expected_idea1 = aggregate_dimension_scores(_CLEAN_CRITICS)
    _, expected_idea2 = aggregate_dimension_scores(_LOW_CRITICS)

    assert load_published_score("m", "C", "sub", idea_index=1) == expected_idea1
    assert load_published_score("m", "C", "sub", idea_index=2) == expected_idea2

    # Not blended with, nor averaged against, the other idea_index's rows.
    assert expected_idea1 != expected_idea2
    assert load_published_score("m", "C", "sub", idea_index=1) != (
        (expected_idea1 + expected_idea2) / 2
    )
    # Default (no idea_index) still averages across ideas -- backward compatible.
    assert load_published_score("m", "C", "sub") == (expected_idea1 + expected_idea2) / 2


def test_load_published_score_returns_none_for_unknown_idea_index(monkeypatch):
    df = pd.DataFrame([
        _row("m", "C", "sub", 1, critic, dims)
        for critic, dims in _CLEAN_CRITICS.items()
    ])
    monkeypatch.setattr("benchmark.baseline_loader._load_scores_dataframe", lambda: df)

    assert load_published_score("m", "C", "sub", idea_index=99) is None
