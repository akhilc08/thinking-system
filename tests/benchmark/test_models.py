from benchmark.models import DimensionScore, Hypothesis, ScoreResult


def test_hypothesis_holds_identifying_fields():
    h = Hypothesis(
        topic="graph neural network expressiveness",
        domain="Computer Science",
        subfield="graph neural network expressiveness",
        text="We hypothesize that...",
        source="our_swarm",
    )
    assert h.domain == "Computer Science"
    assert h.source == "our_swarm"


def test_score_result_holds_aggregation_output():
    h = Hypothesis("t", "Computer Science", "t", "text", "our_swarm")
    result = ScoreResult(
        hypothesis=h,
        per_critic={
            "qwen/qwen3.6-plus": {
                "originality": DimensionScore("originality", 7, "reason"),
            },
        },
        per_dimension_trimmed={"originality": 6.5},
        weighted_total=6.2,
    )
    assert result.weighted_total == 6.2
    assert result.per_dimension_trimmed["originality"] == 6.5
