from benchmark import vendor_bootstrap as vb


def test_lit8d_system_is_the_evidence_grounded_rubric_not_the_simple_default():
    # Guards against silently importing evaluation/absolute_scorer's simpler
    # SYSTEM_PROMPT instead of the real production rubric.
    assert "EVIDENCE-BASED ORIGINALITY" in vb.LIT8D_SYSTEM
    assert "IDEA-LEVEL SCALE RECALIBRATION" in vb.LIT8D_SYSTEM


def test_scoring_dims_and_weights_match_published_values():
    assert vb.SCORING_DIMS == [
        "originality", "feasibility", "clarity", "impact", "specificity",
    ]
    assert vb.SCORING_WEIGHTS == {
        "originality": 2.0,
        "feasibility": 1.0,
        "clarity": 0.5,
        "impact": 1.5,
        "specificity": 0.5,
    }


def test_trimmed_mean_drops_the_highest_value():
    assert vb.trimmed_mean([3, 5, 9]) == 4.0  # drops 9, mean of [3, 5]


def test_vendor_callables_are_importable_and_callable():
    assert callable(vb.score_one)
    assert callable(vb.build_evidence_for_item)
    assert callable(vb.format_evidence_block)
    assert callable(vb.ss_get)
    assert isinstance(vb.SS_BASE, str) and vb.SS_BASE.startswith("https://")
