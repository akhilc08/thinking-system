import subprocess
import sys
from pathlib import Path

from benchmark import vendor_bootstrap as vb

# tests/benchmark/test_vendor_bootstrap.py -> parents[2] is the repo root
# (the directory holding pyproject.toml, src/ and project_config/). Derived
# rather than hardcoded so this test survives the worktree being removed after
# a merge, and runs on any other machine or in CI.
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vendor_bootstrap_is_order_independent():
    """Test that vendor_bootstrap works even if project_config.benchmark is imported first.

    This verifies that renaming our config package to project_config eliminates any
    sys.modules collision between our project's config package and the vendored config.py.
    Both namespaces should coexist cleanly with no import-order sensitivity.
    """
    # Run a subprocess that imports project_config.benchmark first, then vendor_bootstrap
    code = """
import sys
sys.path.insert(0, 'src')
import project_config.benchmark as our_cfg
from benchmark import vendor_bootstrap as vb

# Verify both namespaces work correctly after imports
# 1. Our project_config namespace is still accessible via plain attribute access
assert hasattr(our_cfg, 'VENDOR_ROOT'), "project_config.benchmark missing VENDOR_ROOT"
assert hasattr(our_cfg, 'CACHE_DIR'), "project_config.benchmark missing CACHE_DIR"

# 2. Vendor config is accessible via vendor_bootstrap
assert hasattr(vb.vendor_config, 'CRITIC_MODELS'), "vendor_config missing CRITIC_MODELS"
assert isinstance(vb.vendor_config.CRITIC_MODELS, list), "CRITIC_MODELS should be a list"
assert len(vb.vendor_config.CRITIC_MODELS) > 0, "CRITIC_MODELS should not be empty"
print("SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "SUCCESS" in result.stdout, f"Test did not complete successfully: {result.stdout}"


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
