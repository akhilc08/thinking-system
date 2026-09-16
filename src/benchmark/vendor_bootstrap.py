"""
Single place that wires up imports from vendor/AgentIdeaBench.

Every other module in this package imports vendored scoring primitives from
HERE, never reaches into vendor/AgentIdeaBench directly. This is deliberate:
it's the one file responsible for making sure we use the real production
rubric (LIT8D_SYSTEM from experiments/e13_litverify_rubric8.py) rather than
the simpler default in evaluation/absolute_scorer.py.
"""
import sys

from project_config.benchmark import VENDOR_ROOT

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import config as vendor_config  # noqa: E402  (vendored repo's own config.py)
from experiments.e13_litverify_rubric8 import (  # noqa: E402
    LIT8D_SYSTEM,
    build_evidence_for_item,
    format_evidence_block,
)
from experiments.e10_idea_anchor_calibration import (  # noqa: E402
    score_one,
    trimmed_mean,
)
from utils.constants import SCORING_DIMS, SCORING_WEIGHTS  # noqa: E402
from data_collection.fetch_ss_search import _get as ss_get, SS_BASE  # noqa: E402

__all__ = [
    "vendor_config", "LIT8D_SYSTEM", "build_evidence_for_item",
    "format_evidence_block", "score_one", "trimmed_mean", "SCORING_DIMS",
    "SCORING_WEIGHTS", "ss_get", "SS_BASE",
]
