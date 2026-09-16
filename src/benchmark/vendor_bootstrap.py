"""
Single place that wires up imports from vendor/AgentIdeaBench.

Every other module in this package imports vendored scoring primitives from
HERE, never reaches into vendor/AgentIdeaBench directly. This is deliberate:
it's the one file responsible for making sure we use the real production
rubric (LIT8D_SYSTEM from experiments/e13_litverify_rubric8.py) rather than
the simpler default in evaluation/absolute_scorer.py.
"""
import sys
import importlib.util

from config.benchmark import VENDOR_ROOT

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

# Critical: The vendored code imports "config" directly. Since we just imported
# config.benchmark, sys.modules["config"] is now our project's config package.
# To fix this, we:
# 1. Remove our config package from sys.modules
# 2. Load the vendored config.py using importlib.util and inject it as "config"
# 3. Then import the vendored experiments modules
# This ensures vendored code gets the correct "config" module when it imports.

# Save our config package so we can restore it later if needed
_our_config = sys.modules.pop("config", None)

try:
    # Load vendored config.py and inject it into sys.modules as "config"
    # so vendored code gets it when doing "import config"
    _vendor_config_path = VENDOR_ROOT / "config.py"
    _spec = importlib.util.spec_from_file_location("config", _vendor_config_path)
    vendor_config = importlib.util.module_from_spec(_spec)
    sys.modules["config"] = vendor_config
    _spec.loader.exec_module(vendor_config)  # noqa: E402
finally:
    # Restore our config package if needed (for any code that needs config.benchmark)
    if "config" in sys.modules and sys.modules["config"] is vendor_config:
        # Keep the vendored config in sys.modules for now
        pass
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
