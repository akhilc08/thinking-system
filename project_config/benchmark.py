"""Project-level config for the benchmark harness (not the vendored repo's own config)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AgentIdeaBench"
CACHE_DIR = PROJECT_ROOT / ".cache" / "semantic_scholar"
