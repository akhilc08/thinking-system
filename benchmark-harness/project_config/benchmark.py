"""Project-level config for the benchmark harness (not the vendored repo's own config).

Loads the repo-root .env (shared with the orchestrator) into the process
environment before anything else runs. The vendored repo's own config.py
also loads its own vendor/AgentIdeaBench/.env, but with override=False —
so whatever this loads first wins, and that local vendor .env only fills
in gaps this root one doesn't cover.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AgentIdeaBench"
CACHE_DIR = PROJECT_ROOT / ".cache" / "semantic_scholar"

_ROOT_ENV_FILE = REPO_ROOT / ".env"
if _ROOT_ENV_FILE.exists():
    from dotenv import load_dotenv

    load_dotenv(_ROOT_ENV_FILE, override=False)
