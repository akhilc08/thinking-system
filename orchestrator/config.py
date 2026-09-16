"""Project-level config for the orchestrator. Independent of
vendor/AgentIdeaBench — the orchestrator is a standalone system."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / ".data" / "runs.db"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")

DEFAULT_GENERATION_MODEL = os.environ.get(
    "ORCHESTRATOR_GENERATION_MODEL", "anthropic/claude-sonnet-5"
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ORCHESTRATOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
