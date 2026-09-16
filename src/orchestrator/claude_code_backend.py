"""ClaudeCodeBackend: shells out to the `claude` CLI in headless mode so
swarm runs consume the user's Claude Code plan instead of metered API
tokens (spec §5.1). One subprocess per call. The system prompt is
prepended to the single -p prompt string rather than relying on a
headless-mode system-prompt flag whose exact name may vary by CLI version
— see Task 5's manual verification step.
"""
import json
import subprocess
import time

from config.orchestrator import CLAUDE_CLI_PATH
from orchestrator.backend import BackendError, retry_with_backoff


class ClaudeCodeBackend:
    def __init__(self, cli_path: str = CLAUDE_CLI_PATH, timeout: float = 300.0):
        self.cli_path = cli_path
        self.timeout = timeout

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        def _do_call() -> str:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            try:
                result = subprocess.run(
                    [self.cli_path, "-p", full_prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise BackendError(f"Claude CLI subprocess failed for role={role}: {exc}") from exc

            if result.returncode != 0:
                raise BackendError(f"Claude CLI exited {result.returncode} for role={role}: {result.stderr}")

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise BackendError(f"Claude CLI produced non-JSON stdout for role={role}: {result.stdout!r}") from exc

            text = data.get("result")
            if text is None:
                raise BackendError(f"Claude CLI JSON output missing 'result' key for role={role}: {data}")
            return text

        return retry_with_backoff(_do_call)
