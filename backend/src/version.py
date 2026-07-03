import os
import subprocess


def _repo_root() -> str:
    """Walk upward from this file to find the repo root (where .git lives)."""
    root = os.path.dirname(os.path.abspath(__file__))  # backend/src/
    while True:
        if os.path.isdir(os.path.join(root, ".git")):
            return root
        parent = os.path.dirname(root)
        if parent == root:  # reached /
            break
        root = parent
    return ""


def _git_hash() -> str:
    """Return short git commit hash from .git or RVE_VERSION env var."""
    env = os.environ.get("RVE_VERSION")
    if env:
        return env

    repo_root = _repo_root()
    if not repo_root:
        return "unknown"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_root,
        )
        h = result.stdout.strip()
        if h and not result.returncode:
            return h
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "unknown"


__version__ = _git_hash()
