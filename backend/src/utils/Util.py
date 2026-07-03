"""Logging bridge for Go backend migration.

This module provides compatibility shims for functions that have been
migrated to Go packages under internal/utils/. The actual implementations
now live in Go for better performance and native execution.
"""

import sys


def log(msg, severity="INFO"):
    """Compatibility shim for Go's Log function."""
    # In production, logging goes directly through Go's logger
    # This shim exists only for backward compatibility during migration
    print(f"[{severity}] {msg}", file=sys.stderr if severity == "ERROR" else sys.stdout)


# Color constants moved to Go (internal/utils/util.go)
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
