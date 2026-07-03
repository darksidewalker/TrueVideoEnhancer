"""Video rendering pipeline placeholder.

Actual inference logic has been migrated to Go packages under internal/utils/.
This class exists solely for API compatibility during migration.
"""


class Render:
    """Placeholder render orchestrator. Actual inference goes through Go layer."""

    def __init__(self, **kwargs):
        self._params = kwargs

    def render(self):
        """Stub — real inference happens via Go subprocess invocation."""
        pass
