"""Shared utility helpers used across all agents."""

import os


def read_env_var(*names: str) -> str:
    """Read the first non-empty environment variable from candidate names.

    Strips surrounding whitespace and quotes so values like
    ``GROQ_API_KEY="gsk_abc..."`` are handled cleanly.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return ""
