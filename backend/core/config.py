"""
core/config.py — Centralised settings.

- API keys:         loaded from backend/.env via pydantic-settings
- Pipeline params:  loaded from backend/core/config.yaml via PyYAML

Usage in any module:
    from core.config import pipeline_config
    model = pipeline_config["fact_checker"]["reasoning_model"]
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings

# Resolve paths relative to this file so they work regardless of cwd.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
_YAML_FILE = Path(__file__).resolve().parent / "config.yaml"


class Settings(BaseSettings):
    NEWS_API: str = ""

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
    }


def _load_yaml() -> dict[str, Any]:
    """Load config.yaml; returns empty dict if file is missing."""
    if not _YAML_FILE.exists():
        return {}
    with open(_YAML_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Singletons — import these everywhere
settings = Settings()
pipeline_config: dict[str, Any] = _load_yaml()
