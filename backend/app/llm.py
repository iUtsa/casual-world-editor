"""Shared Claude client access. Returns None when no credentials are
configured so every caller falls back to its deterministic path."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def configured() -> bool:
    if not config.CLAUDE_ENABLED:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()   # `ant auth login` profile


def client():
    if not configured():
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception as e:
        log.info("Claude client unavailable: %s", e)
        return None
