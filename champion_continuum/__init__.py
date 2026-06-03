"""Portable continuity primitive for agent runtimes."""

from .core import Continuum
from .store import ContinuumStore
from .codex_archive import continuity_restore, continuity_status
from .contracts import companion_package_status
from .processor import parse_commands, process_text, render_results, strip_results
from .compose import render_composition, extract_compose

__all__ = [
    "Continuum",
    "ContinuumStore",
    "companion_package_status",
    "continuity_restore",
    "continuity_status",
    "parse_commands",
    "process_text",
    "render_results",
    "strip_results",
    "render_composition",
    "extract_compose",
]

__version__ = "0.6.11"
