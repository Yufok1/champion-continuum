from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import tokenize


_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:\\[^\s\"'<>|]+|(?:[\w.-]+[\\/])+[\w.-]+\.(?:py|js|ts|json|md|txt|ps1|sh|yml|yaml|toml|html|css))"
)


def _dt_ms(value: float) -> int:
    return int(value * 1000)


def _normalize(text: Any, limit: int = 500) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _discover_codex_home(codex_home: str | None = None) -> Path:
    explicit = str(codex_home or os.environ.get("CODEX_HOME") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".codex"


def _session_files(codex_home: str | None = None, limit: int = 200) -> list[Path]:
    home = _discover_codex_home(codex_home)
    root = home / "sessions"
    if not root.exists():
        return []
    files = list(root.rglob("rollout-*.jsonl"))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files[: max(1, int(limit or 200))]


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_collect_strings(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_collect_strings(item))
        return out
    return []


def _extract_paths(value: Any) -> list[str]:
    found: list[str] = []
    for text in _collect_strings(value):
        found.extend(match.rstrip(".,)") for match in _PATH_RE.findall(text))
    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _parse_session(path: Path) -> dict[str, Any] | None:
    session = {
        "session_path": str(path),
        "session_id": "",
        "cwd": "",
        "mtime": path.stat().st_mtime if path.exists() else 0,
        "user_messages": [],
        "assistant_messages": [],
        "tool_names": [],
        "file_mentions": [],
        "search_text": "",
    }
    parts: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if len(raw) > 20000:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                payload = entry.get("payload")
                entry_type = str(entry.get("type") or "")
                if entry_type in {"session_meta", "turn_context"} and isinstance(payload, dict):
                    session["session_id"] = str(payload.get("id") or session["session_id"])
                    session["cwd"] = str(payload.get("cwd") or session["cwd"])
                    parts.append(str(payload.get("cwd") or ""))
                    continue
                if entry_type == "event_msg" and isinstance(payload, dict):
                    event_type = str(payload.get("type") or "")
                    if event_type == "user_message":
                        text = str(payload.get("message") or "")
                        session["user_messages"].append(_normalize(text, 900))
                        session["file_mentions"].extend(_extract_paths(text))
                        parts.append(text)
                    elif event_type == "agent_message":
                        text = str(payload.get("message") or "")
                        session["assistant_messages"].append(_normalize(text, 900))
                        session["file_mentions"].extend(_extract_paths(text))
                        parts.append(text)
                    elif event_type.endswith("_end") or "tool" in event_type:
                        tool = str(payload.get("tool_name") or payload.get("tool") or event_type)
                        session["tool_names"].append(tool)
                        parts.append(tool)
                if entry_type == "response_item" and isinstance(payload, dict):
                    item_type = str(payload.get("type") or "")
                    if item_type == "function_call":
                        tool = str(payload.get("name") or "")
                        if tool:
                            session["tool_names"].append(tool)
                            parts.append(tool)
                    elif item_type == "message":
                        text = "\n".join(_collect_strings(payload.get("content") or []))
                        role = str(payload.get("role") or "")
                        if role == "assistant" and text:
                            session["assistant_messages"].append(_normalize(text, 900))
                            parts.append(text)
    except OSError:
        return None
    session["file_mentions"] = list(dict.fromkeys(session["file_mentions"]))[-16:]
    session["tool_names"] = list(dict.fromkeys(session["tool_names"]))[-16:]
    session["search_text"] = "\n".join(parts).lower()
    return session


def _score(session: dict[str, Any], query_tokens: set[str], cwd: str = "") -> float:
    score = 0.0
    tokens = set(tokenize(session.get("search_text") or ""))
    score += len(query_tokens & tokens) * 4.0
    cwd_hint = str(cwd or "").strip().lower()
    session_cwd = str(session.get("cwd") or "").strip().lower()
    if cwd_hint and session_cwd:
        if session_cwd == cwd_hint:
            score += 80.0
        elif Path(session_cwd).name == Path(cwd_hint).name:
            score += 25.0
        elif cwd_hint in session_cwd or session_cwd in cwd_hint:
            score += 12.0
    age_hours = max(0.0, (datetime.now(timezone.utc).timestamp() - float(session.get("mtime") or 0)) / 3600.0)
    score += max(0.0, 12.0 - min(age_hours / 24.0, 12.0))
    return score


def continuity_status(codex_home: str | None = None, limit: int = 8) -> dict[str, Any]:
    home = _discover_codex_home(codex_home)
    files = _session_files(codex_home, limit=max(limit, 8))
    recent = []
    for path in files[: max(1, int(limit or 8))]:
        parsed = _parse_session(path)
        if not parsed:
            continue
        recent.append(
            {
                "session_id": parsed.get("session_id"),
                "cwd": parsed.get("cwd"),
                "path": str(path),
                "modified_ms": _dt_ms(float(parsed.get("mtime") or 0)),
                "recent_user": (parsed.get("user_messages") or [""])[-1],
            }
        )
    return {
        "status": "ok",
        "codex_home": str(home),
        "session_count_scanned": len(files),
        "recent_sessions": recent,
    }


def continuity_restore(
    summary: str = "",
    cwd: str = "",
    codex_home: str | None = None,
    limit: int = 3,
    scan_limit: int = 80,
) -> dict[str, Any]:
    files = _session_files(codex_home, limit=max(12, min(int(scan_limit or 80), 120)))
    query_tokens = set(tokenize(" ".join([summary, cwd])))
    scored: list[tuple[float, dict[str, Any]]] = []
    for path in files:
        parsed = _parse_session(path)
        if not parsed:
            continue
        score = _score(parsed, query_tokens, cwd=cwd)
        scored.append((score, parsed))
    scored.sort(key=lambda item: (item[0], item[1].get("mtime") or 0), reverse=True)
    matches = scored[: max(1, int(limit or 3))]
    if not matches:
        return {
            "status": "no_archive_match",
            "archive_resume_only": True,
            "summary": summary,
            "cwd": cwd,
            "packet": _fallback_packet(summary=summary, cwd=cwd),
        }
    best_score, best = matches[0]
    packet = _build_packet(best, score=best_score, summary=summary, cwd=cwd)
    return {
        "status": "ok",
        "archive_resume_only": True,
        "best_session": packet["best_session"],
        "matched_sessions": [
            {
                "score": round(score, 3),
                "session_id": item.get("session_id"),
                "cwd": item.get("cwd"),
                "path": item.get("session_path"),
                "modified_ms": _dt_ms(float(item.get("mtime") or 0)),
            }
            for score, item in matches
        ],
        "packet": packet,
    }


def _fallback_packet(summary: str, cwd: str) -> dict[str, Any]:
    return {
        "packet_kind": "continuum_reacclimation",
        "status": "no_archive_match",
        "archive_resume_only": True,
        "objective_seed": summary,
        "cwd": cwd,
        "doctrine": {
            "authority_order": [
                "live runtime or local files",
                "fresh command/test output",
                "continuity records",
                "docs and archive hints",
            ],
            "rule": "Archive continuity helps re-entry; it never decides live truth.",
        },
        "next_reads": [
            "read AGENT_READ_THIS_FIRST.md if present",
            "inspect repo status and local README",
            "search continuity records with the active objective",
            "run project-specific health checks before editing",
        ],
    }


def _build_packet(session: dict[str, Any], score: float, summary: str, cwd: str) -> dict[str, Any]:
    recent_user = (session.get("user_messages") or [])[-5:]
    recent_assistant = (session.get("assistant_messages") or [])[-5:]
    hot_files = (session.get("file_mentions") or [])[-8:]
    hot_tools = (session.get("tool_names") or [])[-8:]
    objective_seed = summary or (recent_user[-1] if recent_user else "resume from continuity archive")
    subject = hot_files[-1] if hot_files else (session.get("cwd") or cwd or session.get("session_path"))
    return {
        "packet_kind": "continuum_reacclimation",
        "status": "archive_restored",
        "archive_resume_only": True,
        "summary": summary,
        "objective_seed": _normalize(objective_seed, 500),
        "subject_key": f"file:{subject}" if hot_files else f"cwd:{subject}",
        "current_pivot_id": "operative_memory_alignment",
        "best_session": {
            "score": round(score, 3),
            "session_id": session.get("session_id"),
            "cwd": session.get("cwd"),
            "path": session.get("session_path"),
            "modified_ms": _dt_ms(float(session.get("mtime") or 0)),
        },
        "recent_user_messages": recent_user,
        "recent_assistant_messages": recent_assistant,
        "hot_files": hot_files,
        "hot_tools": hot_tools,
        "open_loops": recent_user[-1:] if recent_user else [],
        "authority_order": [
            "fresh live/local runtime read",
            "project source and tests",
            "continuity packet",
            "docs and prior archive",
        ],
        "next_reads": [
            "confirm current objective with the latest user request",
            "inspect local project state before writing",
            "use this packet as a recovery hint, not as authority",
            "after any change, run the narrowest available verification",
        ],
    }
