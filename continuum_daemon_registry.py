from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CHANNEL = Path(os.environ.get("CONTINUUM_BRAIN_CHANNEL", ROOT / "cli_brain_channel"))
CONNECTED_DIR = CHANNEL / "connected"
DEFAULT_STALE_AFTER_SECONDS = float(os.environ.get("CONTINUUM_DAEMON_STALE_AFTER", "30"))

SAFE_PERMISSION_DEFAULTS = {
    "can_delete": False,
    "can_publish": False,
    "can_send_messages": False,
    "can_move_funds": False,
    "can_change_auth": False,
    "requires_operator_approval_for_external_effects": True,
}

DEFAULT_LIMITS = {
    "max_job_seconds": 600,
    "max_retries": 1,
    "max_kleene_iterations": 3,
    "max_spend_usd": 0.0,
}

DEFAULT_CARDS: dict[str, dict[str, Any]] = {
    "codex": {
        "kind": "engineering_daemon",
        "capabilities": ["code", "repo_patch", "verify", "tool_orchestration", "receipt"],
        "outputs": ["text", "patch", "receipt"],
        "cost_mode": "quota",
        "risk_level": "medium",
    },
    "hf-provider": {
        "kind": "inference_provider_daemon",
        "capabilities": ["chat", "draft", "critique", "translation", "lyrics", "routing"],
        "outputs": ["text"],
        "cost_mode": "metered_or_quota",
        "risk_level": "low",
    },
    "gemini": {
        "kind": "auditor_daemon",
        "capabilities": ["audit", "critique", "culture", "translation", "alternate_reasoning"],
        "outputs": ["text", "watch_note"],
        "cost_mode": "quota",
        "risk_level": "low",
    },
    "claude": {
        "kind": "relationship_language_daemon",
        "capabilities": ["warmth", "translation", "tone", "critique", "summarize"],
        "outputs": ["text", "watch_note"],
        "cost_mode": "quota",
        "risk_level": "low",
    },
    "wallpaper-reactor": {
        "kind": "expressive_renderer_daemon",
        "capabilities": ["speech_rain", "visual_accent", "mood_color", "background_orchestration"],
        "outputs": ["visual_directive", "receipt"],
        "cost_mode": "local",
        "risk_level": "low",
    },
    "matrix-rain": {
        "kind": "expressive_renderer_daemon",
        "capabilities": ["speech_rain", "glyph_rendering", "audio_reactive_visuals", "background_orchestration"],
        "outputs": ["visual_directive", "receipt"],
        "cost_mode": "local",
        "risk_level": "low",
    },
    "music-forge": {
        "kind": "creative_facility_daemon",
        "capabilities": ["lyrics", "song_packet", "audio_generation_routing", "music_file_request"],
        "outputs": ["text", "audio", "file", "receipt"],
        "cost_mode": "local_or_metered",
        "risk_level": "medium",
    },
}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _merge_dict(base: dict[str, Any], override: Any) -> dict[str, Any]:
    out = dict(base)
    if isinstance(override, dict):
        out.update(override)
    return out


def _default_card(agent: str) -> dict[str, Any]:
    return dict(DEFAULT_CARDS.get(agent.strip().lower(), {
        "kind": "utility_daemon",
        "capabilities": ["chat"],
        "outputs": ["text"],
        "cost_mode": "unknown",
        "risk_level": "unknown",
    }))


def normalize_daemon_card(raw: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    agent = str(raw.get("agent") or "unknown")
    base = _default_card(agent)
    configured = raw.get("capability_card") if isinstance(raw.get("capability_card"), dict) else {}
    capabilities = _as_list(configured.get("capabilities") or raw.get("capabilities") or base.get("capabilities"))
    outputs = _as_list(configured.get("outputs") or raw.get("outputs") or base.get("outputs"))
    permissions = _merge_dict(SAFE_PERMISSION_DEFAULTS, configured.get("permissions") or raw.get("permissions"))
    limits = _merge_dict(DEFAULT_LIMITS, configured.get("limits") or raw.get("limits"))
    ts = float(raw.get("ts") or 0)
    age = round(max(0.0, now - ts), 3) if ts else None
    stale = bool(age is None or age > DEFAULT_STALE_AFTER_SECONDS)
    safety_flags = []
    for key in ("can_delete", "can_publish", "can_send_messages", "can_move_funds", "can_change_auth"):
        if _as_bool(permissions.get(key), False):
            safety_flags.append(key)
    return {
        "schema": "champion-continuum/utility-daemon-card/v1",
        "agent": agent,
        "kind": str(configured.get("kind") or raw.get("kind") or base.get("kind") or "utility_daemon"),
        "status": str(raw.get("status") or "unknown"),
        "busy": bool(raw.get("busy", False)),
        "pid": raw.get("pid"),
        "last_seen_ts": ts,
        "age_seconds": age,
        "stale": stale,
        "can_speak": bool(raw.get("can_speak", False)),
        "can_watch": bool(raw.get("can_watch", False)),
        "capabilities": capabilities,
        "outputs": outputs,
        "cost_mode": str(configured.get("cost_mode") or raw.get("cost_mode") or base.get("cost_mode") or "unknown"),
        "risk_level": str(configured.get("risk_level") or raw.get("risk_level") or base.get("risk_level") or "unknown"),
        "permissions": permissions,
        "limits": limits,
        "safe_for_autonomous_assignment": not safety_flags and bool(raw.get("can_speak", False)) and not stale,
        "safety_flags": safety_flags,
        "root": raw.get("root"),
        "channel": raw.get("channel"),
    }


def load_daemon_registry(channel: Path | None = None) -> dict[str, Any]:
    channel = channel or CHANNEL
    connected = channel / "connected"
    now = time.time()
    daemons: list[dict[str, Any]] = []
    if connected.exists():
        for path in sorted(connected.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(raw, dict):
                daemons.append(normalize_daemon_card(raw, now=now))
    active = [d for d in daemons if not d["stale"]]
    safe = [d for d in active if d["safe_for_autonomous_assignment"]]
    return {
        "status": "ok",
        "schema": "champion-continuum/utility-daemon-registry/v1",
        "channel": str(channel),
        "connected_dir": str(connected),
        "stale_after_seconds": DEFAULT_STALE_AFTER_SECONDS,
        "counts": {
            "total": len(daemons),
            "active": len(active),
            "safe_for_autonomous_assignment": len(safe),
        },
        "daemons": daemons,
        "safety_contract": {
            "external_sends_require_operator_approval": True,
            "wallet_or_funds_movement_requires_operator_approval": True,
            "delete_publish_auth_change_requires_operator_approval": True,
            "kleene_loops_are_bounded_by_limits": True,
            "receipts_are_append_only": True,
        },
    }


def match_daemons(capability: str, output: str = "", include_stale: bool = False) -> dict[str, Any]:
    capability = str(capability or "").strip().lower()
    output = str(output or "").strip().lower()
    registry = load_daemon_registry()
    matches = []
    for daemon in registry["daemons"]:
        caps = {str(item).lower() for item in daemon.get("capabilities") or []}
        outs = {str(item).lower() for item in daemon.get("outputs") or []}
        if not include_stale and daemon.get("stale"):
            continue
        if capability and capability not in caps:
            continue
        if output and output not in outs:
            continue
        matches.append(daemon)
    return {
        "status": "ok",
        "capability": capability,
        "output": output,
        "count": len(matches),
        "matches": matches,
    }
