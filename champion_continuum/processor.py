"""Text-command processor: gives tool-less agents a continuity tool runtime.

A tool-less agent can only emit and read text. This processor is its tool
runtime: the agent writes commands in a fixed grammar inside its normal output,
the processor extracts and executes them against a Continuum store, and emits a
results block the agent reads on its next turn. A human (or the clipboard relay)
moves the text between the two.

Command grammar (case-insensitive, may appear anywhere in the agent's text):

    [[continuum: remember | <text> | tags=a,b]]
    [[continuum: search | <query> | <limit?>]]
    [[continuum: packet | <objective summary>]]
    [[continuum: receipt | <action> | <result summary>]]

The processor answers with one block:

    [[continuum-results]]
    - remembered: "<text>" (tags: a, b)
    - search "<query>": 2 match(es)
        1. <note text> (tags: ...)
    [[/continuum-results]]
"""

from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from .core import Continuum
from .mcp_proxy import MCPProxy

_CMD_RE = re.compile(r"\[\[\s*(continuum|tools|tool)\s*:\s*(.*?)\]\]", re.IGNORECASE | re.DOTALL)

# A model must never author a results block; only the processor emits them. These
# strip fabricated [[continuum-results]] blocks and stray [[/...]] closing tags.
_RESULTS_BLOCK_RE = re.compile(
    r"\[\[\s*continuum-results\s*\]\].*?(?:\[\[\s*/\s*continuum-results\s*\]\]|$)",
    re.IGNORECASE | re.DOTALL,
)
_CLOSE_TAG_RE = re.compile(r"\[\[\s*/[^\]]*\]\]")

_NATIVE_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "continuum_state": {
        "description": "Read native relay state and local event posture.",
        "args": [],
    },
    "continuum_settings": {
        "description": "Read native relay settings and local store posture.",
        "args": [],
    },
    "continuum_health": {
        "description": "Read in-process Continuum native relay health.",
        "args": [],
    },
    "continuum_providers": {
        "description": "Read provider posture when available; otherwise report that app routing owns providers.",
        "args": [],
    },
    "continuum_faculties": {
        "description": "Read native faculty posture and which faculties need MCP/backends.",
        "args": [],
    },
    "continuum_utility_daemons": {
        "description": "Read utility daemon registry when available.",
        "args": [],
    },
    "continuum_match_daemons": {
        "description": "Find utility daemons by capability or output type when a registry is available.",
        "args": ["capability", "output", "include_stale"],
    },
    "continuum_heartbeat": {
        "description": "Publish a local component heartbeat event.",
        "args": ["component", "status", "slot", "note", "capabilities_json"],
    },
    "continuum_music_forge_state": {
        "description": "Read native Music Forge fallback posture.",
        "args": [],
    },
    "continuum_music_compose_packet": {
        "description": "Build a lightweight song prompt and lyrics packet.",
        "args": ["idea", "style", "lyrics", "language", "duration", "avoid"],
    },
    "continuum_music_backend_preset": {
        "description": "Report that public backend presets require the app/MCP Music Forge backend.",
        "args": ["backend", "prompt", "lyrics", "duration", "seed"],
    },
    "continuum_music_hf_space_schema": {
        "description": "Report that HF Space schema inspection requires the app/MCP Music Forge backend.",
        "args": ["space_id"],
    },
    "continuum_music_generate_preset": {
        "description": "Report that real music generation requires the app/MCP Music Forge backend.",
        "args": ["backend", "prompt", "lyrics", "duration", "seed", "title"],
    },
    "continuum_music_generate_hf_space": {
        "description": "Report that real HF Space music generation requires the app/MCP Music Forge backend.",
        "args": ["space_id", "prompt", "payload_json", "api_name", "title"],
    },
    "continuum_translate_packet": {
        "description": "Build a local translation/cultural bridge packet without sending externally.",
        "args": ["raw_message", "target_language", "source_language", "relationship_tone"],
    },
    "continuum_links": {
        "description": "Read local peer/service link registry when available.",
        "args": [],
    },
    "continuum_slots": {
        "description": "List event slots and current native event counts.",
        "args": [],
    },
    "continuum_events": {
        "description": "Read recent native Continuum events from a slot or all slots.",
        "args": ["slot", "limit"],
    },
    "continuum_post_event": {
        "description": "Append a local Continuum event for coordination.",
        "args": ["kind", "slot", "text", "payload_json", "source"],
    },
    "continuum_create_room": {
        "description": "Create a lightweight local room event and return join hints.",
        "args": ["room_label", "speaker_label", "listener_label", "source_lang", "target_lang", "relationship_tone"],
    },
    "continuum_expressive_wallpaper": {
        "description": "Read expressive wallpaper readiness and speech-rain control contract.",
        "args": [],
    },
    "continuum_wallpaper_text": {
        "description": "Queue text for the expressive wallpaper speech-rain bridge.",
        "args": ["text", "mode", "source", "slot"],
    },
    "continuum_remember": {
        "description": "Store a durable Continuum memory record.",
        "args": ["text", "tags", "kind"],
    },
    "continuum_search": {
        "description": "Search durable Continuum memory records.",
        "args": ["query", "limit"],
    },
    "continuum_process_agent_text": {
        "description": "Execute relay commands emitted by a tool-less agent.",
        "args": ["text", "max_tool_calls"],
    },
    "continuum_whatsapp_send_intent": {
        "description": "Draft a WhatsApp send intent; does not send a message.",
        "args": ["to", "text", "payload_json"],
    },
    "continuum_wallet_intent": {
        "description": "Draft a wallet/payment intent; does not move funds.",
        "args": ["amount_sats", "memo", "asset", "payload_json"],
    },
}


def _native_tool_hits(query: str, limit: int = 6) -> list[dict[str, Any]]:
    q = (query or "").lower().strip()
    scored: list[tuple[int, dict[str, Any]]] = []
    for name, spec in _NATIVE_TOOL_SPECS.items():
        hay = f"{name.lower()} {spec['description'].lower()} {' '.join(spec['args']).lower()}"
        score = 1 if not q else 0
        if q == name.lower():
            score += 100
        if q and q in name.lower():
            score += 20
        if q and q in hay:
            score += 5
        for term in q.split():
            if term in hay:
                score += 2
        if score:
            scored.append((score, {
                "server": "native",
                "name": name,
                "description": spec["description"],
                "args": list(spec["args"]),
                "input_schema": {
                    "type": "object",
                    "properties": {arg: {"type": "string"} for arg in spec["args"]},
                },
                "mcp_url": "in-process-native-relay",
            }))
    scored.sort(key=lambda item: -item[0])
    return [item for _, item in scored[:limit]]


def _event_log_path(root: str | Path) -> Path:
    configured = os.environ.get("CONTINUUM_EVENT_LOG")
    if configured:
        return Path(configured)
    root_path = Path(root)
    if root_path.name == "shared_store":
        return root_path.parent / "continuum_link_events.jsonl"
    return root_path / "continuum_link_events.jsonl"


def _native_event(root: str | Path, kind: str, slot: str, payload: dict[str, Any], source: str) -> dict[str, Any]:
    created_ms = int(time.time() * 1000)
    seed = {
        "created_ms": created_ms,
        "kind": kind,
        "slot": slot,
        "source": source,
        "payload": payload,
    }
    event_id = "native_" + sha256(json.dumps(seed, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    event = {
        "schema": "champion-continuum/link-event/v1",
        "event_id": event_id,
        "created_ms": created_ms,
        "kind": kind,
        "slot": slot,
        "source": source,
        "payload": payload,
    }
    path = _event_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def _read_native_events(root: str | Path, slot: str = "*", limit: int = 25) -> list[dict[str, Any]]:
    path = _event_log_path(root)
    if not path.exists():
        return []
    wanted = (slot or "*").strip().lower()
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_slot = str(event.get("slot") or "personal").lower()
        if wanted not in {"*", "all"} and event_slot != wanted:
            continue
        out.append(event)
        if len(out) >= max(1, min(int(limit or 25), 200)):
            break
    out.reverse()
    return out


def _native_slot_counts(root: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in _read_native_events(root, slot="*", limit=200):
        slot = str(event.get("slot") or "personal")
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def _native_payload_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _unsupported_native(tool_name: str, reason: str) -> dict[str, Any]:
    return {
        "kind": "tool",
        "verb": f"native.{tool_name}",
        "ok": False,
        "error": reason,
        "result": {
            "status": "requires_backend",
            "tool": tool_name,
            "reason": reason,
            "next_step": "Use the visible app control or connect/index the local MCP service when this needs real backend execution.",
        },
    }


def _native_wallpaper_state(root: str | Path) -> dict[str, Any]:
    configured = os.environ.get("CONTINUUM_BACKGROUND_MEDIA") or os.environ.get("CONTINUUM_WALLPAPER_MEDIA") or ""
    return {
        "status": "ok",
        "schema": "champion-continuum/expressive-wallpaper/v1",
        "active": True,
        "asset_hint": configured or "deck-selected asset",
        "speech_rain_ready": True,
        "control_contract": {
            "type": "continuum:speech-rain",
            "transport": "in-process native relay -> deck event log -> browser postMessage",
            "tool": "native.continuum_wallpaper_text",
            "event_log": str(_event_log_path(root)),
            "mutates_external_state": False,
        },
    }


def _execute_native_tool(continuum: Continuum, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    root = continuum.store.root
    if tool_name in {"continuum_state", "continuum_settings", "continuum_health"}:
        return {
            "kind": "tool",
            "verb": f"native.{tool_name}",
            "ok": True,
            "result": {
                "status": "ok",
                "service": "champion-continuum-native-relay",
                "store_root": str(root),
                "event_log": str(_event_log_path(root)),
                "native_tool_count": len(_NATIVE_TOOL_SPECS),
                "slot_counts": _native_slot_counts(root),
            },
        }
    if tool_name == "continuum_faculties":
        return {
            "kind": "tool",
            "verb": f"native.{tool_name}",
            "ok": True,
            "result": {
                "status": "ok",
                "in_process": ["memory", "events", "wallpaper_text", "translation_packet", "intent_drafts"],
                "requires_backend": ["music_generation", "hf_space_schema", "provider_calls", "external_sends"],
            },
        }
    if tool_name in {"continuum_providers", "continuum_utility_daemons", "continuum_match_daemons"}:
        return _unsupported_native(tool_name, "This faculty needs the app registry or indexed MCP sidecar for live data.")
    if tool_name == "continuum_music_forge_state":
        return _unsupported_native(tool_name, "Music Forge state is owned by the app/MCP backend; in-process relay can only draft packets.")
    if tool_name == "continuum_music_compose_packet":
        idea = str(arguments.get("idea") or "").strip()
        style = str(arguments.get("style") or "").strip()
        lyrics = str(arguments.get("lyrics") or "").strip()
        language = str(arguments.get("language") or "en").strip()
        duration = str(arguments.get("duration") or "").strip()
        avoid = str(arguments.get("avoid") or "").strip()
        packet = {
            "status": "ok",
            "kind": "music.compose_packet",
            "prompt": ", ".join(part for part in [idea, style, f"{duration} seconds" if duration else ""] if part),
            "lyrics": lyrics,
            "language": language,
            "avoid": avoid,
            "needs_generation_backend": True,
        }
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": packet}
    if tool_name in {"continuum_music_backend_preset", "continuum_music_hf_space_schema", "continuum_music_generate_preset", "continuum_music_generate_hf_space"}:
        return _unsupported_native(tool_name, "Real music backend calls require the app/MCP Music Forge backend.")
    if tool_name == "continuum_translate_packet":
        raw = str(arguments.get("raw_message") or "").strip()
        target = str(arguments.get("target_language") or "en").strip()
        source = str(arguments.get("source_language") or "auto").strip()
        tone = str(arguments.get("relationship_tone") or "warm, natural").strip()
        return {
            "kind": "tool",
            "verb": f"native.{tool_name}",
            "ok": True,
            "result": {
                "status": "ok",
                "raw_message": raw,
                "source_language": source,
                "target_language": target,
                "relationship_tone": tone,
                "note": "Native fallback records the translation intent; use app/provider faculty for actual translation.",
            },
        }
    if tool_name == "continuum_links":
        links_path = Path(root).parent / "continuum_peer_links.json" if Path(root).name == "shared_store" else Path(root) / "continuum_peer_links.json"
        try:
            links = json.loads(links_path.read_text(encoding="utf-8"))
        except Exception:
            links = {"links": []}
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "path": str(links_path), "peer_links": links}}
    if tool_name == "continuum_slots":
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "slot_counts": _native_slot_counts(root)}}
    if tool_name == "continuum_events":
        events = _read_native_events(root, slot=str(arguments.get("slot") or "*"), limit=int(arguments.get("limit") or 25))
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "events": events}}
    if tool_name == "continuum_post_event":
        payload = _native_payload_json(arguments.get("payload_json"))
        text = str(arguments.get("text") or "").strip()
        if text:
            payload.setdefault("text", text)
        event = _native_event(
            root,
            str(arguments.get("kind") or "continuum.message"),
            str(arguments.get("slot") or "personal"),
            payload,
            str(arguments.get("source") or "native"),
        )
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "event": event}}
    if tool_name == "continuum_heartbeat":
        payload = {
            "component": str(arguments.get("component") or "native-client"),
            "status": str(arguments.get("status") or "ready"),
            "note": str(arguments.get("note") or ""),
            "capabilities": _native_payload_json(arguments.get("capabilities_json") or "[]"),
        }
        event = _native_event(root, "continuum.heartbeat", str(arguments.get("slot") or "control"), payload, "native-heartbeat")
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "event": event}}
    if tool_name == "continuum_create_room":
        label = str(arguments.get("room_label") or "room").strip()
        slot = "room-" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] if label else "room"
        payload = {
            "room_label": label,
            "speaker_label": str(arguments.get("speaker_label") or "speaker"),
            "listener_label": str(arguments.get("listener_label") or "listener"),
            "source_lang": str(arguments.get("source_lang") or "auto"),
            "target_lang": str(arguments.get("target_lang") or "en"),
            "relationship_tone": str(arguments.get("relationship_tone") or "warm, natural"),
        }
        event = _native_event(root, "continuum.room.created", slot, payload, "native-room")
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "ok", "slot": slot, "event": event}}
    if tool_name == "continuum_expressive_wallpaper":
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": _native_wallpaper_state(root)}
    if tool_name == "continuum_wallpaper_text":
        text = str(arguments.get("text") or "").strip()
        if not text:
            return {"kind": "tool", "verb": f"native.{tool_name}", "ok": False, "error": "text_required"}
        mode = str(arguments.get("mode") or "rain")
        source = str(arguments.get("source") or "native-wallpaper")
        slot = str(arguments.get("slot") or "wallpaper")
        payload = {"text": text[:2400], "mode": mode, "source": source}
        event = _native_event(root, "continuum.wallpaper.text", slot, payload, source)
        return {
            "kind": "tool",
            "verb": f"native.{tool_name}",
            "ok": True,
            "result": {"status": "ok", "event": event, "browser_command": {"function": "window.continuumWallpaperCommand", "payload": payload}},
        }
    if tool_name == "continuum_remember":
        text = str(arguments.get("text") or "")
        tags = [item.strip() for item in str(arguments.get("tags") or "").split(",") if item.strip()]
        record = continuum.remember(text, tags=tags, kind=str(arguments.get("kind") or "note"))
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": record}
    if tool_name == "continuum_search":
        hits = continuum.search(str(arguments.get("query") or ""), limit=int(arguments.get("limit") or 8))
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": hits}
    if tool_name == "continuum_settings":
        return {
            "kind": "tool",
            "verb": f"native.{tool_name}",
            "ok": True,
            "result": {"status": "ok", "store_root": str(root), "native_tools": sorted(_NATIVE_TOOL_SPECS)},
        }
    if tool_name == "continuum_process_agent_text":
        text = str(arguments.get("text") or "")
        max_calls = int(arguments.get("max_tool_calls") or 8)
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": process_text(text, root=root, max_tool_calls=max_calls)}
    if tool_name == "continuum_whatsapp_send_intent":
        payload = _native_payload_json(arguments.get("payload_json"))
        payload.setdefault("to", str(arguments.get("to") or ""))
        payload.setdefault("text", str(arguments.get("text") or ""))
        event = _native_event(root, "continuum.intent.whatsapp", "intent", payload, "native-whatsapp-intent")
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "intent_only", "sent": False, "event": event}}
    if tool_name == "continuum_wallet_intent":
        payload = _native_payload_json(arguments.get("payload_json"))
        payload.setdefault("amount_sats", arguments.get("amount_sats") or 0)
        payload.setdefault("memo", str(arguments.get("memo") or ""))
        payload.setdefault("asset", str(arguments.get("asset") or "BTC"))
        event = _native_event(root, "continuum.intent.wallet", "intent", payload, "native-wallet-intent")
        return {"kind": "tool", "verb": f"native.{tool_name}", "ok": True, "result": {"status": "intent_only", "funds_moved": False, "event": event}}
    return {"kind": "tool", "verb": f"native.{tool_name}", "ok": False, "error": f"native tool not implemented: {tool_name}"}


def strip_results(text: str) -> str:
    """Remove any [[continuum-results]] block or stray [[/...]] closing tag a model
    fabricated. Results are the processor's alone; model-written ones are
    hallucinations and must be stripped before display AND before command parsing
    (a fake block can contain [[tool: ...]] templates that would otherwise run)."""
    cleaned = _RESULTS_BLOCK_RE.sub("", text or "")
    cleaned = _CLOSE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _short(value: Any, limit: int = 100) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tags_from(arg: str) -> list[str]:
    if arg.lower().startswith("tags="):
        return [t.strip() for t in arg[5:].split(",") if t.strip()]
    return []


def _parse_kv_args(args_list: list[str]) -> dict[str, Any]:
    """Parse key=value pairs into a dictionary, with basic type casting."""
    out = {}
    joined = "|".join(args_list)
    pairs = [p.strip() for p in joined.split(",") if "=" in p]
    for pair in pairs:
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val.lower() == "true":
            out[key] = True
        elif val.lower() == "false":
            out[key] = False
        elif val.isdigit():
            out[key] = int(val)
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def parse_commands(text: str) -> list[dict[str, Any]]:
    """Extract every [[continuum: ...]] or [[tool: ...]] command."""
    commands: list[dict[str, Any]] = []
    for match in _CMD_RE.finditer(text or ""):
        kind = match.group(1).lower()
        parts = [p.strip() for p in match.group(2).split("|")]
        if not parts or not parts[0]:
            continue
        commands.append({"kind": kind, "verb": parts[0].lower(), "args": parts[1:], "raw": match.group(0)})
    return commands


def _calltool_text(res) -> str:
    """Extract the text content from an MCP CallToolResult (or anything)."""
    parts = []
    for block in (getattr(res, "content", None) or []):
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    if parts:
        return "\n".join(parts)
    return "" if res is None else str(res)


def execute(continuum: Continuum, command: dict[str, Any]) -> dict[str, Any]:
    kind = command.get("kind", "continuum")
    verb = command.get("verb", "")
    args = command.get("args", [])

    # EAPTI Provenance Injection
    provenance = {
        "raw_input": command.get("raw", ""),
        "input_lang": "unknown",
        "lattice_signature": "immutable_gate_hash_pending"
    }

    try:
        if kind == "tools":
            query = args[0] if args else ""
            if not query and verb != "search":
                query = verb  # allow [[tools: <intent>]] as well as [[tools: search | <intent>]]
            hits = continuum.search_tools(query, limit=6)
            native_hits = _native_tool_hits(query, limit=6)
            if native_hits:
                seen = {(str(hit.get("server") or ""), str(hit.get("name") or "")) for hit in hits}
                for hit in native_hits:
                    key = (str(hit.get("server") or ""), str(hit.get("name") or ""))
                    if key not in seen:
                        hits.append(hit)
                        seen.add(key)
                    if len(hits) >= 6:
                        break
            return {"kind": "tools", "verb": "search", "ok": True, "query": query, "hits": hits, "provenance": provenance}
        if kind == "tool":
            if "." in verb:
                server_name, tool_name = verb.split(".", 1)
            else:
                server_name, tool_name = None, verb  # resolve to the sole connected server
            arguments = _parse_kv_args(args)
            if (server_name or "").lower() in {"native", "continuum_native"} or (
                server_name is None and tool_name in _NATIVE_TOOL_SPECS
            ):
                return _execute_native_tool(continuum, tool_name, arguments)
            proxy = MCPProxy(continuum.store.root)
            result = proxy.call_tool_sync(server_name, tool_name, arguments)
            text = _calltool_text(result)
            is_err = bool(getattr(result, "isError", False))
            # some servers return isError=False but say "unknown tool" in the content
            if not is_err and re.search(r"(unknown tool|no such tool|tool not found|method not found)", text, re.I):
                is_err = True
            out = {"verb": verb, "kind": kind, "ok": not is_err, "result": result, "provenance": provenance}
            if is_err:
                out["error"] = (text[:300] or "tool error").strip()
            return out

        if verb == "remember":
            text = args[0] if args else ""
            tags: list[str] = []
            for extra in args[1:]:
                tags += _tags_from(extra)
            record = continuum.remember(text, tags=tags)
            return {"verb": verb, "ok": True, "text": text, "tags": tags, "id": record.get("id")}
        if verb == "search":
            query = args[0] if args else ""
            limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 8
            hits = continuum.search(query, limit=limit)
            return {
                "verb": verb,
                "ok": True,
                "query": query,
                "hits": [{"text": h.get("text", ""), "tags": h.get("tags") or []} for h in hits],
            }
        if verb == "packet":
            summary = args[0] if args else ""
            packet = continuum.packet(summary=summary, scan_limit=8)
            out_state = packet.get("output_state") or {}
            return {
                "verb": verb,
                "ok": True,
                "summary": out_state.get("summary", ""),
                "drift": (out_state.get("drift") or {}).get("status"),
            }
        if verb == "receipt":
            action = args[0] if args else "unspecified"
            summary = args[1] if len(args) > 1 else ""
            receipt = continuum.receipt(action=action, summary=summary)
            return {"verb": verb, "ok": True, "action": action, "id": receipt.get("receipt_id")}
        return {"verb": verb, "ok": False, "error": f"unknown command: {verb}"}
    except Exception as exc:  # processor must never crash the agent loop
        return {"verb": verb, "ok": False, "error": str(exc)}


def render_results(results: list[dict[str, Any]]) -> str:
    lines = ["[[continuum-results]]"]
    if not results:
        lines.append("- no continuum commands found in the message")
    for result in results:
        verb = result.get("verb", "?")
        kind = result.get("kind", "continuum")
        if not result.get("ok") and kind != "tool":
            lines.append(f"- {verb}: ERROR - {result.get('error')}")
        elif kind == "tools":
            hits = result.get("hits") or []
            lines.append(f'- tools matching "{result.get("query")}": {len(hits)} found')
            for h in hits:
                tmpl = ", ".join(f"{a}=" for a in (h.get("args") or []))
                desc = (h.get("description") or "").strip().split("\n")[0][:80]
                lines.append(f"    [[tool: {h.get('server')}.{h.get('name')} | {tmpl}]] :: {desc}")
            if not hits:
                lines.append("    (no matching tools; try different words)")
        elif kind == "tool":
            status = "SUCCESS" if result.get("ok") else "ERROR"
            lines.append(f"- [tool: {verb}] {status}:")
            if not result.get("ok") and result.get("error"):
                lines.append(f"    {result.get('error')}")
            # Format MCP CallToolResult (which has 'content')
            mcp_res = result.get("result")
            if hasattr(mcp_res, "content"):
                for content in mcp_res.content:
                    if hasattr(content, "text"):
                        lines.append(f"    {content.text}")
            else:
                lines.append(f"    {mcp_res}")
        elif verb == "remember":
            tagpart = f" (tags: {', '.join(result['tags'])})" if result.get("tags") else ""
            lines.append(f'- remembered: "{_short(result.get("text"))}"{tagpart}')
        elif verb == "search":
            hits = result.get("hits") or []
            lines.append(f'- search "{result.get("query")}": {len(hits)} match(es)')
            for index, hit in enumerate(hits, 1):
                tags = f" (tags: {', '.join(hit['tags'])})" if hit.get("tags") else ""
                lines.append(f"    {index}. {hit['text']}{tags}")
        elif verb == "packet":
            lines.append(f"- packet: {result.get('summary')} [drift: {result.get('drift')}]")
        elif verb == "receipt":
            lines.append(f"- receipt filed: {result.get('action')}")
    lines.append("[[/continuum-results]]")
    return "\n".join(lines)


def process_text(text: str, root: str | Path = ".continuum", max_tool_calls: int = 8) -> dict[str, Any]:
    """Parse, execute, and render every command in an agent's (or human's) text.

    Commands run in the order written and results are attributed in that same
    order. External `tool` calls are capped per turn (max_tool_calls) so a model
    cannot fire a haywire combinatorial of tools; memory verbs stay unbounded."""
    continuum = Continuum(root)
    commands = parse_commands(text)
    results: list[dict[str, Any]] = []
    tool_calls = 0
    for command in commands:
        if command.get("kind") == "tool":
            tool_calls += 1
            if tool_calls > max_tool_calls:
                results.append({
                    "kind": "tool",
                    "verb": command.get("verb", ""),
                    "ok": False,
                    "error": f"sequence cap reached ({max_tool_calls} tool calls/turn); relay the rest next message",
                })
                continue
        results.append(execute(continuum, command))
    return {
        "command_count": len(commands),
        "results": results,
        "rendered": render_results(results),
    }
