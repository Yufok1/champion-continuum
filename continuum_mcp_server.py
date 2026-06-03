from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from champion_continuum import Continuum, process_text
from continuum_daemon_registry import load_daemon_registry, match_daemons
from continuum_link_server import (
    ALL_SLOTS,
    CHANNEL,
    ROOT,
    _append_event,
    _build_heartbeat,
    _build_room_session,
    _make_event,
    _peer_link_state,
    _recent_events,
    _settings_state,
    _slot_counts,
    _state,
)
from continuum_music_forge import (
    compose_song_packet,
    generate_music_preset,
    generate_hf_space_song,
    hf_space_schema,
    music_backend_preset_payload,
    music_forge_state,
)
from continuum_provider_registry import provider_registry_state
from continuum_translation_faculty import build_translation_faculty_packet, translation_faculty_state
from continuum_whatsapp_adapter import build_send_intent, build_wallet_intent


SHARED_STORE_ROOT = Path(os.environ.get("CONTINUUM_SHARED_STORE", CHANNEL / "shared_store"))

BACKGROUND_MEDIA_CANDIDATES = [
    "continuum_wallpaper.html",
    "continuum_wallpaper.webm",
    "continuum_wallpaper.mp4",
    "continuum_wallpaper.gif",
    "continuum_wallpaper.png",
    "continuum_wallpaper.jpg",
    "continuum_wallpaper.jpeg",
]


def _continuum() -> Continuum:
    SHARED_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    return Continuum(SHARED_STORE_ROOT)


def _json_object(text: str, field_name: str = "payload_json") -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return value


def _wallpaper_preset_settings() -> dict[str, dict[str, Any]]:
    return {
        "aurora": {
            "colorPreset": "aurora",
            "pattern": "harmonic",
            "direction": "diagonal",
            "fontSize": 18,
            "density": 86,
            "intensity": 78,
            "speed": 58,
        },
        "hyperneon": {
            "colorPreset": "hyperneon",
            "pattern": "rainbow",
            "direction": "toward",
            "fontSize": 20,
            "density": 92,
            "intensity": 92,
            "speed": 70,
        },
        "calm": {
            "colorPreset": "zen",
            "pattern": "classic",
            "direction": "down",
            "fontSize": 14,
            "density": 52,
            "intensity": 45,
            "speed": 24,
            "settingsPanel": "minimize",
        },
        "presentation": {
            "colorPreset": "crystal",
            "pattern": "classic",
            "direction": "down",
            "fontSize": 24,
            "density": 42,
            "intensity": 64,
            "speed": 32,
            "settingsPanel": "minimize",
        },
        "audio": {
            "audioReactive": True,
            "audioDiagonals": True,
            "audioReverse": False,
            "colorPreset": "prism",
            "fontSize": 16,
            "density": 88,
            "intensity": 80,
        },
        "council": {
            "colorPreset": "neon",
            "pattern": "pentad",
            "direction": "toward",
            "fontSize": 18,
            "density": 90,
            "intensity": 88,
            "speed": 62,
        },
        "chaos": {
            "command": "chaos_once",
            "colorPreset": "bassstorm",
            "fontSize": 17,
            "density": 100,
            "intensity": 95,
        },
    }


def _wallpaper_control_payload(
    text: str = "",
    settings_json: str | dict[str, Any] = "",
    command: str = "",
    source: str = "mcp-wallpaper-control",
    slot: str = "wallpaper",
) -> dict[str, Any]:
    if isinstance(settings_json, dict):
        settings = settings_json
    else:
        settings = _json_object(settings_json or "", "settings_json")
    clean = (text or "").strip()
    return {
        "text": clean[:2400],
        "settings": settings,
        "settings_json": json.dumps(settings, ensure_ascii=False, sort_keys=True) if settings else "",
        "command": (command or str(settings.get("command") or "")).strip(),
        "source": source or "mcp-wallpaper-control",
        "slot": slot or "wallpaper",
    }


def _append_or_raise(event: dict[str, Any]) -> dict[str, Any]:
    error = _append_event(event)
    if error:
        raise RuntimeError(f"event log write failed: {error}")
    return event


def _expressive_wallpaper_state() -> dict[str, Any]:
    configured = os.environ.get("CONTINUUM_BACKGROUND_MEDIA") or os.environ.get("CONTINUUM_WALLPAPER_MEDIA") or ""
    selected = configured.strip()
    if not selected:
        asset_dir = ROOT / "assets"
        for name in BACKGROUND_MEDIA_CANDIDATES:
            candidate = asset_dir / name
            if candidate.exists():
                selected = str(candidate)
                break
    suffix = Path(selected.split("?", 1)[0]).suffix.lower() if selected else ""
    return {
        "status": "ok",
        "schema": "champion-continuum/expressive-wallpaper/v1",
        "active": bool(selected),
        "asset": selected,
        "kind": "web_wallpaper" if suffix in {".html", ".htm"} else ("video" if suffix in {".webm", ".mp4", ".mov", ".m4v"} else ("image" if selected else "none")),
        "speech_rain_ready": bool(selected and suffix in {".html", ".htm"}),
        "control_contract": {
            "types": ["continuum:speech-rain", "continuum:wallpaper-control"],
            "transport": "deck postMessage to embedded wallpaper iframe",
            "inputs": [
                "assistant_text",
                "council_text",
                "daemon_directive",
                "continuum_wallpaper_text",
                "continuum_wallpaper_control",
                "continuum_wallpaper_preset",
            ],
            "outputs": ["glyph_rain", "pattern", "direction", "color", "speed", "intensity", "font_size", "audio_reactivity", "settings_modal"],
            "settings_json_keys": [
                "fontSize", "characterSize", "pattern", "direction", "primaryColor", "secondaryColor",
                "speed", "intensity", "density", "characterSet", "customCharacters", "colorPreset",
                "hueReactivity", "saturationGain", "brightnessDepth", "audioReactive", "audioReverse",
                "audioDiagonals", "autoOrchestrator", "reverseFlow", "settingsPanel", "canvasOpacity",
            ],
            "commands": [
                "chaos_once", "toggle_audio", "audio_on", "audio_off", "auto_on", "auto_off",
                "reverse_flow", "settings_open", "settings_minimize", "settings_close",
            ],
            "presets": sorted(_wallpaper_preset_settings()),
            "mutates_external_state": False,
        },
        "tool_control": [
            {
                "name": "continuum_wallpaper_text",
                "event_kind": "continuum.wallpaper.text",
                "slot": "wallpaper",
                "note": "Queues text for the local deck wallpaper bridge; no external network effect.",
            },
            {
                "name": "continuum_wallpaper_control",
                "event_kind": "continuum.wallpaper.control",
                "slot": "wallpaper",
                "note": "Queues settings, modal, audio-reactive, and orchestration commands.",
            },
            {
                "name": "continuum_wallpaper_preset",
                "event_kind": "continuum.wallpaper.control",
                "slot": "wallpaper",
                "note": "Applies a named preset such as aurora, audio, council, or presentation.",
            },
        ],
    }


def create_mcp(host: str, port: int) -> FastMCP:
    mcp = FastMCP(
        "Champion Continuum MCP",
        instructions=(
            "Champion Continuum exposes memory, tool relay, room, event, provider, "
            "translation, music-forge, WhatsApp-intent, wallet-intent, and receipt surfaces for "
            "tool-less agents. External effects are intents only unless an operator "
            "adds an approval layer."
        ),
        host=host,
        port=port,
        sse_path="/mcp/sse",
        message_path="/mcp/messages/",
        streamable_http_path="/mcp",
    )

    @mcp.tool()
    def continuum_health() -> dict[str, Any]:
        """Read the local Continuum MCP service health."""
        return {
            "status": "ok",
            "service": "champion-continuum-mcp",
            "root": str(ROOT),
            "channel": str(CHANNEL),
            "shared_store": str(SHARED_STORE_ROOT),
            "time": time.time(),
            "transports": {
                "sse": f"http://{host}:{port}/mcp/sse",
                "streamable_http": f"http://{host}:{port}/mcp",
            },
        }

    @mcp.tool()
    def continuum_state() -> dict[str, Any]:
        """Read the Continuum link/event service state."""
        return _state()

    @mcp.tool()
    def continuum_settings() -> dict[str, Any]:
        """Read settings, facilities, providers, privacy posture, and peer/service links."""
        return _settings_state()

    @mcp.tool()
    def continuum_slots() -> dict[str, Any]:
        """List event slots and current slot counts."""
        return {"status": "ok", "slots": list(_state().get("slots") or []), "slot_counts": _slot_counts()}

    @mcp.tool()
    def continuum_faculties() -> dict[str, Any]:
        """Read translation and cultural bridge faculty readiness."""
        return {"status": "ok", "translation_faculty": translation_faculty_state()}

    @mcp.tool()
    def continuum_providers() -> dict[str, Any]:
        """Read available model/provider routing posture."""
        return {"status": "ok", "provider_registry": provider_registry_state()}

    @mcp.tool()
    def continuum_utility_daemons() -> dict[str, Any]:
        """Read live utility daemon capability cards and safety posture."""
        return load_daemon_registry(CHANNEL)

    @mcp.tool()
    def continuum_expressive_wallpaper() -> dict[str, Any]:
        """Read expressive wallpaper readiness and the council speech-rain/settings control contract."""
        return _expressive_wallpaper_state()

    @mcp.tool()
    def continuum_wallpaper_text(
        text: str,
        mode: str = "rain",
        source: str = "mcp-wallpaper",
        slot: str = "wallpaper",
    ) -> dict[str, Any]:
        """Queue text for the local expressive wallpaper speech-rain bridge."""
        clean = (text or "").strip()
        if not clean:
            return {"status": "error", "error": "text_required"}
        payload = {
            "text": clean[:2400],
            "mode": mode or "rain",
            "source": source or "mcp-wallpaper",
        }
        event = _make_event(
            {
                "kind": "continuum.wallpaper.text",
                "slot": slot or "wallpaper",
                "payload": payload,
            },
            source=source or "mcp-wallpaper",
        )
        return {
            "status": "ok",
            "event": _append_or_raise(event),
            "browser_command": {
                "function": "window.continuumWallpaperCommand",
                "payload": payload,
            },
        }

    @mcp.tool()
    def continuum_wallpaper_control(
        text: str = "",
        settings_json: str = "",
        command: str = "",
        source: str = "mcp-wallpaper-control",
        slot: str = "wallpaper",
    ) -> dict[str, Any]:
        """Queue expressive wallpaper settings, audio-reactive, modal, and orchestration commands."""
        payload = _wallpaper_control_payload(text, settings_json, command, source, slot)
        if not payload.get("text") and not payload.get("settings") and not payload.get("command"):
            return {"status": "error", "error": "text_settings_or_command_required"}
        event = _make_event(
            {
                "kind": "continuum.wallpaper.control",
                "slot": payload["slot"],
                "payload": payload,
            },
            source=payload["source"],
        )
        return {
            "status": "ok",
            "event": _append_or_raise(event),
            "browser_command": {
                "function": "window.continuumWallpaperCommand",
                "payload": payload,
            },
        }

    @mcp.tool()
    def continuum_wallpaper_preset(
        preset: str = "council",
        text: str = "",
        source: str = "mcp-wallpaper-preset",
        slot: str = "wallpaper",
    ) -> dict[str, Any]:
        """Apply a named expressive wallpaper preset."""
        presets = _wallpaper_preset_settings()
        preset_name = (preset or "council").strip().lower()
        settings = dict(presets.get(preset_name) or presets["council"])
        command = str(settings.pop("command", ""))
        payload = _wallpaper_control_payload(
            text=text,
            settings_json=settings,
            command=command,
            source=source or f"mcp-wallpaper-preset:{preset_name}",
            slot=slot,
        )
        event = _make_event(
            {
                "kind": "continuum.wallpaper.control",
                "slot": payload["slot"],
                "payload": payload,
            },
            source=payload["source"],
        )
        return {
            "status": "ok",
            "preset": preset_name if preset_name in presets else "council",
            "available_presets": sorted(presets),
            "event": _append_or_raise(event),
            "browser_command": {
                "function": "window.continuumWallpaperCommand",
                "payload": payload,
            },
        }

    @mcp.tool()
    def continuum_match_daemons(capability: str = "", output: str = "", include_stale: bool = False) -> dict[str, Any]:
        """Find live utility daemons by capability/output type."""
        return match_daemons(capability=capability, output=output, include_stale=include_stale)

    @mcp.tool()
    def continuum_music_forge_state() -> dict[str, Any]:
        """Read Music Forge readiness, output directory, and suggested public HF music Spaces."""
        return music_forge_state()

    @mcp.tool()
    def continuum_music_compose_packet(
        idea: str,
        style: str = "",
        lyrics: str = "",
        language: str = "en-US",
        duration: str = "30 seconds",
        avoid: str = "Do not mimic living artists or request a copyrighted song clone.",
    ) -> dict[str, Any]:
        """Build a song prompt/lyrics packet for a music generation backend."""
        return compose_song_packet(
            idea=idea,
            style=style,
            lyrics=lyrics,
            language=language,
            duration=duration,
            avoid=avoid,
        )

    @mcp.tool()
    def continuum_music_hf_space_schema(space_id: str = "ACE-Step/Ace-Step-v1.5") -> dict[str, Any]:
        """Inspect a Hugging Face music Space API before generating audio."""
        return hf_space_schema(space_id)

    @mcp.tool()
    def continuum_music_backend_preset(
        backend: str = "ace_jam",
        prompt: str = "",
        lyrics: str = "",
        duration: float = 30.0,
        seed: int = -1,
    ) -> dict[str, Any]:
        """Build a ready-to-call public music backend payload."""
        return music_backend_preset_payload(
            backend=backend,
            prompt=prompt,
            lyrics=lyrics,
            duration=duration,
            seed=seed,
        )

    @mcp.tool()
    def continuum_music_generate_preset(
        backend: str = "ace_jam",
        prompt: str = "",
        lyrics: str = "",
        duration: float = 30.0,
        seed: int = -1,
        title: str = "",
    ) -> dict[str, Any]:
        """Generate music through a known public HF Space preset and save returned audio files locally."""
        result = generate_music_preset(
            backend=backend,
            prompt=prompt,
            lyrics=lyrics,
            duration=duration,
            seed=seed,
            title=title,
        )
        event = _make_event(
            {
                "kind": "continuum.music.generated",
                "slot": "facilities",
                "payload": {
                    "backend": backend,
                    "title": title or prompt[:80],
                    "status": result.get("status"),
                    "run_dir": result.get("run_dir"),
                    "manifest_path": result.get("manifest_path"),
                    "saved_files": result.get("saved_files", []),
                },
            },
            source="mcp-music-forge",
        )
        result["event"] = _append_or_raise(event)
        return result

    @mcp.tool()
    def continuum_music_generate_hf_space(
        space_id: str = "ACE-Step/Ace-Step-v1.5",
        prompt: str = "",
        payload_json: str = "",
        api_name: str = "/predict",
        title: str = "",
    ) -> dict[str, Any]:
        """Call a Hugging Face music Space and save returned audio files locally."""
        result = generate_hf_space_song(
            space_id=space_id,
            prompt=prompt,
            payload_json=payload_json,
            api_name=api_name,
            title=title,
        )
        event = _make_event(
            {
                "kind": "continuum.music.generated",
                "slot": "facilities",
                "payload": {
                    "space_id": space_id,
                    "title": title or prompt[:80],
                    "status": result.get("status"),
                    "run_dir": result.get("run_dir"),
                    "manifest_path": result.get("manifest_path"),
                    "saved_files": result.get("saved_files", []),
                },
            },
            source="mcp-music-forge",
        )
        result["event"] = _append_or_raise(event)
        return result

    @mcp.tool()
    def continuum_links() -> dict[str, Any]:
        """Read the registered peer/service link registry."""
        return {"status": "ok", "links": _peer_link_state()}

    @mcp.tool()
    def continuum_events(slot: str = ALL_SLOTS, limit: int = 25) -> dict[str, Any]:
        """Read recent Continuum events from a slot or from all slots."""
        return {"status": "ok", "slot": slot or ALL_SLOTS, "events": _recent_events(limit=limit, slot=slot or ALL_SLOTS)}

    @mcp.tool()
    def continuum_post_event(
        kind: str = "continuum.message",
        slot: str = "personal",
        text: str = "",
        payload_json: str = "",
        source: str = "mcp",
    ) -> dict[str, Any]:
        """Append a redacted local Continuum event for rooms, chat, or coordination."""
        payload = _json_object(payload_json)
        if text:
            payload.setdefault("text", text)
        event = _make_event({"kind": kind, "slot": slot, "payload": payload}, source=source or "mcp")
        return {"status": "ok", "event": _append_or_raise(event)}

    @mcp.tool()
    def continuum_heartbeat(
        component: str = "mcp-client",
        status: str = "ready",
        slot: str = "control",
        note: str = "",
        capabilities_json: str = "[]",
    ) -> dict[str, Any]:
        """Record liveness for a deck, agent, adapter, room, or peer Continuum."""
        try:
            capabilities = json.loads(capabilities_json or "[]")
        except json.JSONDecodeError:
            capabilities = []
        payload = {
            "component": component,
            "status": status,
            "slot": slot,
            "note": note,
            "capabilities": capabilities if isinstance(capabilities, list) else [],
        }
        event = _make_event(_build_heartbeat(payload), source="mcp-heartbeat")
        return {"status": "ok", "event": _append_or_raise(event)}

    @mcp.tool()
    def continuum_create_room(
        room_label: str = "Continuum conversation room",
        speaker_label: str = "Speaker A",
        listener_label: str = "Speaker B",
        source_lang: str = "auto",
        target_lang: str = "en-US",
        relationship_tone: str = "warm and clear",
    ) -> dict[str, Any]:
        """Create a local room session and return its slot and join paths."""
        room_code, room_payload = _build_room_session(
            {
                "room_label": room_label,
                "speaker_label": speaker_label,
                "listener_label": listener_label,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "relationship_tone": relationship_tone,
            }
        )
        event = _make_event(room_payload, source="mcp-room")
        return {
            "status": "ok",
            "room_code": room_code,
            "room_slot": event.get("slot"),
            "join_paths": event.get("payload", {}).get("join_paths", {}),
            "event": _append_or_raise(event),
        }

    @mcp.tool()
    def continuum_translate_packet(
        raw_message: str,
        target_language: str = "en-US",
        source_language: str = "auto",
        relationship_tone: str = "warm and clear",
    ) -> dict[str, Any]:
        """Build a local translation/cultural bridge packet without sending externally."""
        return build_translation_faculty_packet(
            raw_content=raw_message,
            source_lang=source_language,
            target_lang=target_language,
            conversation_profile=relationship_tone,
            glossary_text="",
            provider_plan="council-first",
        )

    @mcp.tool()
    def continuum_whatsapp_send_intent(
        to: str = "",
        text: str = "",
        payload_json: str = "",
    ) -> dict[str, Any]:
        """Draft a WhatsApp send intent. This does not send a WhatsApp message."""
        payload = _json_object(payload_json)
        if to:
            payload.setdefault("to", to)
        if text:
            payload.setdefault("text", text)
        event = _make_event(build_send_intent(payload), source="mcp-whatsapp-intent")
        return {"status": "ok", "event": _append_or_raise(event)}

    @mcp.tool()
    def continuum_wallet_intent(
        amount_sats: int = 0,
        memo: str = "",
        asset: str = "BTC",
        payload_json: str = "",
    ) -> dict[str, Any]:
        """Draft a wallet/payment intent. This does not move funds."""
        payload = _json_object(payload_json)
        payload.setdefault("amount_sats", amount_sats)
        payload.setdefault("memo", memo)
        payload.setdefault("asset", asset)
        event = _make_event(build_wallet_intent(payload), source="mcp-wallet-intent")
        return {"status": "ok", "event": _append_or_raise(event)}

    @mcp.tool()
    def continuum_remember(text: str, tags: str = "", kind: str = "note") -> dict[str, Any]:
        """Store a durable Continuum memory record."""
        tag_list = [item.strip() for item in (tags or "").split(",") if item.strip()]
        return _continuum().remember(text=text, kind=kind or "note", tags=tag_list)

    @mcp.tool()
    def continuum_search(query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search durable Continuum memory records."""
        return _continuum().search(query=query, limit=limit)

    @mcp.tool()
    def continuum_process_agent_text(text: str, max_tool_calls: int = 8) -> dict[str, Any]:
        """Execute [[continuum:]], [[tools:]], and [[tool:]] commands emitted by a tool-less agent."""
        return process_text(text, root=SHARED_STORE_ROOT, max_tool_calls=max_tool_calls)

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion Continuum MCP service.")
    parser.add_argument("--host", default=os.environ.get("CONTINUUM_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTINUUM_MCP_PORT", "7872")))
    parser.add_argument("--transport", choices=["sse", "streamable-http"], default=os.environ.get("CONTINUUM_MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    CHANNEL.mkdir(parents=True, exist_ok=True)
    SHARED_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    mcp = create_mcp(args.host, args.port)
    print(f"Champion Continuum MCP service listening on http://{args.host}:{args.port}/mcp/sse")
    mcp.run(args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
