from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from champion_continuum import Continuum, process_text
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
from continuum_provider_registry import provider_registry_state
from continuum_translation_faculty import build_translation_faculty_packet, translation_faculty_state
from continuum_whatsapp_adapter import build_send_intent, build_wallet_intent


SHARED_STORE_ROOT = Path(os.environ.get("CONTINUUM_SHARED_STORE", CHANNEL / "shared_store"))


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


def _append_or_raise(event: dict[str, Any]) -> dict[str, Any]:
    error = _append_event(event)
    if error:
        raise RuntimeError(f"event log write failed: {error}")
    return event


def create_mcp(host: str, port: int) -> FastMCP:
    mcp = FastMCP(
        "Champion Continuum MCP",
        instructions=(
            "Champion Continuum exposes memory, tool relay, room, event, provider, "
            "translation, WhatsApp-intent, wallet-intent, and receipt surfaces for "
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
