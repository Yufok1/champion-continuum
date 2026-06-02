from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import re
import secrets
import threading
import time
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from continuum_daemon_registry import load_daemon_registry, match_daemons
from continuum_whatsapp_adapter import (
    asset_registry_state,
    build_send_intent,
    build_wallet_intent,
    normalize_webhook,
    verify_meta_signature,
    whatsapp_config_state,
)
from continuum_provider_registry import provider_registry_state
from continuum_translation_faculty import translation_faculty_state


ROOT = Path(__file__).resolve().parent
CHANNEL = Path(os.environ.get("CONTINUUM_BRAIN_CHANNEL", ROOT / "cli_brain_channel"))
EVENT_LOG = CHANNEL / "continuum_link_events.jsonl"
PEER_LINKS_FILE = CHANNEL / "continuum_peer_links.json"
MAX_BODY_BYTES = 1_000_000
MAX_BACKLOG = 200
MAX_PEER_LINKS = 5
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:7870",
    "http://localhost:7870",
    "http://127.0.0.1:7871",
    "http://localhost:7871",
)
DEFAULT_SLOTS = (
    "personal",
    "voice",
    "whatsapp",
    "council",
    "business",
    "reputation",
    "wallet",
    "ipfs",
    "marketplace",
    "facilities",
    "control",
)
ALL_SLOTS = "*"
AUTH_TOKEN = os.environ.get("CONTINUUM_LINK_TOKEN") or secrets.token_urlsafe(32)
AUTH_TOKEN_SOURCE = "env" if os.environ.get("CONTINUUM_LINK_TOKEN") else "ephemeral"
RAW_TEXT_KEYS = {"raw", "raw_content", "raw_message", "body", "text", "message", "memo"}
IDENTIFIER_KEYS = {
    "to",
    "from",
    "phone",
    "phone_number",
    "recipient",
    "recipient_id",
    "wa_id",
    "display_phone_number",
    "audio_media_id",
    "media_id",
    "audio_link",
    "media_link",
}

STARTED_AT = time.time()
SUBSCRIBERS: dict[queue.Queue[dict[str, Any]], str] = {}
SUBSCRIBERS_LOCK = threading.RLock()
EVENT_LOCK = threading.RLock()
PEER_LINKS_LOCK = threading.RLock()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _hash_string(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _redacted_string(value: str, category: str) -> dict[str, Any]:
    value = str(value or "")
    return {
        "redacted": True,
        "category": category,
        "sha256": _hash_string(value),
        "length": len(value),
    }


def _store_raw_content() -> bool:
    return _bool_env("CONTINUUM_LINK_STORE_RAW", False)


def _store_identifiers() -> bool:
    return _bool_env("CONTINUUM_LINK_STORE_IDENTIFIERS", False)


def _privacy_scrub(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if isinstance(value, dict):
        return {str(k): _privacy_scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_privacy_scrub(item, key) for item in value]
    if isinstance(value, str):
        if lowered in RAW_TEXT_KEYS and not _store_raw_content():
            return _redacted_string(value, "raw_text")
        if lowered in IDENTIFIER_KEYS and not _store_identifiers():
            return _redacted_string(value, "identifier")
    return value


def _allowed_origins() -> set[str]:
    configured = os.environ.get("CONTINUUM_LINK_ALLOWED_ORIGINS")
    if configured:
        return {origin.strip() for origin in configured.split(",") if origin.strip()}
    return set(DEFAULT_ALLOWED_ORIGINS)


def _auth_digest() -> str:
    return _hash_string(AUTH_TOKEN)[:16]


def _auth_state() -> dict[str, Any]:
    return {
        "required": True,
        "token_source": AUTH_TOKEN_SOURCE,
        "token_sha256_prefix": _auth_digest(),
        "accepted": ["Authorization: Bearer <token>", "X-Continuum-Token: <token>", "token query parameter for EventSource"],
        "secrets_exposed": False,
    }


def _webhook_signature_required(host: str) -> bool:
    return _bool_env("CONTINUUM_REQUIRE_WHATSAPP_SIGNATURE", host not in LOCAL_HOSTS)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_slot(value: Any) -> str:
    slot = str(value or "personal").strip().lower()
    slot = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in slot)
    slot = slot.strip("-_")
    return slot or "personal"


def _event_slot(event: dict[str, Any]) -> str:
    return _normalize_slot(event.get("slot") or "personal")


def _make_event(payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
    created_ms = int(payload.get("created_ms") or _now_ms())
    kind = str(payload.get("kind") or "continuum.event")
    slot = _normalize_slot(payload.get("slot") or "personal")
    body = payload.get("payload")
    if body is None:
        body = {
            key: value
            for key, value in payload.items()
            if key not in {"schema", "event_id", "created_ms", "kind", "source", "slot"}
        }
    body = _privacy_scrub(body)
    seed = {
        "created_ms": created_ms,
        "kind": kind,
        "source": payload.get("source") or source,
        "slot": slot,
        "payload": body,
    }
    event_id = str(payload.get("event_id") or ("clink_" + sha256(_canonical_json(seed).encode("utf-8")).hexdigest()[:20]))
    return {
        "schema": "champion-continuum/link-event/v1",
        "event_id": event_id,
        "created_ms": created_ms,
        "kind": kind,
        "slot": slot,
        "source": str(payload.get("source") or source),
        "payload": body,
    }


def _subscriber_allows(subscription_slot: str, event: dict[str, Any]) -> bool:
    return subscription_slot == ALL_SLOTS or subscription_slot == _event_slot(event)


def _append_event(event: dict[str, Any]) -> str | None:
    try:
        CHANNEL.mkdir(parents=True, exist_ok=True)
        with EVENT_LOCK:
            with EVENT_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    with SUBSCRIBERS_LOCK:
        dead: list[queue.Queue[dict[str, Any]]] = []
        for subscriber, subscription_slot in SUBSCRIBERS.items():
            if not _subscriber_allows(subscription_slot, event):
                continue
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                dead.append(subscriber)
        for subscriber in dead:
            SUBSCRIBERS.pop(subscriber, None)
    return None


def _recent_events(limit: int = 50, slot: str = ALL_SLOTS) -> list[dict[str, Any]]:
    if not EVENT_LOG.exists():
        return []
    limit = max(1, min(int(limit or 50), MAX_BACKLOG))
    subscription_slot = ALL_SLOTS if slot == ALL_SLOTS else _normalize_slot(slot)
    try:
        lines = EVENT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and _subscriber_allows(subscription_slot, payload):
            out.append(payload)
            if len(out) >= limit:
                break
    out.reverse()
    return out


def _slot_counts() -> dict[str, int]:
    counts: dict[str, int] = {slot: 0 for slot in DEFAULT_SLOTS}
    if not EVENT_LOG.exists():
        return counts
    try:
        lines = EVENT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return counts
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            slot = _event_slot(payload)
            counts[slot] = counts.get(slot, 0) + 1
    return counts


def _load_peer_links() -> list[dict[str, Any]]:
    with PEER_LINKS_LOCK:
        try:
            payload = json.loads(PEER_LINKS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    links = payload.get("links") if isinstance(payload, dict) else payload
    if not isinstance(links, list):
        return []
    out: list[dict[str, Any]] = []
    for item in links[:MAX_PEER_LINKS]:
        if isinstance(item, dict):
            out.append(item)
    return out


def _save_peer_links(links: list[dict[str, Any]]) -> None:
    CHANNEL.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "champion-continuum/peer-links/v1",
        "updated_ms": _now_ms(),
        "max_links": MAX_PEER_LINKS,
        "links": links[:MAX_PEER_LINKS],
    }
    with PEER_LINKS_LOCK:
        PEER_LINKS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _peer_link_state() -> dict[str, Any]:
    links = _load_peer_links()
    return {
        "schema": "champion-continuum/peer-links/v1",
        "mode": "metadata_registry",
        "max_links": MAX_PEER_LINKS,
        "count": len(links),
        "links": links,
        "note": "Peer SSE links are registered targets. This service does not auto-dial them yet.",
        "auth": {
            "local_link_token_required": True,
            "remote_token_storage": "hash_only_by_default",
        },
        "channels": {
            "whatsapp": "conversation channel and webhook adapter",
            "google": "future OAuth/contact/calendar/identity lane; not required for local rooms",
        },
    }


def _build_peer_link(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("label") or payload.get("name") or "Peer Continuum").strip()[:80]
    url = str(payload.get("url") or payload.get("sse_url") or "").strip()
    slot = _normalize_slot(payload.get("slot") or payload.get("default_slot") or "personal")
    auth_hint = str(payload.get("auth_hint") or "remote_token_required").strip()[:80]
    token = str(payload.get("token") or payload.get("remote_token") or "").strip()
    token_sha = _hash_string(token) if token else ""
    link_id_seed = {"label": label, "url": url, "slot": slot}
    link_id = str(payload.get("link_id") or ("peer_" + sha256(_canonical_json(link_id_seed).encode("utf-8")).hexdigest()[:12]))
    return {
        "schema": "champion-continuum/peer-link/v1",
        "link_id": link_id,
        "label": label,
        "url": url,
        "default_slot": slot,
        "auth_hint": auth_hint,
        "token_sha256": token_sha,
        "token_present": bool(token),
        "enabled": bool(payload.get("enabled", True)),
        "created_ms": _now_ms(),
        "last_seen_ms": 0,
        "capabilities": [str(item) for item in payload.get("capabilities", [])[:32]] if isinstance(payload.get("capabilities"), list) else [],
        "external_connection_opened": False,
    }


def _state() -> dict[str, Any]:
    with SUBSCRIBERS_LOCK:
        subscriber_count = len(SUBSCRIBERS)
        subscriber_slots = dict.fromkeys(DEFAULT_SLOTS, 0)
        subscriber_slots[ALL_SLOTS] = 0
        for slot in SUBSCRIBERS.values():
            subscriber_slots[slot] = int(subscriber_slots.get(slot, 0)) + 1
    recent = _recent_events(limit=5)
    return {
        "status": "ok",
        "service": "champion-continuum-link",
        "schema": "champion-continuum/link-state/v1",
        "root": str(ROOT),
        "channel": str(CHANNEL),
        "event_log": str(EVENT_LOG),
        "uptime_seconds": round(time.time() - STARTED_AT, 3),
        "auth": _auth_state(),
        "privacy": {
            "store_raw_content": _store_raw_content(),
            "store_identifiers": _store_identifiers(),
            "default": "hash_and_length_only",
        },
        "subscriber_count": subscriber_count,
        "subscriber_slots": subscriber_slots,
        "slots": list(DEFAULT_SLOTS),
        "slot_counts": _slot_counts(),
        "peer_links": _peer_link_state(),
        "utility_daemons": load_daemon_registry(CHANNEL),
        "recent_event_ids": [str(event.get("event_id") or "") for event in recent],
        "endpoints": {
            "health": "/health",
            "state": "/state",
            "settings": "/settings",
            "slots": "/slots",
            "faculties": "/faculties",
            "providers": "/providers",
            "heartbeat": "/heartbeat",
            "daemons": "/daemons",
            "daemon_match": "/daemons/match?capability=translation&output=text",
            "links": "/links",
            "link_register": "/link/register",
            "events": "/events?slot=personal&limit=50",
            "sse": "/sse?slot=personal",
            "sse_all": "/sse?slot=*",
            "post_event": "/event",
            "room_create": "/room/create",
            "whatsapp_config": "/whatsapp/config",
            "whatsapp_webhook": "/whatsapp/webhook",
            "whatsapp_send_intent": "/whatsapp/send-intent",
            "wallet_intent": "/wallet/intent",
            "assets": "/assets",
            "council_intent": "/council/intent",
            "business_intent": "/business/intent",
            "ipfs_intent": "/ipfs/intent",
        },
        "adapter_posture": {
            "translation_faculty": translation_faculty_state(),
            "whatsapp": whatsapp_config_state()
            | {
                "mode": "adapter_slot",
                "slot": "whatsapp",
                "official_path": "WhatsApp Business Cloud API",
                "secret_storage": "operator_supplied_environment_or_local_settings",
                "send_requires_operator_approval": True,
            },
            "wallet": {
                "mode": "non_custodial_intent_slot",
                "slot": "wallet",
                "bitcoin_rail": "Lightning via NIP-47/NWC or NIP-57 zap receipts",
                "custody": False,
                "seed_storage": False,
                "funds_move_without_operator": False,
            },
        },
        "nostr_posture": {
            "mode": "continuum_native_local_first",
            "relay_published": False,
            "signing_required_for_local_drafts": False,
            "note": "Nostr relay/signing is a later layer; this service exposes the local Continuum event stream.",
        },
        "ipfs_posture": {
            "mode": "optional_archive_intent",
            "slot": "ipfs",
            "local_pin_possible": True,
            "remote_pin_required": False,
            "cid_claimed": False,
            "paid_pinning_required": False,
            "note": "IPFS is free as a protocol. Persistence requires local pinning, peer pinning, or a pinning provider.",
        },
    }


def _settings_state() -> dict[str, Any]:
    state = _state()
    return {
        "status": "ok",
        "schema": "champion-continuum/link-settings/v1",
        "service": state["service"],
        "root": state["root"],
        "channel": state["channel"],
        "mode": {
            "server": "single_local_link_service",
            "slots": state["slots"],
            "peer_link_capacity": _peer_link_state()["max_links"],
            "peer_link_count": _peer_link_state()["count"],
        },
        "auth": state["auth"],
        "privacy": state["privacy"],
        "providers": provider_registry_state(),
        "faculties": translation_faculty_state(),
        "utility_daemons": load_daemon_registry(CHANNEL),
        "peer_links": _peer_link_state(),
        "endpoints": state["endpoints"],
        "operator_notes": [
            "One local link service can stream many slots.",
            "Up to five peer SSE targets can be registered as metadata.",
            "Remote peer tokens are represented by hashes by default.",
            "External sends, wallet movement, relay publish, and IPFS pinning remain approval-gated.",
        ],
    }


def _new_room_code() -> str:
    return re.sub(r"[^A-Za-z0-9]", "", secrets.token_urlsafe(8)).upper()[:10]


def _build_room_session(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    room_code = str(payload.get("room_code") or _new_room_code()).strip().upper()
    room_code = re.sub(r"[^A-Z0-9]", "", room_code)[:16] or _new_room_code()
    room_slot = _normalize_slot(payload.get("slot") or f"room-{room_code[:8].lower()}")
    languages = payload.get("languages") if isinstance(payload.get("languages"), dict) else {}
    requested_slots = payload.get("slots") if isinstance(payload.get("slots"), list) else ["personal", "voice", "whatsapp"]
    return room_code, {
        "kind": "continuum.room_session",
        "slot": room_slot,
        "payload": {
            "schema": "champion-continuum/room-session/v1",
            "room_code_sha256": _hash_string(room_code),
            "room_slot": room_slot,
            "room_label": _privacy_scrub(str(payload.get("room_label") or "Continuum conversation room"), "raw_message"),
            "speaker_label": _privacy_scrub(str(payload.get("speaker_label") or "Speaker A"), "recipient"),
            "listener_label": _privacy_scrub(str(payload.get("listener_label") or "Speaker B"), "recipient"),
            "source_lang": str(languages.get("source") or payload.get("source_lang") or "auto"),
            "target_lang": str(languages.get("target") or payload.get("target_lang") or "en-US"),
            "relationship_tone": str(payload.get("relationship_tone") or "warm and clear"),
            "allowed_slots": [_normalize_slot(slot) for slot in requested_slots],
            "join_paths": {
                "sse": f"/sse?slot={room_slot}",
                "events": f"/events?slot={room_slot}&limit=50",
                "auth": "Continuum link token required",
            },
            "message_sent": False,
            "public_invite_published": False,
            "operator_approval_required": True,
        },
    }


def _build_heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        capabilities = []
    return {
        "kind": "continuum.heartbeat",
        "slot": _normalize_slot(payload.get("slot") or "control"),
        "payload": {
            "schema": "champion-continuum/heartbeat/v1",
            "component": str(payload.get("component") or "continuum-client"),
            "status": str(payload.get("status") or "ready"),
            "ttl_seconds": int(payload.get("ttl_seconds") or 30),
            "capabilities": [str(item) for item in capabilities[:32]],
            "note": _privacy_scrub(str(payload.get("note") or ""), "raw_message"),
            "external_effects_performed": False,
        },
    }


class ContinuumLinkHandler(BaseHTTPRequestHandler):
    server_version = "ChampionContinuumLink/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[continuum-link] {self.client_address[0]} - {format % args}")

    def _origin(self) -> str:
        return str(self.headers.get("Origin") or "").strip()

    def _origin_allowed(self) -> bool:
        origin = self._origin()
        return not origin or origin in _allowed_origins()

    def _set_cors_headers(self) -> None:
        origin = self._origin()
        if origin and origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        elif not origin:
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:7870")
            self.send_header("Vary", "Origin")

    def _auth_token_from_request(self, parsed: Any | None = None) -> str:
        auth = str(self.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        header_token = str(self.headers.get("X-Continuum-Token") or "").strip()
        if header_token:
            return header_token
        if parsed is not None:
            query_token = (parse_qs(parsed.query).get("token") or [""])[0]
            if query_token:
                return query_token
        return ""

    def _authorized(self, parsed: Any | None = None) -> bool:
        return hmac.compare_digest(self._auth_token_from_request(parsed), AUTH_TOKEN)

    def _send_auth_error(self) -> None:
        self._send_json({"status": "error", "error": "unauthorized", "auth": _auth_state()}, status=401)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._set_cors_headers()
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization, x-continuum-token, x-hub-signature-256")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def _send_sse(self, event: dict[str, Any], event_name: str = "continuum") -> None:
        event_id = str(event.get("event_id") or "")
        data = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"id: {event_id}\n".encode("utf-8"))
        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
        for line in data.splitlines() or [""]:
            self.wfile.write(f"data: {line}\n".encode("utf-8"))
        self.wfile.write(b"\n")
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        if not self._origin_allowed():
            self._send_json({"status": "error", "error": "origin_not_allowed"}, status=403)
            return
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization, x-continuum-token, x-hub-signature-256")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not self._origin_allowed():
            self._send_json({"status": "error", "error": "origin_not_allowed"}, status=403)
            return
        if parsed.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "champion-continuum-link",
                    "uptime_seconds": time.time() - STARTED_AT,
                    "auth_required": True,
                    "token_sha256_prefix": _auth_digest(),
                }
            )
            return
        if parsed.path != "/whatsapp/webhook" and not self._authorized(parsed):
            self._send_auth_error()
            return
        if parsed.path == "/state":
            self._send_json(_state())
            return
        if parsed.path == "/settings":
            self._send_json(_settings_state())
            return
        if parsed.path == "/slots":
            self._send_json({"status": "ok", "slots": list(DEFAULT_SLOTS), "slot_counts": _slot_counts()})
            return
        if parsed.path == "/faculties":
            self._send_json({"status": "ok", "translation_faculty": translation_faculty_state()})
            return
        if parsed.path == "/providers":
            self._send_json({"status": "ok", "provider_registry": provider_registry_state()})
            return
        if parsed.path == "/daemons":
            self._send_json(load_daemon_registry(CHANNEL))
            return
        if parsed.path == "/daemons/match":
            query = parse_qs(parsed.query)
            capability = (query.get("capability") or [""])[0]
            output = (query.get("output") or [""])[0]
            include_stale = (query.get("include_stale") or ["0"])[0].lower() in {"1", "true", "yes", "on"}
            self._send_json(match_daemons(capability=capability, output=output, include_stale=include_stale))
            return
        if parsed.path == "/links":
            self._send_json({"status": "ok", "peer_links": _peer_link_state()})
            return
        if parsed.path == "/assets":
            self._send_json({"status": "ok", "registry": asset_registry_state()})
            return
        if parsed.path == "/whatsapp/config":
            self._send_json({"status": "ok", "config": whatsapp_config_state()})
            return
        if parsed.path == "/whatsapp/webhook":
            query = parse_qs(parsed.query)
            mode = (query.get("hub.mode") or [""])[0]
            challenge = (query.get("hub.challenge") or [""])[0]
            token = (query.get("hub.verify_token") or [""])[0]
            expected = os.environ.get("WHATSAPP_VERIFY_TOKEN") or ""
            if mode == "subscribe" and expected and hmac.compare_digest(token, expected):
                raw = challenge.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json({"status": "error", "error": "webhook_verification_failed"}, status=403)
            return
        if parsed.path == "/events":
            query = parse_qs(parsed.query)
            limit = int((query.get("limit") or ["50"])[0])
            slot = (query.get("slot") or [ALL_SLOTS])[0]
            self._send_json({"status": "ok", "slot": slot, "events": _recent_events(limit=limit, slot=slot)})
            return
        if parsed.path == "/sse":
            query = parse_qs(parsed.query)
            slot = (query.get("slot") or ["personal"])[0]
            subscription_slot = ALL_SLOTS if slot == ALL_SLOTS else _normalize_slot(slot)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors_headers()
            self.end_headers()
            subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=MAX_BACKLOG)
            with SUBSCRIBERS_LOCK:
                SUBSCRIBERS[subscriber] = subscription_slot
            try:
                self.wfile.write(b": champion-continuum-link connected\n\n")
                self.wfile.flush()
                self._send_sse(
                    _make_event(
                        {"kind": "continuum.state", "slot": "control", "payload": _state()},
                        source="link-server",
                    ),
                    "state",
                )
                for event in _recent_events(limit=25, slot=subscription_slot):
                    self._send_sse(event)
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    self._send_sse(event)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with SUBSCRIBERS_LOCK:
                    SUBSCRIBERS.pop(subscriber, None)
            return
        self._send_json({"status": "error", "error": "not_found", "path": parsed.path}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._origin_allowed():
            self._send_json({"status": "error", "error": "origin_not_allowed"}, status=403)
            return
        if parsed.path not in {
            "/event",
            "/heartbeat",
            "/link/register",
            "/room/create",
            "/whatsapp/webhook",
            "/whatsapp/send-intent",
            "/wallet/intent",
            "/council/intent",
            "/business/intent",
            "/ipfs/intent",
        }:
            self._send_json({"status": "error", "error": "not_found", "path": parsed.path}, status=404)
            return
        if parsed.path != "/whatsapp/webhook" and not self._authorized(parsed):
            self._send_auth_error()
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json({"status": "error", "error": "empty_body"}, status=400)
            return
        if length > MAX_BODY_BYTES:
            self._send_json({"status": "error", "error": "body_too_large", "max_bytes": MAX_BODY_BYTES}, status=413)
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json({"status": "error", "error": "invalid_json", "detail": str(exc)}, status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"status": "error", "error": "object_required"}, status=400)
            return
        if parsed.path == "/whatsapp/webhook":
            app_secret = os.environ.get("WHATSAPP_APP_SECRET") or ""
            signature_required = bool(getattr(self.server, "require_whatsapp_signature", False))
            signature_checked = bool(app_secret)
            signature_valid = (
                verify_meta_signature(raw, app_secret, self.headers.get("X-Hub-Signature-256"))
                if signature_checked
                else None
            )
            if signature_required and not app_secret:
                self._send_json({"status": "error", "error": "whatsapp_app_secret_required"}, status=503)
                return
            if signature_checked and not signature_valid:
                self._send_json({"status": "error", "error": "invalid_meta_signature"}, status=403)
                return
            events = [
                _make_event(item, source="whatsapp-webhook")
                for item in normalize_webhook(
                    payload,
                    store_raw=_store_raw_content(),
                    store_identifiers=_store_identifiers(),
                )
            ]
            append_errors: list[str] = []
            for event in events:
                append_error = _append_event(event)
                if append_error:
                    append_errors.append(append_error)
            if append_errors:
                self._send_json({"status": "error", "error": "event_log_write_failed", "details": append_errors}, status=500)
                return
            self._send_json(
                {
                    "status": "ok",
                    "received": len(events),
                    "signature_required": signature_required,
                    "signature_checked": signature_checked,
                    "signature_valid": signature_valid,
                    "events": events,
                },
                status=202,
            )
            return
        if parsed.path == "/heartbeat":
            event = _make_event(_build_heartbeat(payload), source="heartbeat")
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        if parsed.path == "/link/register":
            peer = _build_peer_link(payload)
            if not peer["url"].startswith(("http://", "https://")):
                self._send_json({"status": "error", "error": "valid_sse_url_required"}, status=400)
                return
            links = [item for item in _load_peer_links() if item.get("link_id") != peer["link_id"]]
            if len(links) >= MAX_PEER_LINKS:
                self._send_json({"status": "error", "error": "max_peer_links_reached", "max_links": MAX_PEER_LINKS}, status=409)
                return
            links.append(peer)
            _save_peer_links(links)
            event = _make_event(
                {
                    "kind": "continuum.peer_link.registered",
                    "slot": "control",
                    "payload": {
                        "link_id": peer["link_id"],
                        "label": peer["label"],
                        "url": peer["url"],
                        "default_slot": peer["default_slot"],
                        "token_present": peer["token_present"],
                        "external_connection_opened": False,
                    },
                },
                source="peer-link",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "peer_link": peer, "event": event}, status=201)
            return
        if parsed.path == "/room/create":
            room_code, room_payload = _build_room_session(payload)
            event = _make_event(room_payload, source="room-session")
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json(
                {
                    "status": "ok",
                    "room_code": room_code,
                    "room_slot": event.get("slot"),
                    "join_paths": event.get("payload", {}).get("join_paths", {}),
                    "event": event,
                },
                status=201,
            )
            return
        if parsed.path == "/whatsapp/send-intent":
            event = _make_event(
                build_send_intent(
                    payload,
                    store_raw=_store_raw_content(),
                    store_identifiers=_store_identifiers(),
                ),
                source="whatsapp-intent",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        if parsed.path == "/wallet/intent":
            event = _make_event(
                build_wallet_intent(
                    payload,
                    store_raw=_store_raw_content(),
                    store_identifiers=_store_identifiers(),
                ),
                source="wallet-intent",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        if parsed.path == "/council/intent":
            event = _make_event(
                {
                    "kind": str(payload.get("kind") or "council.agent_intent"),
                    "slot": "council",
                    "payload": {
                        "agent": str(payload.get("agent") or "operator-selected-council"),
                        "task": _privacy_scrub(str(payload.get("task") or ""), "raw_message"),
                        "capability": str(payload.get("capability") or "observe_draft_review"),
                        "requested_autonomy": str(payload.get("requested_autonomy") or "draft_only"),
                        "allowed_external_effects": False,
                        "operator_approval_required": True,
                        "merkle_receipt_required": True,
                        "cascade_lattice_receipt_required": True,
                        "result": _privacy_scrub(payload.get("result") or {}, "result"),
                    },
                },
                source="council-intent",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        if parsed.path == "/business/intent":
            requested_lanes = payload.get("lanes")
            if not isinstance(requested_lanes, list):
                requested_lanes = ["whatsapp", "wallet", "reputation", "ipfs"]
            event = _make_event(
                {
                    "kind": "business.crypto_directive_intent",
                    "slot": "business",
                    "payload": {
                        "directive": _privacy_scrub(str(payload.get("directive") or ""), "raw_message"),
                        "agent": str(payload.get("agent") or "operator-selected-council"),
                        "lanes": [str(lane) for lane in requested_lanes],
                        "default_settlement_asset": "BTC",
                        "preferred_ecosystem_token": "TPT",
                        "whatsapp_role": "conversation_channel",
                        "wallet_role": "operator_approved_signing_and_sats_transfer",
                        "reputation_role": "receipt_and_trust_weight",
                        "ipfs_role": "optional_redacted_receipt_archive",
                        "marketplace_role": "optional_listing_or_service_offer",
                        "operator_approval_required": True,
                        "external_effects_performed": False,
                        "funds_moved": False,
                        "message_sent": False,
                        "relay_published": False,
                        "ipfs_pinned": False,
                        "next_drafts": {
                            "wallet_intent": "/wallet/intent",
                            "whatsapp_send_intent": "/whatsapp/send-intent",
                            "ipfs_intent": "/ipfs/intent",
                            "council_intent": "/council/intent",
                        },
                    },
                },
                source="business-intent",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        if parsed.path == "/ipfs/intent":
            event = _make_event(
                {
                    "kind": "ipfs.archive_intent",
                    "slot": "ipfs",
                    "payload": {
                        "cid": str(payload.get("cid") or ""),
                        "content_sha256": str(payload.get("content_sha256") or ""),
                        "content_label": _privacy_scrub(str(payload.get("content_label") or ""), "raw_message"),
                        "local_pin_requested": bool(payload.get("local_pin_requested", True)),
                        "remote_pin_requested": bool(payload.get("remote_pin_requested", False)),
                        "provider": str(payload.get("provider") or "local_ipfs_node"),
                        "paid_provider_required": False,
                        "operator_approval_required": True,
                        "archive_performed": False,
                    },
                },
                source="ipfs-intent",
            )
            append_error = _append_event(event)
            if append_error:
                self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
                return
            self._send_json({"status": "ok", "event": event}, status=201)
            return
        event = _make_event(payload, source="http")
        append_error = _append_event(event)
        if append_error:
            self._send_json({"status": "error", "error": "event_log_write_failed", "detail": append_error}, status=500)
            return
        self._send_json({"status": "ok", "event": event}, status=201)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Champion Continuum SSE/link service.")
    parser.add_argument("--host", default=os.environ.get("CONTINUUM_LINK_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTINUUM_LINK_PORT", "7871")))
    args = parser.parse_args()

    CHANNEL.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ContinuumLinkHandler)
    server.require_whatsapp_signature = _webhook_signature_required(args.host)
    print(f"Champion Continuum link service listening on http://{args.host}:{args.port}/sse")
    print(f"Continuum link auth token source: {AUTH_TOKEN_SOURCE}; sha256 prefix: {_auth_digest()}")
    if AUTH_TOKEN_SOURCE == "ephemeral":
        print("Set CONTINUUM_LINK_TOKEN to a stable local secret before using external adapters.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Champion Continuum link service stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
