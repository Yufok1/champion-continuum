from __future__ import annotations

import hmac
import os
from hashlib import sha256
from typing import Any


DEFAULT_GRAPH_VERSION = "v25.0"
ASSET_REGISTRY: dict[str, dict[str, Any]] = {
    "BTC": {
        "symbol": "BTC",
        "role": "neutral_settlement_base",
        "zap_compatible": True,
        "rails": ["lightning", "nip57_zap_receipt", "nip47_nostr_wallet_connect", "walletconnect"],
        "custody": False,
        "note": "Use BTC/Lightning for zaps, receipts, access signals, and final settlement intents.",
    },
    "TPT": {
        "symbol": "TPT",
        "role": "tokenpocket_ecosystem_utility",
        "contract_address": "0xECa41281c24451168a37211F0bc2b8645AF45092",
        "zap_compatible": False,
        "rails": ["walletconnect"],
        "custody": False,
        "note": "Use TPT as the optional TokenPocket ecosystem utility or reputation/gating signal if any token is used, not as the base zap rail.",
    },
}


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _privacy_string(value: str, *, store_raw: bool = False) -> dict[str, Any]:
    value = str(value or "")
    out: dict[str, Any] = {
        "sha256": _hash_text(value),
        "length": len(value),
        "redacted": not store_raw,
    }
    if store_raw:
        out["value"] = value
    return out


def _privacy_identifier(value: str, *, store_identifier: bool = False) -> dict[str, Any]:
    value = str(value or "").strip()
    out: dict[str, Any] = {
        "sha256": _hash_text(value),
        "redacted": not store_identifier,
    }
    if store_identifier:
        out["value"] = value
    return out


def asset_registry_state() -> dict[str, Any]:
    return {
        "schema": "champion-continuum/asset-registry/v1",
        "assets": ASSET_REGISTRY,
        "whatsapp_native_token": None,
        "whatsapp_money_role": "messaging_channel_and_regional_payment_surface_only",
        "default_settlement_asset": "BTC",
        "preferred_ecosystem_token": "TPT",
        "operator_preferred_wallet": "TokenPocket",
        "financial_advice": False,
    }


def _asset_profile(symbol: str) -> dict[str, Any]:
    normalized = str(symbol or "BTC").strip().upper()
    profile = ASSET_REGISTRY.get(normalized)
    if profile:
        return dict(profile)
    return {
        "symbol": normalized or "UNKNOWN",
        "role": "external_wallet_asset",
        "zap_compatible": False,
        "rails": ["walletconnect"],
        "custody": False,
        "note": "Unverified wallet asset. Treat as external wallet intent only until researched.",
    }


def whatsapp_config_state(env: Any = os.environ) -> dict[str, Any]:
    """Return WhatsApp adapter readiness without exposing secrets."""
    access_token = str(env.get("WHATSAPP_ACCESS_TOKEN") or "")
    phone_number_id = str(env.get("WHATSAPP_PHONE_NUMBER_ID") or "")
    verify_token = str(env.get("WHATSAPP_VERIFY_TOKEN") or "")
    app_secret = str(env.get("WHATSAPP_APP_SECRET") or "")
    graph_version = str(env.get("WHATSAPP_GRAPH_VERSION") or DEFAULT_GRAPH_VERSION)
    return {
        "schema": "champion-continuum/whatsapp-config/v1",
        "official_path": "WhatsApp Business Cloud API",
        "graph_version": graph_version,
        "configured": bool(access_token and phone_number_id),
        "phone_number_id_configured": bool(phone_number_id),
        "access_token_configured": bool(access_token),
        "verify_token_configured": bool(verify_token),
        "app_secret_configured": bool(app_secret),
        "send_endpoint": f"https://graph.facebook.com/{graph_version}/<PHONE_NUMBER_ID>/messages",
        "webhook_endpoint": "/whatsapp/webhook",
        "secrets_exposed": False,
        "send_performed_by_adapter": False,
    }


def verify_meta_signature(raw_body: bytes, app_secret: str, signature_header: str | None) -> bool:
    if not app_secret:
        return False
    header = str(signature_header or "")
    if not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(app_secret.encode("utf-8"), raw_body, sha256).hexdigest()
    return hmac.compare_digest(expected, header)


def build_text_message(to: str, text: str, preview_url: bool = False) -> dict[str, Any]:
    return {
        "messaging_product": "whatsapp",
        "to": str(to or "").strip(),
        "type": "text",
        "text": {
            "preview_url": bool(preview_url),
            "body": str(text or ""),
        },
    }


def build_audio_message(to: str, media_id: str = "", media_link: str = "") -> dict[str, Any]:
    audio: dict[str, str]
    if str(media_id or "").strip():
        audio = {"id": str(media_id).strip()}
    elif str(media_link or "").strip():
        audio = {"link": str(media_link).strip()}
    else:
        raise ValueError("audio message requires media_id or media_link")
    return {
        "messaging_product": "whatsapp",
        "to": str(to or "").strip(),
        "type": "audio",
        "audio": audio,
    }


def build_send_intent(
    payload: dict[str, Any],
    *,
    store_raw: bool = False,
    store_identifiers: bool = False,
) -> dict[str, Any]:
    to = str(payload.get("to") or "").strip()
    text = str(payload.get("text") or "").strip()
    media_id = str(payload.get("audio_media_id") or payload.get("media_id") or "").strip()
    media_link = str(payload.get("audio_link") or payload.get("media_link") or "").strip()
    if media_id or media_link:
        message_payload = build_audio_message(to, media_id=media_id, media_link=media_link)
        intent_type = "audio"
    else:
        message_payload = build_text_message(to, text, preview_url=bool(payload.get("preview_url")))
        intent_type = "text"
    return {
        "kind": "whatsapp.outbound_intent",
        "slot": "whatsapp",
        "payload": {
            "adapter": "whatsapp_business_cloud_api",
            "intent_type": intent_type,
            "message_payload": message_payload if store_raw and store_identifiers else None,
            "recipient": _privacy_identifier(to, store_identifier=store_identifiers),
            "text": _privacy_string(text, store_raw=store_raw) if text else None,
            "audio_media_id": _privacy_identifier(media_id, store_identifier=store_identifiers) if media_id else None,
            "audio_link": _privacy_identifier(media_link, store_identifier=store_identifiers) if media_link else None,
            "source_event_id": str(payload.get("source_event_id") or ""),
            "operator_approval_required": True,
            "send_performed": False,
            "graph_request_ready": bool(to and (text or media_id or media_link)),
            "unredacted_runtime_payload_required_to_send": not (store_raw and store_identifiers),
        },
    }


def normalize_webhook(
    payload: dict[str, Any],
    *,
    store_raw: bool = False,
    store_identifiers: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") if isinstance(change.get("value"), dict) else {}
            metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
            contacts = {
                str(contact.get("wa_id") or ""): contact
                for contact in value.get("contacts") or []
                if isinstance(contact, dict)
            }
            for message in value.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                message_type = str(message.get("type") or "unknown")
                content: dict[str, Any] = {"type": message_type}
                if message_type == "text":
                    text = str((message.get("text") or {}).get("body") or "")
                    content["text"] = _privacy_string(text, store_raw=store_raw)
                elif message_type == "audio":
                    audio = message.get("audio") if isinstance(message.get("audio"), dict) else {}
                    content["audio"] = {
                        "id": _privacy_identifier(str(audio.get("id") or ""), store_identifier=store_identifiers),
                        "mime_type": str(audio.get("mime_type") or ""),
                        "sha256": str(audio.get("sha256") or ""),
                        "voice": bool(audio.get("voice")),
                    }
                    content["media_fetch_required"] = bool(audio.get("id"))
                else:
                    content["unsupported_message_type"] = True
                    content["raw_message"] = message if store_raw else None
                sender = str(message.get("from") or "")
                contact = contacts.get(sender) or {}
                events.append(
                    {
                        "kind": "whatsapp.inbound_message",
                        "slot": "whatsapp",
                        "payload": {
                            "adapter": "whatsapp_business_cloud_api",
                            "phone_number_id": str(metadata.get("phone_number_id") or ""),
                            "display_phone_number": str(metadata.get("display_phone_number") or ""),
                            "wa_message_id": str(message.get("id") or ""),
                            "from": _privacy_identifier(sender, store_identifier=store_identifiers),
                            "timestamp": str(message.get("timestamp") or ""),
                            "contact_profile_name": _privacy_string(
                                str((contact.get("profile") or {}).get("name") or ""),
                                store_raw=store_raw,
                            ) if contact else None,
                            "content": content,
                            "raw_logged": bool(store_raw),
                            "identifiers_logged": bool(store_identifiers),
                        },
                    }
                )
            for status in value.get("statuses") or []:
                if not isinstance(status, dict):
                    continue
                events.append(
                    {
                        "kind": "whatsapp.delivery_status",
                        "slot": "whatsapp",
                        "payload": {
                            "adapter": "whatsapp_business_cloud_api",
                            "phone_number_id": str(metadata.get("phone_number_id") or ""),
                            "wa_message_id": str(status.get("id") or ""),
                            "recipient_id": _privacy_identifier(
                                str(status.get("recipient_id") or ""),
                                store_identifier=store_identifiers,
                            ),
                            "status": str(status.get("status") or ""),
                            "timestamp": str(status.get("timestamp") or ""),
                            "conversation": status.get("conversation") or {},
                            "pricing": status.get("pricing") or {},
                            "identifiers_logged": bool(store_identifiers),
                        },
                    }
                )
    return events


def build_wallet_intent(
    payload: dict[str, Any],
    *,
    store_raw: bool = False,
    store_identifiers: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "wallet.payment_intent",
        "slot": "wallet",
        "payload": {
            "provider": str(payload.get("provider") or "TokenPocket"),
            "rail": str(payload.get("rail") or "walletconnect"),
            "asset": _asset_profile(str(payload.get("asset") or "BTC")),
            "amount_msats": payload.get("amount_msats"),
            "amount_sats": payload.get("amount_sats"),
            "amount_units": payload.get("amount_units"),
            "recipient": _privacy_identifier(str(payload.get("recipient") or ""), store_identifier=store_identifiers),
            "memo": _privacy_string(str(payload.get("memo") or ""), store_raw=store_raw),
            "source_event_id": str(payload.get("source_event_id") or ""),
            "operator_approval_required": True,
            "custody": False,
            "seed_storage": False,
            "private_key_storage": False,
            "funds_moved": False,
            "signing_performed": False,
        },
    }
