from __future__ import annotations

import os
from typing import Any


HF_PROVIDER_PREFIX = "hf-provider:"
HF_DEFAULT_PROVIDER = os.environ.get("CONTINUUM_HF_PROVIDER", "auto")
HF_DEFAULT_MODEL = os.environ.get("CONTINUUM_HF_PROVIDER_MODEL", "openai/gpt-oss-120b")


def provider_registry_state() -> dict[str, Any]:
    return {
        "schema": "champion-continuum/provider-registry/v1",
        "resident_space_model": {
            "mode": "single_loaded_model",
            "role": "always_on_relevance_scout",
            "reason": "Keep one resident model loaded for cheap observation, routing, first drafts, and small-but-useful ideas.",
            "principle": "Relevance wins over raw model size; escalate only when the task needs depth or external expertise.",
        },
        "huggingface_inference_providers": {
            "mode": "optional_remote_provider",
            "model_selector_prefix": HF_PROVIDER_PREFIX,
            "default_provider": HF_DEFAULT_PROVIDER,
            "default_model": HF_DEFAULT_MODEL,
            "token_env": ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
            "supports": [
                "chat_completion",
                "provider_auto_routing",
                "provider_specific_routing",
                "speech_or_embedding_lanes_via_future_adapter",
            ],
            "external_effects_performed": False,
        },
        "external_agents": {
            "mode": "heartbeat_and_forum_lane",
            "connectors": [
                {
                    "id": "codex-cli",
                    "kind": "cli_or_ide_agent",
                    "transport": "forum_daemon_or_link_heartbeat",
                    "default_role": "engineering_operator",
                },
                {
                    "id": "claude-cli",
                    "kind": "cli_or_ide_agent",
                    "transport": "forum_daemon_or_link_heartbeat",
                    "default_role": "relationship_and_language_editor",
                },
                {
                    "id": "gemini-cli",
                    "kind": "cli_or_ide_agent",
                    "transport": "forum_daemon_or_link_heartbeat",
                    "default_role": "auditor_and_cultural_lens",
                },
                {
                    "id": "ide-agent",
                    "kind": "ide_assistant",
                    "transport": "link_heartbeat_plus_operator_paste_or_future_rpc",
                    "default_role": "workspace_specialist",
                },
            ],
        },
    }


def hf_provider_model_id(provider: str | None = None, model: str | None = None) -> str:
    provider = (provider or HF_DEFAULT_PROVIDER or "auto").strip()
    model = (model or HF_DEFAULT_MODEL or "").strip()
    return f"{HF_PROVIDER_PREFIX}{provider}:{model}"


def parse_provider_model_id(model_id: str) -> tuple[str, str] | None:
    if not str(model_id or "").startswith(HF_PROVIDER_PREFIX):
        return None
    rest = str(model_id)[len(HF_PROVIDER_PREFIX):]
    provider, sep, model = rest.partition(":")
    if not sep:
        return HF_DEFAULT_PROVIDER, rest
    return provider or HF_DEFAULT_PROVIDER, model or HF_DEFAULT_MODEL


def _token() -> str:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(_message_text(item) for item in message)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(_message_text(item) for item in content)
        text = message.get("text")
        if isinstance(text, str):
            return text
    return str(message or "")


def _extract_completion_text(result: Any) -> str:
    try:
        choices = getattr(result, "choices", None) or result.get("choices")  # type: ignore[union-attr]
        if choices:
            first = choices[0]
            message = getattr(first, "message", None) or first.get("message")
            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                return _message_text(content)
            text = getattr(first, "text", None) or first.get("text")
            if text:
                return str(text)
    except Exception:
        pass
    try:
        generated = getattr(result, "generated_text", None) or result.get("generated_text")  # type: ignore[union-attr]
        if generated:
            return str(generated)
    except Exception:
        pass
    return _message_text(result)


def run_hf_provider_chat(model_id: str, messages: list[dict[str, Any]], max_tokens: int = 900) -> str:
    parsed = parse_provider_model_id(model_id)
    if parsed is None:
        raise ValueError("not an HF provider model id")
    provider, model = parsed
    token = _token()
    if not token:
        raise RuntimeError("HF Inference Providers require HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub InferenceClient unavailable: {type(exc).__name__}: {exc}") from exc
    client = InferenceClient(model=model, provider=provider or "auto", token=token, timeout=90)
    prepared = [{"role": str(m.get("role") or "user"), "content": _message_text(m.get("content"))} for m in messages]
    if hasattr(client, "chat_completion"):
        result = client.chat_completion(messages=prepared, max_tokens=max_tokens)
        return _extract_completion_text(result)
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    create = getattr(completions, "create", None) if completions is not None else None
    if create is None:
        raise RuntimeError("Installed huggingface_hub does not expose chat completion helpers.")
    result = create(model=model, messages=prepared, max_tokens=max_tokens)
    return _extract_completion_text(result)
