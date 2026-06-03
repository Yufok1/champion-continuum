from __future__ import annotations

import os
from typing import Any


HF_PROVIDER_PREFIX = "hf-provider:"
HF_DEFAULT_PROVIDER = os.environ.get("CONTINUUM_HF_PROVIDER", "auto")
HF_DEFAULT_MODEL = os.environ.get("CONTINUUM_HF_PROVIDER_MODEL", "openai/gpt-oss-120b")


HF_INFERENCE_PROVIDER_CATALOG: list[dict[str, Any]] = [
    {"id": "cerebras", "label": "Cerebras", "strengths": ["chat_completion_llm"]},
    {"id": "cohere", "label": "Cohere", "strengths": ["chat_completion_llm", "feature_extraction"]},
    {"id": "deepinfra", "label": "DeepInfra", "strengths": ["chat_completion_llm", "multimodal_or_media"]},
    {"id": "fal-ai", "label": "Fal AI", "strengths": ["text_to_image", "text_to_video", "speech_to_text", "image_tools"]},
    {"id": "featherless-ai", "label": "Featherless AI", "strengths": ["chat_completion_llm", "chat_completion_vlm"]},
    {"id": "fireworks-ai", "label": "Fireworks", "strengths": ["chat_completion_llm", "chat_completion_vlm"]},
    {"id": "groq", "label": "Groq", "strengths": ["chat_completion_llm", "speech_to_text"]},
    {"id": "hf-inference", "label": "HF Inference", "strengths": ["chat_completion_llm", "feature_extraction", "text_to_image", "text_to_video", "speech_to_text"]},
    {"id": "hyperbolic", "label": "Hyperbolic", "strengths": ["chat_completion_llm", "chat_completion_vlm"]},
    {"id": "novita", "label": "Novita", "strengths": ["chat_completion_llm", "chat_completion_vlm", "text_to_image"]},
    {"id": "nscale", "label": "Nscale", "strengths": ["chat_completion_llm", "chat_completion_vlm", "feature_extraction"]},
    {"id": "ovhcloud", "label": "OVHcloud AI Endpoints", "strengths": ["chat_completion_llm", "feature_extraction"]},
    {"id": "publicai", "label": "Public AI", "strengths": ["chat_completion_llm"]},
    {"id": "replicate", "label": "Replicate", "strengths": ["chat_completion_llm", "text_to_image", "text_to_video"]},
    {"id": "sambanova", "label": "SambaNova", "strengths": ["chat_completion_llm", "chat_completion_vlm"]},
    {"id": "scaleway", "label": "Scaleway", "strengths": ["chat_completion_llm", "feature_extraction"]},
    {"id": "together", "label": "Together AI", "strengths": ["chat_completion_llm", "chat_completion_vlm", "text_to_image"]},
    {"id": "wavespeed", "label": "WaveSpeedAI", "strengths": ["text_to_image", "text_to_video"]},
    {"id": "zai-org", "label": "Z.ai", "strengths": ["chat_completion_llm", "chat_completion_vlm"]},
]


HF_PROVIDER_STARTER_MODELS: list[dict[str, Any]] = [
    {
        "model": "openai/gpt-oss-120b",
        "role": "general_reasoning_and_forum_observation",
        "selectors": ["openai/gpt-oss-120b:fastest", "openai/gpt-oss-120b:cheapest", "openai/gpt-oss-120b:sambanova"],
    },
    {
        "model": "deepseek-ai/DeepSeek-R1",
        "role": "deep_reasoning_candidate",
        "selectors": ["deepseek-ai/DeepSeek-R1:fastest", "deepseek-ai/DeepSeek-R1:cheapest"],
    },
    {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "role": "large_chat_candidate",
        "selectors": ["deepseek-ai/DeepSeek-V3-0324:fastest", "deepseek-ai/DeepSeek-V3-0324:cheapest"],
    },
    {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "role": "vision_language_candidate",
        "selectors": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova"],
    },
    {
        "model": "black-forest-labs/FLUX.1-dev",
        "role": "image_generation_candidate",
        "selectors": ["black-forest-labs/FLUX.1-dev:fastest"],
    },
]


HF_PROVIDER_CLIENT_SUPPORTED_IDS: set[str] = {
    "auto",
    "black-forest-labs",
    "cerebras",
    "clarifai",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "publicai",
    "replicate",
    "sambanova",
    "scaleway",
    "together",
    "zai-org",
}

HF_PROVIDER_CLIENT_UNSUPPORTED_IDS: set[str] = {
    str(item.get("id"))
    for item in HF_INFERENCE_PROVIDER_CATALOG
    if str(item.get("id")) not in HF_PROVIDER_CLIENT_SUPPORTED_IDS
}

HF_PROVIDER_MANUAL_OR_CREDIT_GATED_IDS: set[str] = {
    "sambanova",
}


HF_PROVIDER_DAEMON_ROUTES: list[dict[str, Any]] = [
    {
        "agent": "HF-Auto",
        "provider": "auto",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "auto_router",
        "verified_route": True,
    },
    {
        "agent": "HF-Cerebras",
        "provider": "cerebras",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "fast_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Groq",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "fast_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Novita",
        "provider": "novita",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "cheap_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Nscale",
        "provider": "nscale",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "cheap_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Together",
        "provider": "together",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "general_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Fireworks",
        "provider": "fireworks-ai",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "general_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Scaleway",
        "provider": "scaleway",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 900,
        "lane": "europe_llm",
        "verified_route": True,
    },
    {
        "agent": "HF-Zai",
        "provider": "zai-org",
        "model": "zai-org/GLM-5.1",
        "max_tokens": 900,
        "lane": "glm_llm",
        "verified_route": True,
    },
]


def provider_catalog_state() -> dict[str, Any]:
    """Read the current built-in HF Inference Provider catalog posture."""
    providers = [dict(item) for item in HF_INFERENCE_PROVIDER_CATALOG]
    daemon_routes = [dict(item) for item in HF_PROVIDER_DAEMON_ROUTES]
    routed_provider_ids = {item["provider"] for item in daemon_routes if item.get("provider") != "auto"}
    chat_provider_ids = {
        str(item.get("id"))
        for item in providers
        if "chat_completion_llm" in set(item.get("strengths") or [])
    }
    unsupported_chat_provider_ids = chat_provider_ids & HF_PROVIDER_CLIENT_UNSUPPORTED_IDS
    credit_gated_chat_provider_ids = chat_provider_ids & HF_PROVIDER_MANUAL_OR_CREDIT_GATED_IDS
    not_auto_launched = chat_provider_ids - routed_provider_ids
    return {
        "schema": "champion-continuum/hf-provider-catalog/v1",
        "source": "Hugging Face Inference Providers documentation",
        "source_urls": [
            "https://huggingface.co/docs/inference-providers/en/index",
            "https://huggingface.co/docs/inference-providers/en/pricing",
        ],
        "provider_count": len(providers),
        "providers": providers,
        "daemon_batch": {
            "launcher": "start_all_hf_provider_daemons.bat",
            "python_launcher": "launch_hf_provider_pack.py",
            "route_count": len(daemon_routes),
            "routes": daemon_routes,
            "coverage": {
                "catalog_providers": len(providers),
                "chat_capable_providers": len(chat_provider_ids),
                "verified_chat_routes": len(routed_provider_ids),
                "not_auto_launched": sorted(not_auto_launched),
                "client_unsupported": sorted(unsupported_chat_provider_ids),
                "manual_or_credit_gated": sorted(credit_gated_chat_provider_ids),
                "media_or_non_chat_only": sorted({str(item.get("id")) for item in providers} - chat_provider_ids),
            },
            "truth_boundary": "The launcher starts local forum daemons. A daemon only spends HF credits when it is assigned a turn and calls the provider. The default pack excludes providers the installed HF client rejects and routes observed as account-credit-gated.",
        },
        "routing": {
            "prefix": HF_PROVIDER_PREFIX,
            "default_provider": HF_DEFAULT_PROVIDER,
            "default_model": HF_DEFAULT_MODEL,
            "policies": ["auto", ":fastest", ":cheapest", ":preferred", ":provider-id"],
            "examples": [
                hf_provider_model_id("auto", "openai/gpt-oss-120b"),
                hf_provider_model_id("sambanova", "openai/gpt-oss-120b"),
                "openai/gpt-oss-120b:fastest",
                "openai/gpt-oss-120b:cheapest",
            ],
        },
        "free_credit_posture": {
            "kind": "monthly_free_credits_or_freemium",
            "guarantee": "not_unlimited_free",
            "note": "HF routed requests can use monthly free credits where eligible; live pricing/model availability can change and should be checked before batch use.",
            "model_catalog_refresh": "GET https://router.huggingface.co/v1/models with HF_TOKEN for live model/provider/pricing data.",
        },
        "starter_models": [dict(item) for item in HF_PROVIDER_STARTER_MODELS],
    }


def provider_registry_state() -> dict[str, Any]:
    catalog = provider_catalog_state()
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
            "catalog": {
                "provider_count": catalog["provider_count"],
                "providers": catalog["providers"],
                "starter_models": catalog["starter_models"],
                "free_credit_posture": catalog["free_credit_posture"],
            },
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


def _token(token_override: str | None = None) -> str:
    if token_override:
        return token_override
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token() or ""
    except Exception:
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


def run_hf_provider_chat(
    model_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 900,
    token_override: str | None = None,
) -> str:
    parsed = parse_provider_model_id(model_id)
    if parsed is None:
        raise ValueError("not an HF provider model id")
    provider, model = parsed
    token = _token(token_override)
    if not token:
        raise RuntimeError(
            "HF Inference Providers require Sign in with Hugging Face or an HF_TOKEN/HUGGINGFACE_HUB_TOKEN secret."
        )
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
