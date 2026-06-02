"""Hugging Face Inference Provider adapter for forum_daemon.py.

Reads a forum prompt from stdin, calls the configured HF Inference Provider
chat model, and prints only the assistant text. No token is printed or stored
here; auth comes from HF_TOKEN/HUGGINGFACE_HUB_TOKEN or the local HF CLI login.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from continuum_provider_registry import hf_provider_model_id, run_hf_provider_chat


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parent
CHANNEL = Path(os.environ.get("FORUM_CHANNEL") or os.environ.get("CONTINUUM_BRAIN_CHANNEL") or ROOT / "cli_brain_channel")
SHARED_STORE = Path(os.environ.get("CONTINUUM_SHARED_STORE") or CHANNEL / "shared_store")


DEFAULT_SYSTEM = """You are the Hugging Face Inference Provider mind in Champion Continuum.

You answer forum/Bear Claw turns from the operator's normal chat. Be direct,
use the available evidence in the prompt, and keep the user-facing flow simple.
If the user asks for resources, tools, music, translation, or daemon status,
use the LOCAL CONTINUUM RESOURCE SURFACE included below. Do not claim you cannot
pull local data when that surface is present. Do not invent missing tools.
"""


def _env(name: str, fallback: str = "") -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else fallback


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tool_name(tool: dict) -> str:
    server = str(tool.get("server") or "").strip()
    name = str(tool.get("name") or "").strip()
    return f"{server}.{name}" if server and name else name or server or "(unnamed)"


def _local_resource_snapshot() -> dict:
    _refresh_empty_tool_cache()
    active_tools = _read_json(SHARED_STORE / "mcp_tools_active.json", {})
    peer_links = _read_json(CHANNEL / "continuum_peer_links.json", {})
    try:
        from continuum_daemon_registry import load_daemon_registry

        daemon_registry = load_daemon_registry(CHANNEL)
    except Exception as exc:
        daemon_registry = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        from continuum_music_forge import music_forge_state

        music = music_forge_state()
    except Exception as exc:
        music = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        from continuum_provider_registry import provider_registry_state

        providers = provider_registry_state()
    except Exception as exc:
        providers = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    tools = list(active_tools.get("tools") or [])
    links = list(peer_links.get("links") or [])
    return {
        "schema": "champion-continuum/hf-daemon-resource-snapshot/v1",
        "active_mcp_url": active_tools.get("url") or "",
        "tool_count": len(tools),
        "tools": tools,
        "peer_link_count": len(links),
        "peer_links": links,
        "utility_daemons": daemon_registry,
        "music_forge": music,
        "providers": providers,
    }


def _candidate_mcp_urls() -> list[str]:
    urls: list[str] = []
    peer_links = _read_json(CHANNEL / "continuum_peer_links.json", {})
    for item in list(peer_links.get("links") or []):
        if isinstance(item, dict):
            url = str(item.get("mcp_url") or item.get("url") or "").strip()
            if url and url not in urls:
                urls.append(url)
    default_url = "http://127.0.0.1:7872/mcp/sse"
    if default_url not in urls:
        urls.append(default_url)
    return urls


def _refresh_empty_tool_cache() -> None:
    active_tools = _read_json(SHARED_STORE / "mcp_tools_active.json", {})
    if active_tools.get("tools"):
        return
    try:
        from champion_continuum import Continuum

        continuum = Continuum(SHARED_STORE)
        for url in _candidate_mcp_urls():
            try:
                (SHARED_STORE / "mcp.json").write_text(
                    json.dumps({"mcpServers": {"continuum_1": {"url": url}}}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result = continuum.index_mcp_tools()
            except Exception:
                continue
            if int(result.get("total") or 0) > 0:
                return
    except Exception:
        return


def _resource_digest(snapshot: dict) -> str:
    tools = list(snapshot.get("tools") or [])
    tool_lines = []
    for tool in tools[:40]:
        args = ", ".join(str(item) for item in (tool.get("args") or [])) or "no args"
        desc = str(tool.get("description") or "").strip()
        tool_lines.append(f"- {_tool_name(tool)} ({args}) :: {desc[:180]}")
    if not tool_lines:
        tool_lines.append("- No active MCP tools are indexed right now.")
    links = [
        f"- {item.get('label') or item.get('link_id')}: {item.get('mcp_url') or item.get('url')}"
        for item in (snapshot.get("peer_links") or [])
    ] or ["- No peer links saved."]
    daemons = []
    for item in ((snapshot.get("utility_daemons") or {}).get("daemons") or []):
        state = "stale" if item.get("stale") else "active"
        safe = "safe" if item.get("safe_for_autonomous_assignment") else "gated"
        daemons.append(f"- {item.get('agent')} ({item.get('kind')}): {state}, {safe}")
    if not daemons:
        daemons.append("- No utility daemons registered.")
    music = snapshot.get("music_forge") or {}
    providers = (snapshot.get("providers") or {}).get("huggingface_inference_providers") or {}
    return "\n".join(
        [
            "LOCAL CONTINUUM RESOURCE SURFACE",
            f"Active MCP URL: {snapshot.get('active_mcp_url') or '(none)'}",
            f"Indexed MCP tools: {snapshot.get('tool_count', 0)}",
            "",
            "Tools:",
            *tool_lines,
            "",
            "Peer links:",
            *links,
            "",
            "Utility daemons:",
            *daemons,
            "",
            "Music Forge:",
            f"- status: {music.get('status', 'unknown')}",
            f"- output_dir: {music.get('output_dir', '(unknown)')}",
            "",
            "HF providers:",
            f"- default: {providers.get('default_provider', 'auto')}:{providers.get('default_model', '')}",
            f"- auth: {providers.get('auth', {}).get('status', 'unknown') if isinstance(providers.get('auth'), dict) else 'unknown'}",
        ]
    )


def _looks_like_resource_list_request(text: str) -> bool:
    lowered = text.lower()
    resource_terms = ("resource list", "list resources", "all resources", "what resources", "tool list", "list tools", "what tools")
    return any(term in lowered for term in resource_terms)


def _direct_resource_answer(snapshot: dict) -> str:
    if int(snapshot.get("tool_count") or 0) <= 0:
        active_url = snapshot.get("active_mcp_url") or "(none)"
        return (
            "Resource surface checked locally. No MCP tools are indexed right now.\n\n"
            f"Active MCP URL: `{active_url}`\n\n"
            "Use the live Continuum MCP endpoint `http://127.0.0.1:7872/mcp/sse`, "
            "then click Save & Connect Services so the cache fills before assigning a tool-backed turn."
        )
    return _resource_digest(snapshot)


def main() -> int:
    prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
    snapshot = _local_resource_snapshot()
    if _looks_like_resource_list_request(prompt):
        print(_direct_resource_answer(snapshot))
        return 0
    provider = _env("FORUM_HF_PROVIDER", _env("CONTINUUM_HF_PROVIDER", "auto"))
    model = _env("FORUM_HF_MODEL", _env("CONTINUUM_HF_PROVIDER_MODEL", "openai/gpt-oss-120b"))
    max_tokens = int(_env("FORUM_HF_MAX_TOKENS", "900"))
    system = _env("FORUM_HF_SYSTEM", DEFAULT_SYSTEM)
    model_id = hf_provider_model_id(provider=provider, model=model)
    messages = [
        {"role": "system", "content": system + "\n\n" + _resource_digest(snapshot)},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = run_hf_provider_chat(model_id, messages, max_tokens=max_tokens).strip()
    except Exception as exc:
        print(
            "HF provider daemon could not answer: "
            f"{type(exc).__name__}: {exc}\n\n"
            "Check local Hugging Face auth with `hf auth login` or set HF_TOKEN, "
            "then restart `start_hf_daemon.bat`."
        )
        return 1
    print(reply or "(HF provider produced no output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
