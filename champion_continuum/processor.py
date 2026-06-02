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

import re
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
            return {"kind": "tools", "verb": "search", "ok": True, "query": query, "hits": hits, "provenance": provenance}
        if kind == "tool":
            if "." in verb:
                server_name, tool_name = verb.split(".", 1)
            else:
                server_name, tool_name = None, verb  # resolve to the sole connected server
            arguments = _parse_kv_args(args)
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
