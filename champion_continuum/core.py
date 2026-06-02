from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .codex_archive import continuity_restore, continuity_status
from .contracts import (
    AUTHORITY_ORDER,
    DRIFT_CLASSES,
    FAILURE_CLASSES,
    OPERATIONAL_LOOP,
    RECOVERY_QUESTIONS,
    build_browser_handoff,
    build_doctor_report,
    build_output_state,
    build_query_thread,
    build_receipt,
    companion_package_status,
)
from .mcp_proxy import MCPProxy
from .store import ContinuumStore


class Continuum:
    """Portable continuity memory and packet generator."""

    def __init__(self, root: str | Path = ".continuum") -> None:
        self.store = ContinuumStore(root)
        self._mcp: MCPProxy | None = None

    @property
    def mcp(self) -> MCPProxy:
        if self._mcp is None:
            self._mcp = MCPProxy(self.store.root)
        return self._mcp

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        return self.mcp.get_tools_sync()

    # ---- Tool Surface: a FRESH active cache, scoped to the current connection ----
    # The append-only memory store must not be the tool surface: it accumulates every
    # tool from every server ever connected (all under server="external"), so a new
    # connection looked contaminated by old ones. Instead each connect overwrites a
    # single active cache file with exactly the current server's full tool list.

    def _active_cache_path(self) -> Path:
        return self.store.root / "mcp_tools_active.json"

    def _legacy_mcp_tools(self) -> list[dict[str, Any]]:
        """Compatibility fallback for stores that only have mcp_tool records.

        The active cache is authoritative when present. This fallback keeps older
        local records/tests readable without letting stale records contaminate a
        live connected server surface.
        """
        tools: list[dict[str, Any]] = []
        for record in self.store.iter_records():
            if record.kind != "mcp_tool":
                continue
            md = record.metadata or {}
            tools.append({
                "server": str(md.get("server") or ""),
                "name": str(md.get("name") or ""),
                "description": str(md.get("description") or record.text or ""),
                "args": list(md.get("args") or []),
                "input_schema": md.get("input_schema") or {},
                "mcp_url": "",
            })
        return tools

    def _active_url(self) -> str:
        try:
            cfg = json.loads((self.store.root / "mcp.json").read_text(encoding="utf-8"))
            for srv in (cfg.get("mcpServers") or {}).values():
                if srv.get("url"):
                    return str(srv["url"])
        except (ValueError, OSError):
            pass
        return ""

    def index_mcp_tools(self) -> dict[str, Any]:
        """Discover the CURRENT server's tools (one network call) and overwrite the
        active cache with the full set. Dynamic-switch safe: reconnecting to a
        different SSE URL replaces the surface entirely — no stale tools survive."""
        tools = self.list_mcp_tools()
        url = self._active_url()
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            schema = tool.get("input_schema") or {}
            props = list((schema.get("properties") or {}).keys())
            normalized.append({
                "server": str(tool.get("server") or ""),
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "args": props,
                "input_schema": schema,
                "mcp_url": url,
            })
        cache_path = self._active_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"url": url, "ts": time.time(), "tools": normalized}, ensure_ascii=False),
            encoding="utf-8")
        return {
            "discovered": len(normalized),
            "total": len(normalized),
            "indexed_new": len(normalized),
            "url": url,
            "servers": sorted({t["server"] for t in normalized if t["server"]}),
        }

    def active_tools(self) -> list[dict[str, Any]]:
        """The full tool list of the currently-connected server (no network)."""
        cache_path = self._active_cache_path()
        if not cache_path.exists():
            return self._legacy_mcp_tools()
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(data.get("tools") or [])
        except (ValueError, OSError):
            return []

    def indexed_tool_summary(self) -> dict[str, Any]:
        tools = self.active_tools()
        return {
            "count": len(tools),
            "servers": sorted({str(t.get("server") or "") for t in tools if t.get("server")}),
            "url": self._active_url(),
        }

    def search_tools(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Ranked search over the ACTIVE server's tools only (no cross-server bleed)."""
        tools = self.active_tools()
        q = (query or "").lower().strip()
        if not q:
            return tools[:limit]
        scored: list[tuple[int, dict[str, Any]]] = []
        for t in tools:
            name = str(t.get("name") or "").lower()
            hay = f"{name} {str(t.get('description') or '').lower()} {' '.join(t.get('args') or []).lower()}"
            score = 0
            if q == name:
                score += 100
            if q in name:
                score += 10
            if q in hay:
                score += 1
            for term in q.split():
                if term in hay:
                    score += 1
            if score:
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:limit]]

    def init(self) -> dict[str, Any]:
        return self.store.initialize()

    def status(self) -> dict[str, Any]:
        return self.store.status()

    def doctor(self, cwd: str = "") -> dict[str, Any]:
        return build_doctor_report(self.status(), cwd=cwd)

    def remember(
        self,
        text: str,
        kind: str = "note",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return asdict(self.store.remember(text=text, kind=kind, tags=tags, metadata=metadata))

    def ingest_file(self, path: str | Path, kind: str = "document", tags: list[str] | None = None) -> dict[str, Any]:
        source = Path(path)
        text = source.read_text(encoding="utf-8", errors="replace")
        return self.remember(
            text=text,
            kind=kind,
            tags=tags or ["ingested_file"],
            metadata={"source_path": str(source)},
        )

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        return self.store.search(query=query, limit=limit)

    def archive_status(self, codex_home: str | None = None, limit: int = 8) -> dict[str, Any]:
        return continuity_status(codex_home=codex_home, limit=limit)

    def restore(
        self,
        summary: str = "",
        cwd: str = "",
        codex_home: str | None = None,
        limit: int = 3,
        scan_limit: int = 80,
    ) -> dict[str, Any]:
        result = continuity_restore(
            summary=summary,
            cwd=cwd,
            codex_home=codex_home,
            limit=limit,
            scan_limit=scan_limit,
        )
        packet = result.get("packet") if isinstance(result, dict) else None
        if isinstance(packet, dict):
            self.store.save_packet(packet, name="archive_restore")
            self.store.remember(
                text=packet.get("objective_seed") or summary or "continuity restore",
                kind="continuity_packet",
                tags=["archive_restore", str(packet.get("status") or "")],
                metadata={"packet": packet},
            )
        return result

    def packet(
        self,
        summary: str = "",
        cwd: str = "",
        codex_home: str | None = None,
        limit: int = 6,
        scan_limit: int = 48,
    ) -> dict[str, Any]:
        archive = self.restore(summary=summary, cwd=cwd, codex_home=codex_home, limit=3, scan_limit=scan_limit)
        memory_hits = self.search(summary or cwd or "continuity", limit=limit)
        root_status = self.status()
        companions = companion_package_status()
        query_thread = build_query_thread(summary=summary, cwd=cwd, root_status=root_status)
        output_state = build_output_state(
            summary=summary,
            cwd=cwd,
            root_status=root_status,
            archive_restore=archive,
            memory_hits=memory_hits,
            companions=companions,
        )
        packet = {
            "packet_kind": "continuum_bootstrap_packet",
            "summary": summary,
            "cwd": cwd,
            "continuum_root": str(self.store.root.resolve()),
            "status": "ready",
            "authority_order": AUTHORITY_ORDER,
            "rules": [
                "Do not claim remembered context unless it appears in the packet or live files.",
                "Treat archive restore as recovery context, not live authority.",
                "Before editing, identify verified, inferred, unknown, active seam, and dependency order.",
                "After editing, run the narrowest available verification.",
            ],
            "query_thread": query_thread,
            "output_state": output_state,
            "drift_classes": DRIFT_CLASSES,
            "failure_classes": FAILURE_CLASSES,
            "operational_loop": OPERATIONAL_LOOP,
            "recovery_questions": RECOVERY_QUESTIONS,
            "companion_packages": companions,
            "receipt_protocol": {
                "command": "continuum receipt <action> --summary <result>",
                "rule": "Every meaningful mutation should leave one structured receipt.",
            },
            "memory_hits": memory_hits,
            "archive_restore": archive,
            "next_actions": [
                "Read AGENT_READ_THIS_FIRST.md if this came from a drop-in bundle.",
                "Run continuum status and continuum search for the active objective.",
                "Run continuum doctor to inspect companion packages and root attachment.",
                "Use continuum handoff when handing this bundle to a browser/no-tools agent.",
                "Inspect the local repo state before making changes.",
                "If Champion Council payload is available, use continuum engine --self-test before launching it.",
            ],
        }
        self.store.save_packet(packet, name="bootstrap_packet")
        return packet

    def receipt(
        self,
        action: str,
        summary: str = "",
        status: str = "ok",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt = build_receipt(
            action=action,
            summary=summary,
            status=status,
            tags=tags,
            metadata=metadata,
            root_status=self.status(),
        )
        self.store.save_packet(receipt, name="receipt")
        self.store.remember(
            text=summary or action,
            kind="receipt",
            tags=["receipt", status, *(tags or [])],
            metadata={"receipt": receipt},
        )
        return receipt

    def handoff(
        self,
        summary: str = "",
        cwd: str = "",
        audience: str = "browser",
        codex_home: str | None = None,
        scan_limit: int = 24,
    ) -> dict[str, Any]:
        packet = self.packet(summary=summary, cwd=cwd, codex_home=codex_home, scan_limit=scan_limit)
        markdown = build_browser_handoff(summary=summary, packet=packet, audience=audience)
        handoff = {
            "packet_kind": "continuum_handoff",
            "audience": audience,
            "summary": summary,
            "markdown": markdown,
            "packet": packet,
        }
        self.store.save_packet(handoff, name="handoff")
        return handoff
