"""Optional cascade-lattice backbone binding for Continuum.

cascade-lattice (operator-built) is the provenance/observation/tracing backbone.
Every relay step (a human paste, a model command, a tool execution, a memory op)
flows through a LatticeTrace, which:

  - chains a merkle provenance Receipt per step (parent_cid -> the breadcrumb),
  - converges heterogeneous signals via SymbioticAdapter.interpret -> Event,
  - links events causally in a CausationGraph (forward/backward tracing),
  - color-codes each step by kind (color codification).

It degrades gracefully: with cascade-lattice absent, the lightweight in-memory
trail still works, so the relay never depends on the backbone being installed.
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:  # the backbone is optional
    import cascade_lattice as _cl

    HAVE_CASCADE = True
except Exception:  # pragma: no cover - environment without the backbone
    _cl = None
    HAVE_CASCADE = False


# Color codification per relay event kind (RGB), matching ArcadeFeedback's field.
KIND_COLORS: dict[str, tuple[int, int, int]] = {
    "session": (180, 180, 180),
    "connect": (90, 180, 255),
    "exploration": (90, 255, 210),
    "operator": (80, 200, 255),
    "oracle": (255, 190, 95),
    "user_relay": (80, 200, 255),    # cyan  - you pasted/relayed
    "router": (255, 230, 120),
    "model_request": (120, 180, 255),
    "model_response": (150, 120, 255),
    "memory": (120, 255, 160),       # green - remember/search
    "tool_search": (255, 210, 90),   # amber - [[tools: search]]
    "tool_call": (255, 140, 60),     # orange- [[tool: ...]] executed
    "tool_result": (200, 160, 255),  # violet- converged result
    "hold": (255, 80, 80),           # red   - awaiting human gate
    "error": (255, 60, 60),
}


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 260:
            return value[:257] + "..."
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(_safe_scalar(v)) for v in list(value)[:8])
    if isinstance(value, dict):
        return {str(k): _safe_scalar(v) for k, v in list(value.items())[:16]}
    return str(value)[:260]


class LatticeTrace:
    """Per-session provenance + causation trace over relay steps."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.enabled = HAVE_CASCADE
        self.last_cid: Optional[str] = None
        self.steps: list[dict[str, Any]] = []  # always-on lightweight breadcrumb
        self._adapter = _cl.SymbioticAdapter() if HAVE_CASCADE else None
        self._graph = _cl.CausationGraph() if HAVE_CASCADE else None

    def converge(self, signal: Any):
        """Interpret any signal into a Cascade Event via the SymbioticAdapter."""
        if self._adapter is None:
            return None
        try:
            return self._adapter.interpret(signal)
        except Exception:
            return None

    def observe(self, kind: str, summary: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record one relay step: chain a merkle Receipt + link it causally.

        Returns a lightweight breadcrumb row (always populated) carrying the
        color, parent linkage, and - when the backbone is present - the cid and
        merkle_root proving the step actually happened."""
        payload = {"kind": kind, "summary": summary, **(data or {})}
        safe_payload = {str(k): _safe_scalar(v) for k, v in payload.items()}
        parent_step = self.steps[-1]["n"] if self.steps else None
        row: dict[str, Any] = {
            "n": len(self.steps) + 1,
            "kind": kind,
            "summary": summary,
            "color": KIND_COLORS.get(kind, (200, 200, 200)),
            "ts": time.time(),
            "parent_cid": self.last_cid,
            "parent_step": parent_step,
            "cid": None,
            "merkle_root": None,
            "event_id": None,
            "data": safe_payload,
        }
        for key in (
            "model", "role", "tool", "verb", "ok", "latency_ms", "char_count",
            "token_est", "command_count", "cache_ids", "hits", "server", "url",
        ):
            if key in safe_payload:
                row[key] = safe_payload[key]
        if HAVE_CASCADE:
            try:
                receipt = _cl.store_observe(
                    model_id=f"continuum:{self.session_id}",
                    data=safe_payload,
                    parent_cid=self.last_cid,
                    sync=False,
                )
                row["cid"] = receipt.cid
                row["merkle_root"] = receipt.merkle_root
                self.last_cid = receipt.cid
                event = self.converge(payload)
                if event is not None and self._graph is not None:
                    self._graph.add_event(event)
                    row["event_id"] = getattr(event, "event_id", None)
            except Exception:
                pass  # never let provenance break the relay
        self.steps.append(row)
        return row

    def breadcrumb(self, limit: int = 60) -> list[dict[str, Any]]:
        """The trail the agent and the human both read - real, ordered, grounded."""
        return self.steps[-limit:]

    def stats(self) -> dict[str, Any]:
        return {
            "backbone": "cascade-lattice" if self.enabled else "in-memory only",
            "steps": len(self.steps),
            "head_cid": self.last_cid,
        }
