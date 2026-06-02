"""Composition aesthetic rendering — the agent composes its own user-facing surface.

The model emits a single directive per message:

    [[compose: {"blocks": [ ... ]}]]

and this module turns that directive into a *safe* HTML fragment that the surface
renders beside the chat. The agent gets spatial agency over how its turn is
presented — message by message — WITHOUT ever emitting raw HTML/JS.

Safety spine (this is the whole point, and it is non-negotiable):
  * The model never emits markup. It emits a small, declarative block grammar.
  * Every text field is HTML-escaped.
  * Colours/tones are *names* that map to a fixed palette. No arbitrary CSS,
    no inline style strings from the model, no <script>, no event handlers,
    no urls. The renderer is the only thing that writes tags.
  * Unknown block types and unknown tones degrade gracefully (skipped / neutral)
    rather than failing the turn.

This mirrors the quinesmith discipline: the agent may *evolve its presentation*,
but only inside a grammar that cannot break out. If the grammar itself is ever
allowed to evolve, that edit is a quine and goes through operator-run verification.
"""

from __future__ import annotations

import html
import json
from typing import Any

# Named palette — the ONLY colours a block may reference. The model picks a name;
# the renderer owns the value. (Tuned to the Space's muted dark aesthetic; Gemini
# owns final values.)
_PALETTE = {
    "insight": ("#7aa2f7", "rgba(122,162,247,0.10)"),   # calm blue
    "success": ("#9ece6a", "rgba(158,206,106,0.10)"),   # green
    "warn":    ("#e0af68", "rgba(224,175,104,0.10)"),   # amber
    "alert":   ("#f7768e", "rgba(247,118,142,0.10)"),   # rose
    "quiet":   ("#9aa5ce", "rgba(154,165,206,0.08)"),   # slate
    "scarab":  ("#bb9af7", "rgba(187,154,247,0.10)"),   # violet
}
_DEFAULT_TONE = "quiet"

_MAX_BLOCKS = 24
_MAX_TEXT = 4000

_STYLE = """
<style>
.cc-stage{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;display:flex;
 flex-direction:column;gap:10px;}
.cc-block{border-radius:12px;padding:12px 14px;line-height:1.45;}
.cc-callout{border-left:3px solid var(--cc-accent);background:var(--cc-bg);}
.cc-callout .cc-title{font-weight:600;color:var(--cc-accent);margin-bottom:4px;
 letter-spacing:.2px;}
.cc-card{border:1px solid var(--cc-accent);background:var(--cc-bg);}
.cc-card .cc-title{font-weight:600;color:var(--cc-accent);margin-bottom:6px;}
.cc-heading{font-weight:700;color:var(--cc-accent);letter-spacing:.3px;}
.cc-columns{display:flex;gap:10px;flex-wrap:wrap;}
.cc-columns .cc-col{flex:1 1 160px;border-radius:10px;padding:10px 12px;
 background:var(--cc-bg);border:1px solid var(--cc-accent);}
.cc-columns .cc-col .cc-title{font-weight:600;color:var(--cc-accent);margin-bottom:4px;}
.cc-kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;}
.cc-kv .cc-k{color:var(--cc-accent);font-weight:600;}
.cc-kv .cc-v{opacity:.92;}
.cc-badges{display:flex;gap:6px;flex-wrap:wrap;}
.cc-badge{font-size:.78em;padding:3px 9px;border-radius:999px;
 border:1px solid var(--cc-accent);color:var(--cc-accent);background:var(--cc-bg);}
.cc-code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;
 background:rgba(0,0,0,.30);border-radius:10px;padding:10px 12px;white-space:pre-wrap;
 overflow-x:auto;border:1px solid rgba(255,255,255,.06);}
.cc-divider{height:1px;background:rgba(255,255,255,.10);margin:2px 0;}
</style>
"""


def _esc(value: Any, limit: int = _MAX_TEXT) -> str:
    return html.escape(str(value)[:limit])


def _tone(name: Any) -> tuple[str, str]:
    return _PALETTE.get(str(name).lower().strip(), _PALETTE[_DEFAULT_TONE])


def _wrap(tone: Any, klass: str, inner: str) -> str:
    accent, bg = _tone(tone)
    return (f'<div class="cc-block {klass}" '
            f'style="--cc-accent:{accent};--cc-bg:{bg}">{inner}</div>')


def _render_block(block: dict) -> str:
    btype = str(block.get("type", "")).lower().strip()
    tone = block.get("tone", _DEFAULT_TONE)

    if btype == "callout":
        title = block.get("title")
        head = f'<div class="cc-title">{_esc(title)}</div>' if title else ""
        body = f'<div>{_esc(block.get("body",""))}</div>'
        return _wrap(tone, "cc-callout", head + body)

    if btype == "card":
        title = block.get("title")
        head = f'<div class="cc-title">{_esc(title)}</div>' if title else ""
        body = f'<div>{_esc(block.get("body",""))}</div>'
        return _wrap(tone, "cc-card", head + body)

    if btype == "heading":
        return _wrap(tone, "cc-heading", _esc(block.get("text", "")))

    if btype == "columns":
        cols = block.get("cols") or []
        cells = []
        for c in cols[:6]:
            if not isinstance(c, dict):
                continue
            t = c.get("title")
            h = f'<div class="cc-title">{_esc(t)}</div>' if t else ""
            cells.append(f'<div class="cc-col">{h}<div>{_esc(c.get("body",""))}</div></div>')
        return _wrap(tone, "cc-columns", "".join(cells))

    if btype == "kv":
        items = block.get("items") or []
        rows = []
        for it in items[:40]:
            if not isinstance(it, dict):
                continue
            rows.append(f'<div class="cc-k">{_esc(it.get("k",""),200)}</div>'
                        f'<div class="cc-v">{_esc(it.get("v",""),800)}</div>')
        return _wrap(tone, "", f'<div class="cc-kv">{"".join(rows)}</div>')

    if btype == "badges":
        labels = block.get("labels") or []
        chips = "".join(f'<span class="cc-badge">{_esc(l,80)}</span>' for l in labels[:20])
        return _wrap(tone, "", f'<div class="cc-badges">{chips}</div>')

    if btype == "code":
        return _wrap(tone, "", f'<pre class="cc-code">{_esc(block.get("body",""))}</pre>')

    if btype == "divider":
        return '<div class="cc-divider"></div>'

    # Unknown block: degrade to a quiet callout of whatever text we can find.
    text = block.get("body") or block.get("text") or ""
    if text:
        return _wrap(_DEFAULT_TONE, "cc-callout", f'<div>{_esc(text)}</div>')
    return ""


def render_composition(spec: Any) -> tuple[str, bool]:
    """Render a compose directive into a safe HTML fragment.

    `spec` may be a JSON string (what the model emits) or an already-parsed dict.
    Returns (html, ok). On any parse failure returns ("", False) so the caller can
    simply leave the stage unchanged — a bad directive never breaks the turn.
    """
    if not spec:
        return "", False
    data = spec
    if isinstance(spec, str):
        try:
            data = json.loads(spec)
        except (ValueError, TypeError):
            return "", False
    if not isinstance(data, dict):
        return "", False
    blocks = data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return "", False

    parts = []
    for block in blocks[:_MAX_BLOCKS]:
        if isinstance(block, dict):
            try:
                rendered = _render_block(block)
            except Exception:
                rendered = ""
            if rendered:
                parts.append(rendered)
    if not parts:
        return "", False
    return _STYLE + '<div class="cc-stage">' + "".join(parts) + "</div>", True


# Directive extraction: pull the [[compose: ...]] block out of model text.
# Tolerant of nested braces by scanning for the matching ]] after "compose:".
def extract_compose(text: str) -> tuple[str | None, str]:
    """Find the first [[compose: ...]] directive in `text`.

    Returns (spec_json_or_None, text_without_directive). Presentation is removed
    from the chat transcript so the user reads prose; the composition renders on
    the stage. (This is presentation extraction, NOT result-stripping — the model
    *chose* to emit a render directive; we honour it on the surface.)
    """
    if not text:
        return None, text
    marker = "[[compose:"
    low = text.lower()
    i = low.find(marker)
    if i == -1:
        return None, text
    j = text.find("]]", i)
    if j == -1:
        return None, text
    spec = text[i + len(marker):j].strip()
    before, after = text[:i].rstrip(), text[j + 2:].lstrip()
    cleaned = (before + (" " if before and after else "") + after).strip()
    return spec, cleaned


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sample = {
        "blocks": [
            {"type": "heading", "tone": "scarab", "text": "Mission Readout"},
            {"type": "callout", "tone": "insight", "title": "What I did",
             "body": "Searched memory, found 3 prior runs of this task."},
            {"type": "columns", "tone": "quiet", "cols": [
                {"title": "Tools used", "body": "web_search, imagine"},
                {"title": "Confidence", "body": "0.82"},
            ]},
            {"type": "kv", "tone": "success", "items": [
                {"k": "latency", "v": "1.2s"}, {"k": "tokens", "v": "640"}]},
            {"type": "badges", "tone": "warn", "labels": ["verified", "merkle-chained"]},
            {"type": "code", "body": "<script>alert(1)</script>"},  # must be escaped
        ]
    }
    out, ok = render_composition(json.dumps(sample))
    assert ok, "render failed"
    assert "<script>" not in out, "XSS leak!"
    assert "&lt;script&gt;" in out, "code not escaped"
    spec, cleaned = extract_compose("prose here [[compose: {\"blocks\":[]}]] tail")
    assert spec == '{"blocks":[]}' and cleaned == "prose here tail"
    print("compose.py self-test OK  (len html=%d)" % len(out))
