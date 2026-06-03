"""Co-pilot Deck — a local surface to OBSERVE and INTERACT with Claude's own
continuum interactions. There is NO model in this app. No HuggingFace weights,
no inference. The brain is Claude (the agent in the chat), operating the continuum
as a library and writing to a shared store on disk. This deck just renders what
Claude does, live, and relays the operator's input back to Claude.

The bus is the filesystem (shared root):
  feed.jsonl  - the running transcript (operator + claude turns)
  inbox.jsonl - operator messages waiting for Claude to read
  stage.html  - Claude's latest composed surface (the [[compose]] output)
  .continuum/ - the continuum memory store Claude writes to

Operator types here -> appended to inbox.jsonl + feed.jsonl.
Claude (from its runtime) reads inbox.jsonl, acts on the continuum, appends its
turn to feed.jsonl and writes stage.html. A timer re-reads disk so the operator
watches it happen.
"""
from __future__ import annotations

import html as _html
import json
import time
from pathlib import Path

import gradio as gr

try:
    from champion_continuum import Continuum
except Exception:  # deck still renders feed/stage without the store
    Continuum = None

ROOT = Path(__file__).parent / "copilot_deck_state"
ROOT.mkdir(exist_ok=True)
FEED = ROOT / "feed.jsonl"
INBOX = ROOT / "inbox.jsonl"
STAGE = ROOT / "stage.html"
CONT = Continuum(root=str(ROOT / ".continuum")) if Continuum else None

ACCENT = {"operator": "#7aa2f7", "claude": "#bb9af7"}


def _append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def render_feed() -> str:
    msgs = _read_jsonl(FEED)
    if not msgs:
        return ("<div style='opacity:.5;padding:14px'>Waiting for Claude. "
                "Type below and Claude will read it from the inbox and act on the continuum.</div>")
    rows = []
    for m in msgs[-80:]:
        role = m.get("role", "claude")
        who = "You" if role == "operator" else "Claude"
        accent = ACCENT.get(role, "#9aa5ce")
        text = _html.escape(str(m.get("text", "")))
        rows.append(
            f"<div style='margin:8px 0;padding:8px 12px;border-left:3px solid {accent};"
            f"background:rgba(255,255,255,.03);border-radius:8px'>"
            f"<div style='color:{accent};font-weight:600;font-size:.8em;letter-spacing:.3px'>{who}</div>"
            f"<div style='white-space:pre-wrap;line-height:1.45'>{text}</div></div>"
        )
    return "<div>" + "".join(rows) + "</div>"


def render_stage() -> str:
    if STAGE.exists():
        return STAGE.read_text(encoding="utf-8")
    return ("<div style='opacity:.5;padding:14px'>Claude composes its surface here "
            "as it works — readouts, plans, status panels.</div>")


def render_memory() -> list[list[str]]:
    if not CONT:
        return []
    rows = []
    try:
        for r in CONT.store.iter_records():
            rows.append([getattr(r, "kind", "") or "", (getattr(r, "text", "") or "")[:90]])
    except Exception:
        return []
    return rows[-60:]


def on_send(message: str):
    if message and message.strip():
        rec = {"ts": time.time(), "role": "operator", "text": message.strip()}
        _append(INBOX, rec)
        _append(FEED, rec)
    return "", render_feed()


def tick():
    return render_feed(), render_stage(), render_memory()


CSS = """
.gradio-container { max-width: 1200px !important; }
#deck-feed { height: 56vh; overflow-y: auto; padding: 6px; }
#deck-stage { min-height: 56vh; overflow-y: auto; padding: 6px; }
"""

with gr.Blocks(title="Co-pilot Deck — Claude on the Continuum", css=CSS,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="stone")) as deck:
    gr.Markdown("## Co-pilot Deck\nObserve and talk to Claude operating the continuum. "
                "No model runs here — Claude is the brain, writing to a shared store you watch live.")
    with gr.Row():
        with gr.Column(scale=3):
            feed = gr.HTML(render_feed(), elem_id="deck-feed")
            with gr.Row():
                box = gr.Textbox(placeholder="Talk to Claude (lands in the inbox Claude reads)...",
                                 show_label=False, scale=5, autofocus=True)
                send = gr.Button("Send", scale=1, variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Stage"):
                    stage = gr.HTML(render_stage(), elem_id="deck-stage")
                with gr.TabItem("Memory"):
                    memory = gr.Dataframe(headers=["kind", "text"], datatype=["str", "str"],
                                          interactive=False, value=render_memory())

    send.click(on_send, box, [box, feed])
    box.submit(on_send, box, [box, feed])
    timer = gr.Timer(1.5)
    timer.tick(tick, None, [feed, stage, memory])

if __name__ == "__main__":
    deck.launch(server_name="127.0.0.1", server_port=7870)
