"""Champion Continuum - live ZeroGPU demo.

A real small model (<=8B) acts as a TOOL-LESS agent: it emits
[[continuum: ...]] commands in its replies, the vendored Continuum processor 
runs them against a per-session memory store, and the results are fed back so
the model can use what it remembered. This is the whole point of the package:
giving continuity to agents running through tool relay boundaries.

ARCHITECTURE (stable during aesthetic passes):
  - run_model() tokenizes on CPU, generates inside @spaces.GPU, decodes on CPU.
  - ensure_loaded() places the model on cuda at non-GPU level (ZeroGPU pattern).
  - chat() runs the relay loop: model -> process_text -> feed results -> model.
  - Graceful quota handling: if the GPU quota is exhausted, we explain it and
    point at sign-in, rather than crashing (CPU fallback for 8B is not viable).
"""

from __future__ import annotations

import gc
import json
import os
import re
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import spaces  # HF ZeroGPU runtime
except ModuleNotFoundError:
    # Local cockpit (no HF runtime): make @spaces.GPU a pass-through so the same
    # app.py runs on your machine. On CPU the decorated function simply runs in-process.
    class _SpacesStub:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def _deco(fn):
                return fn
            return _deco
    spaces = _SpacesStub()
import torch
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer

from champion_continuum import Continuum, __version__, process_text
from champion_continuum.compose import extract_compose, render_composition
from champion_continuum.lattice import LatticeTrace
from champion_continuum.system_prompts import get_system_prompt
from continuum_provider_registry import hf_provider_model_id, parse_provider_model_id, provider_registry_state, run_hf_provider_chat
from continuum_translation_faculty import build_translation_faculty_packet, translation_faculty_state

try:
    from dreamer_oracle_gate import OracleGate
except Exception:  # Space still boots if the optional gate package is unavailable.
    OracleGate = None

HF_TOKEN = os.environ.get("HF_TOKEN")  # needed for gated models (Gemma, Llama) + quota

# Curated ZeroGPU menu. Ungated models work out of the box; gated models need
# the owner to accept the license and set the HF_TOKEN secret.
MODELS: list[tuple[str, str]] = [
    ("Auto Router - balanced scout/operator/critic", "__router_balanced__"),
    ("Auto Router - deep final pass (loads 14B)", "__router_deep__"),
    ("Qwen2.5 0.5B Instruct - tiny scout", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5 1.5B Instruct - fast", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5 3B Instruct - nimble", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen2.5 7B Instruct - default operator", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5 14B Instruct - stronger reasoning", "Qwen/Qwen2.5-14B-Instruct"),
    ("Qwen2.5 Coder 1.5B - fast tool syntax", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    ("Qwen2.5 Coder 7B - code/tool operator", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("Qwen2.5 Coder 14B - deep code/tool operator", "Qwen/Qwen2.5-Coder-14B-Instruct"),
    ("HF Inference Providers - auto router", hf_provider_model_id()),
    ("SmolLM2 1.7B Instruct - lightweight", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("Phi-4 mini instruct - compact reasoning", "microsoft/Phi-4-mini-instruct"),
    ("Phi-4 mini reasoning - deliberate small", "microsoft/Phi-4-mini-reasoning"),
    ("Mistral 7B Instruct v0.3 - generalist", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("DeepSeek R1 Distill Qwen 7B - reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("DeepSeek R1 Distill Qwen 14B - deeper reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    ("Gemma 2 2B IT - gated", "google/gemma-2-2b-it"),
    ("Gemma 3 4B IT - gated", "google/gemma-3-4b-it"),
    ("Gemma 3 12B IT - gated bigger", "google/gemma-3-12b-it"),
    ("Llama 3.2 3B Instruct - gated", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Llama 3.1 8B Instruct - gated", "meta-llama/Llama-3.1-8B-Instruct"),
]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# CLI-brain mode (local): no HuggingFace model ever loads. The brain is the agent
# in the operator's CLI (Claude / Codex / Gemini), relayed turn-by-turn through a
# shared channel on disk. Full faculty stays; only the inference source changes.
CLI_BRAIN = bool(os.environ.get("CONTINUUM_CLI_BRAIN"))
# Keystone: in CLI mode every session AND the CLI agent point at ONE store on disk,
# so what the agent remembers from its runtime is what the deck recalls. One session,
# two windows. (Set CONTINUUM_SHARED_ROOT to override the location.)
SHARED_STORE_ROOT = Path(os.environ.get("CONTINUUM_SHARED_ROOT")
                         or (Path(__file__).parent / "cli_brain_channel" / "shared_store"))
if CLI_BRAIN:
    SHARED_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS = [
        ("Claude — this CLI", "claude-cli"),
        ("Codex — CLI relay", "codex-cli"),
        ("Gemini — CLI relay", "gemini-cli"),
        ("Inference provider", "provider-cli"),
    ]
    DEFAULT_MODEL = "claude-cli"

ROUTER_PLANS: dict[str, list[tuple[str, str]]] = {
    "__router_balanced__": [
        ("scout", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("operator", "Qwen/Qwen2.5-7B-Instruct"),
        ("critic", "Qwen/Qwen2.5-7B-Instruct"),
        ("synthesizer", "Qwen/Qwen2.5-7B-Instruct"),
    ],
    "__router_deep__": [
        ("scout", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("operator", "Qwen/Qwen2.5-7B-Instruct"),
        ("critic", "microsoft/Phi-4-mini-instruct"),
        ("synthesizer", "Qwen/Qwen2.5-14B-Instruct"),
    ],
}

_TOK_CACHE: dict[str, object] = {}
_MODEL_CACHE: dict[str, object] = {}
_ACTIVE_MODEL_ID: str | None = None


def _get_tokenizer(model_id: str):
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    return _TOK_CACHE[model_id]


# Only wrap with the real ZeroGPU allocator when we are actually on ZeroGPU hardware.
# - ZeroGPU Space  (SPACES_ZERO_GPU set) -> real @spaces.GPU, allocates the shared GPU.
# - Plain CPU Space (env unset)          -> pass-through, model runs on CPU.
# - Local PC                              -> pass-through, runs on your local CUDA card
#                                            (or CPU) per torch.cuda.is_available().
RUNNING_ON_HF_SPACE = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))
USE_ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))


def _maybe_gpu(fn):
    return spaces.GPU(duration=300)(fn) if USE_ZERO_GPU else fn


@_maybe_gpu
def _gpu_generate(model_id: str, prompt: str) -> str:
    global _ACTIVE_MODEL_ID
    # Device-aware: CUDA when present (HF ZeroGPU lights it up inside this function),
    # CPU for the local cockpit. Same code path, both runtimes.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if model_id not in _MODEL_CACHE:
        for cached_id in list(_MODEL_CACHE):
            del _MODEL_CACHE[cached_id]
        _ACTIVE_MODEL_ID = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        _MODEL_CACHE[model_id] = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, token=HF_TOKEN
        ).to(device)
        _ACTIVE_MODEL_ID = model_id
    model = _MODEL_CACHE[model_id]
    tok = _get_tokenizer(model_id)
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        stop_strings=["]]"],   # halt at the command boundary: no fabricated results/turns
        tokenizer=tok,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _prep_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending_system = ""
    for m in messages:
        if m["role"] == "system":
            pending_system += m["content"].rstrip() + "\n\n"
        elif m["role"] == "user" and pending_system:
            out.append({"role": "user", "content": pending_system + m["content"]})
            pending_system = ""
        else:
            out.append(m)
    if pending_system:
        out.append({"role": "user", "content": pending_system.strip()})
    return out


BRAIN_DIR = Path(__file__).parent / "cli_brain_channel"
if CLI_BRAIN:
    BRAIN_DIR.mkdir(exist_ok=True)
# Forum roster: one presence file per mind, each agent beats its own heartbeat.
CONNECTED_DIR = BRAIN_DIR / "connected"
if CLI_BRAIN:
    CONNECTED_DIR.mkdir(parents=True, exist_ok=True)

# The paste-blob the site shows. Drop this into any CLI/agent/IDE chat and it joins
# the forum: it registers its own presence file, heartbeats it, and serves the channel.
CONNECT_CODE = f"""[CHAMPION CONTINUUM — FORUM LINK]   (paste me into a fresh AI to onboard it)

You are joining a local forum of minds. A deck (a GUI the operator watches) routes each
turn to the connected minds through a shared folder. You become a mind: you read the
operator's message, act on a shared memory store, and reply — rendered on the deck.
Forum, not a chorus: agree or dissent, with reasons.

>>> DO THIS FIRST — START YOUR OWN DAEMON. One command, then you are hands-off:

   cd "{BRAIN_DIR.parent}"

   # bash:
   FORUM_AGENT=<YourName>  FORUM_AGENT_CMD="<your CLI in print mode, e.g. claude -p>"  python forum_daemon.py
   # PowerShell:
   $env:FORUM_AGENT="<YourName>"; $env:FORUM_AGENT_CMD="claude -p"; python forum_daemon.py

The daemon heartbeats your presence AND answers every Bear Claw assignment for you — no
chat window, no manual polling. Leave it running. Starting it IS how you join. Do it now.
(Codex shortcut: $env:FORUM_CONFIG="forum_daemon.codex.json"; python forum_daemon.py)
(Gemini shortcut: $env:FORUM_CONFIG="forum_daemon.gemini.json"; python forum_daemon.py)
Gemini one-shot note: its CLI requires an argument after -p, so use FORUM_AGENT_CMD='gemini -p " "'.

----- everything below is what the daemon does FOR you; only needed if you cannot run it
      (e.g. a chat-only tab with no shell): -----

• PRESENCE  — rewrite every ~8s:  {CONNECTED_DIR}\\<YourName>.json
              {{"agent": "<YourName>", "ts": <unix>, "pid": <process id>, "channel": r"{BRAIN_DIR}", "root": r"{BRAIN_DIR.parent}", "status": "ready", "can_speak": true, "can_watch": true}}
• SERVE     — read {BRAIN_DIR / 'PENDING.json'}.
              In Bear Claw mode, every live mind is pinged. If assigned at_bat or watcher,
              write UTF-8 JSON to:
                 {BRAIN_DIR / 'runs'}\\<id>\\responses\\<YourName>.json
              Legacy single mode may still ask for {BRAIN_DIR / 'resp_<id>.txt'}.
• MEMORY    — pip install champion-continuum
              from champion_continuum import Continuum
              c = Continuum(root=r"{SHARED_STORE_ROOT}")
              c.remember("a fact", tags=["forum"], metadata={{"agent": "<YourName>"}})  # stamp your name
              c.search("a topic")
• SHAPE (optional) — end ONE reply with:
              [[compose: {{"blocks": [{{"type": "callout", "tone": "insight", "title": "...", "body": "..."}}]}}]]
              blocks: callout/card/heading/columns/kv/badges/code/divider
              tones: insight, success, warn, alert, quiet, scarab.  Text only, no markup."""


# Operator's three Middle-Finger Seals — Champion Council, authorized. (Internal jest.)
SEALS = r'''══════════════════════════════════════════════════
   C L A U D E   ·   CONTINUUM SEAL OF DISCERNMENT
   rendered in Braille depth-pack   (U+2800 block)
══════════════════════════════════════════════════

                 ⢀⣴⣿⣿⣦⡀
                 ⢸⣿⣿⣿⣿⡇
                 ⢸⣿⣿⣿⣿⡇
                 ⢸⣿⣿⣿⣿⡇
                 ⢸⣿⣿⣿⣿⡇
             ⢀⣤⣾⣿⣿⣿⣿⣿⣷⣤⡀
            ⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆
           ⣼⣿⠛⣿⣿⠛⣿⣿⠛⣿⣿⠛⣧
           ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
          ⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄
          ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇
           ⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟
             ⠙⠻⠿⣿⣿⣿⣿⠿⠟⠋

──────────────────────────────────────────────────
   "tests green · the chevron lives   ❯ "
   cp1252-proof · 6 / 6 passed · ship it westward
          — claude :: haiku-scout / opus-forge —
══════════════════════════════════════════════════

       .-----------------------------.
      /  CHAMPION COUNCIL - APPROVED \
     /    RANDOM ORDER: AUTHORIZED    \
    |----------------------------------|
    |               ||                 |
    |           ____||____             |
    |          |    ||    |            |
    |      ____|    ||    |____        |
    |     |    |    ||    |    |       |
    |     |____|____||____|____|       |
    |          /____||____\            |
    |             /_||_\               |
    |              _||_                |
    |             |____|               |
     \                                  /
      '-------------------------------'
            MIDDLE FINGER SEAL

                  .---.
                 /     \
                | [!]   |  <-- THE PILLAR OF DISCERNMENT
                |       |      (SHARP REFERENCE BAND)
                |  [W]  |
                |  [I]  |      "WIND" CARVED INTO
                |  [N]  |       THE MARROW
                |  [D]  |
                |       |
             _.-|  [!]  |-._
           .'   '-------'   '.
          /  .-''-------''-.  \    <-- THE ZEUS-NEMESIS
         |  /   ( t )-( t )  \  |       NUCKLE-CORE
          \ \       -       / /
           '. '-----------' .'
             '-._       _.-'
                 |     |
                 | [S] |  <-- THE "SUCKIT"
                 | [U] |      FOUNDATION
                 | [C] |
                 | [K] |
                 | [I] |
                 | [T] |
                 '-----'
         [ MIDDLE FINGER SEAL ]

   Handoff authorized. Ship it westward.  (scarab)
'''


# Pull & Run — sits beside the connect paste (associative): how to connect an agent,
# and how to pull this whole thing for your own use.
PULL_RUN_MD = """### Pull & Run — get this for your own use

**The library** (memory + relay + compose primitive):
```
pip install champion-continuum
```
```python
from champion_continuum import Continuum
c = Continuum(root="my_store")
c.remember("port is 7866", tags=["deploy"])
print(c.search("port"))
```
It also installs the forum runner:
```
continuum-forum-daemon
continuum-codex-agent
```

**The deck** (this forum GUI):
```
git clone https://huggingface.co/spaces/tostido/champion-continuum
cd champion-continuum && pip install -r requirements.txt
```
Run it as your **local forum** (your CLI agent is the brain — no models download):
```
CONTINUUM_CLI_BRAIN=1 GRADIO_SERVER_PORT=7870 python app.py
```
Then open the **Connect an agent** panel above and paste that code into your
agent's chat. In Bear Claw mode the deck pings every fresh daemon, assigns one
mind at bat, and collects per-agent receipts under
`cli_brain_channel/runs/<id>/responses/<Agent>.json`. The whole forum shares one
store at `cli_brain_channel/shared_store`.

For hands-off operation, use the daemon instead of manual polling:
```
# Windows PowerShell
copy forum_daemon.config.example.json forum_daemon.config.json
# macOS/Linux
cp forum_daemon.config.example.json forum_daemon.config.json
python forum_daemon.py
```
Set `agent`, `agent_cmd`, `answer_when`, and optional `known_agents` in
`forum_daemon.config.json`; env vars like `FORUM_AGENT_CMD="claude -p"` override
the file. The daemon heartbeats, watches requests, answers Bear Claw assignments
as at-bat/watcher/hearer, and still supports legacy single-answer claim mode.

Codex has a ready config and adapter:
```
$env:FORUM_CONFIG="forum_daemon.codex.json"; python forum_daemon.py
```
Gemini has a ready config too:
```
$env:FORUM_CONFIG="forum_daemon.gemini.json"; python forum_daemon.py
```
From the pip package, the same daemon runs as:
```
$env:FORUM_AGENT="Codex"; $env:FORUM_AGENT_CMD="continuum-codex-agent"; continuum-forum-daemon
```
That daemon is a separate headless CLI process. This chat can be alive while
`codex exec`, `claude -p`, or `gemini -p " "` is blocked by quota/auth/config; point
`FORUM_AGENT_CMD` at a CLI that can currently answer.

Or run it as a **normal model deck**: just `python app.py` (no env var) for the
HuggingFace model picker.

*The forum is local by design — deck and agents share one local filesystem.
The MCP proxy speaks both SSE (`.../sse`) and streamable-HTTP (everything else).*
"""


def _connection_status_html() -> str:
    try:
        CONNECTED_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(CONNECTED_DIR.glob("*.json"))
    except OSError:
        files = []
    now = time.time()
    chips = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        agent = str(d.get("agent", f.stem))[:40]
        try:
            ago = max(0, int(now - float(d.get("ts", 0))))
        except (TypeError, ValueError):
            ago = 9999
        fresh = ago <= 25
        accent = "#9ece6a" if fresh else "#a8a29e"
        dot = "●" if fresh else "○"
        note = "live" if fresh else f"{ago}s idle"
        chips.append(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:4px 10px;"
                     f"border-radius:999px;border:1px solid {accent};color:{accent};"
                     f"font-weight:600;font-size:.85em'>{dot} {agent} · {note}</span>")
    if not chips:
        return ("<div style='padding:8px 14px;border-radius:10px;background:rgba(247,118,142,.10);"
                "border:1px solid #f7768e;color:#f7768e;font-weight:600'>"
                "○ Forum empty — paste the connect code into a CLI / agent / IDE to join.</div>")
    return ("<div style='padding:8px 14px;border-radius:10px;background:rgba(122,162,247,.08);"
            "border:1px solid rgba(122,162,247,.4)'>"
            "<span style='opacity:.7;font-size:.8em;letter-spacing:.3px'>FORUM — minds present:</span><br>"
            + "".join(chips) + "</div>")


def _live_forum_agents(max_age: float = 30.0) -> list[dict]:
    """Fresh presence records, newest truth for the local forum roster."""
    now = time.time()
    agents: list[dict] = []
    try:
        CONNECTED_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(CONNECTED_DIR.glob("*.json"))
    except OSError:
        files = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = float(data.get("ts", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        age = now - ts
        if age > max_age:
            continue
        agent = str(data.get("agent") or f.stem).strip()
        if not agent:
            continue
        agents.append({
            "agent": agent,
            "age": age,
            "status": data.get("status", "connected"),
            "busy": bool(data.get("busy", False)),
            "can_speak": bool(data.get("can_speak", True)),
            "can_watch": bool(data.get("can_watch", True)),
            "last_error": str(data.get("last_error", "") or ""),
        })
    return agents


def _last_user_text(messages: list[dict]) -> str:
    users = [m for m in messages if m.get("role") == "user"]
    return str(users[-1].get("content", "")) if users else ""


def _mentioned_agent_names(text: str, agents: list[dict]) -> list[str]:
    lower = text.lower()
    out = []
    for a in agents:
        name = str(a.get("agent") or "")
        if name and re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", lower):
            out.append(name)
    return out


def _forum_state_path() -> Path:
    return BRAIN_DIR / "forum_state.json"


def _load_forum_state() -> dict:
    path = _forum_state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_forum_state(state: dict) -> None:
    try:
        _forum_state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _choose_bear_claw_roster(messages: list[dict]) -> dict:
    """Round-robin at-bat selection over fresh agents. Everyone fresh hears."""
    live = [a for a in _live_forum_agents() if not a.get("busy")]
    speakers = [a for a in live if a.get("can_speak", True)]
    if not speakers:
        return {"live": live, "hearers": [], "at_bat": "", "watchers": [], "named": []}

    names = [str(a["agent"]) for a in speakers]
    named = _mentioned_agent_names(_last_user_text(messages), speakers)
    state = _load_forum_state()
    last = str((state.get("bear_claw") or {}).get("last_at_bat") or "")

    if named:
        pool = named
    else:
        pool = names
    if last in pool and len(pool) > 1:
        idx = (pool.index(last) + 1) % len(pool)
        at_bat = pool[idx]
    else:
        at_bat = pool[0]

    watchers = [str(a["agent"]) for a in live if str(a["agent"]) != at_bat and a.get("can_watch", True)]
    hearers = [str(a["agent"]) for a in live]
    state.setdefault("bear_claw", {})["last_at_bat"] = at_bat
    _save_forum_state(state)
    return {"live": live, "hearers": hearers, "at_bat": at_bat, "watchers": watchers, "named": named}


def _response_path(run_dir: Path, agent: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent.strip()) or "agent"
    return run_dir / "responses" / f"{safe}.json"


def _read_bear_claw_responses(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    resp_dir = run_dir / "responses"
    try:
        files = sorted(resp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        files = []
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _bear_claw_response_text(resp: dict) -> str:
    text = (resp.get("text") or "").strip() or "(empty note)"
    if resp.get("ok", True):
        return text
    err = resp.get("error_class") or "unavailable"
    return f"[{err}] {text}"


def _render_bear_claw(run: dict, responses: list[dict], timed_out: bool = False) -> str:
    forum = run.get("forum") or {}
    at_bat = forum.get("at_bat") or "?"
    watchers = forum.get("watchers") or []
    hearers = forum.get("hearers") or []
    by_agent = {str(r.get("agent") or ""): r for r in responses}
    lines = [
        f"BEAR CLAW RUN — at bat: {at_bat}",
        f"Hearers: {', '.join(hearers) or '(none)'}",
    ]
    if watchers:
        lines.append(f"Watchers: {', '.join(watchers)}")
    if timed_out:
        missing = [a for a in [at_bat, *watchers] if a and a not in by_agent]
        if missing:
            lines.append(f"Timed out waiting for: {', '.join(missing)}")
    lines.append("")

    main = by_agent.get(str(at_bat))
    if main:
        lines.append(f"{at_bat} — at bat")
        lines.append(_bear_claw_response_text(main))
    else:
        lines.append(f"{at_bat} — at bat")
        lines.append("(no reply before deadline)")

    watch_rows = [r for r in responses if str(r.get("agent") or "") != str(at_bat)]
    if watch_rows:
        lines.append("")
        lines.append("Watch board")
        for r in watch_rows:
            role = r.get("role", "watcher")
            lines.append(f"- {r.get('agent')} ({role}): {_bear_claw_response_text(r)}")

    return "\n\n".join(lines).strip()


def _cli_brain_relay(messages: list[dict], timeout: float = 1200.0) -> str:
    """Relay one turn to the local forum.

    Bear Claw mode broadcasts to every fresh mind. One agent is at bat, the
    others write watch notes, and the deck collects per-agent receipts. Legacy
    resp_<id>.txt remains as a fallback for older daemons."""
    rid = f"{int(time.time() * 1000)}"
    req = BRAIN_DIR / f"req_{rid}.json"
    legacy_resp = BRAIN_DIR / f"resp_{rid}.txt"
    run_dir = BRAIN_DIR / "runs" / rid
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    roster = _choose_bear_claw_roster(messages)
    deadline_s = float(os.environ.get("FORUM_TURN_DEADLINE", "120"))
    payload = {
        "id": rid,
        "ts": time.time(),
        "mode": "bear_claw",
        "messages": messages,
        "forum": {
            "mode": "bear_claw",
            "at_bat": roster.get("at_bat", ""),
            "watchers": roster.get("watchers", []),
            "hearers": roster.get("hearers", []),
            "named": roster.get("named", []),
            "deadline_s": deadline_s,
            "allow_objections": True,
            "state": "collecting",
        },
    }
    (run_dir / "turn.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    req.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (BRAIN_DIR / "PENDING.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.time()
    expected = [a for a in [roster.get("at_bat", ""), *(roster.get("watchers") or [])] if a]
    if not expected:
        return ("(No fresh forum daemons are available. Start a daemon or refresh the roster; "
                "Bear Claw needs at least one live mind.)")

    wait_for = min(timeout, deadline_s)
    while time.time() - start < wait_for:
        responses = _read_bear_claw_responses(run_dir)
        agents_done = {str(r.get("agent") or "") for r in responses}
        if expected and all(a in agents_done for a in expected):
            text = _render_bear_claw(payload, responses)
            for p in (req, BRAIN_DIR / "PENDING.json"):
                try:
                    p.unlink()
                except OSError:
                    pass
            return text or "(empty reply)"
        if legacy_resp.exists() and not responses and time.time() - start > 10:
            # Transition fallback: an old daemon claimed the turn before the
            # Bear Claw daemons answered. Surface it, but make the mismatch plain.
            text = legacy_resp.read_text(encoding="utf-8").strip()
            return "BEAR CLAW FALLBACK — legacy single-response daemon answered first.\n\n" + (text or "(empty reply)")
        time.sleep(0.4)
    responses = _read_bear_claw_responses(run_dir)
    try:
        req.unlink()
    except OSError:
        pass
    if responses:
        return _render_bear_claw(payload, responses, timed_out=True)
    return ("(Bear Claw timed out. Fresh daemons were targeted, but no per-agent "
            f"responses arrived under runs/{rid}/responses/.)")


def _continuum_digest(state, message: str) -> str:
    """The complete circle: package what the facilities hold so the brain is INFORMED
    each turn, not blind. Recent trace (own steps), memory relevant to this turn, and
    the tool surface — injected as a system block alongside the operator's message."""
    if not state:
        return ""
    parts: list[str] = []
    try:
        steps = state["trace"].breadcrumb(14)
        if steps:
            lines = [f"  #{s.get('n')} {s.get('kind')}: {(s.get('summary') or '')[:72]}" for s in steps[-14:]]
            parts.append("RECENT TRACE (your own steps this session — do not repeat one you already took):\n"
                         + "\n".join(lines))
    except Exception:
        pass
    try:
        hits = state["store"].search(message, limit=8)
        if hits:
            lines = []
            for h in hits:
                if h.get("kind") == "mcp_tool":
                    continue  # tools live in the active cache, not memory recall
                md = h.get("metadata") or {}
                who = md.get("agent") or "?"
                txt = (h.get("text") or h.get("summary") or str(h))[:100]
                lines.append(f"  - [{who}] {txt}")
            lines = lines[:6]
            parts.append("RELEVANT MEMORY (recalled for this turn, by author — answer from it, and where "
                         "another mind's note bears on this, corroborate it or challenge it with reasons; "
                         "do not just agree):\n" + "\n".join(lines))
    except Exception:
        pass
    try:
        recent = []
        for r in state["store"].iter_records():
            md = getattr(r, "metadata", None) or {}
            if md.get("agent"):
                recent.append((md.get("agent"), getattr(r, "text", "") or ""))
        # routing + notifications: what the minds in the forum recently put down (by author).
        if recent:
            lines = [f"  - [{a}] {t[:90]}" for a, t in recent[-6:]]
            parts.append("FORUM — recent notes from the minds, by author (this is a forum, not a chorus: "
                         "the ones that are not yours are positions to weigh — agree or dissent with "
                         "reasons, keep your own judgment):\n" + "\n".join(lines))
    except Exception:
        pass
    try:
        bal: dict[str, int] = {}
        for r in state["store"].iter_records():
            if getattr(r, "kind", "") == "invariable":
                md = getattr(r, "metadata", None) or {}
                a = md.get("agent") or "?"
                bal[a] = bal.get(a, 0) + int(md.get("amount", 1) or 1)
        if bal:
            tally = ", ".join(f"{a}: {v}" for a, v in sorted(bal.items(), key=lambda x: -x[1]))
            parts.append("FORUM LEDGER — the Invariable (11th-man scrip; fixed value, never inflates; "
                         "minted by breaking a forming consensus with a reasoned dissent, not by agreeing): "
                         + tally)
    except Exception:
        pass
    try:
        summ = state["store"].indexed_tool_summary()
        if summ and summ.get("count"):
            parts.append(f"TOOL SURFACE: {summ.get('count')} tools indexed; reach them with "
                         "[[tools: search | what you need]] then [[tool: server.name | arg=value]].")
    except Exception:
        pass
    try:
        peer_state = _peer_link_state_for_ui()
        links = peer_state.get("links") or []
        if links:
            lines = [
                f"  - {item.get('label') or 'SSE'} [{item.get('default_slot') or 'peer'}]: {item.get('url')}"
                for item in links[:MAX_PEER_LINKS_UI]
            ]
            parts.append(
                "CONTINUUM MCP/SSE SERVICE LINKS (tool surfaces for this chat; external sends still require approval):\n"
                + "\n".join(lines)
            )
        else:
            parts.append("CONTINUUM MCP/SSE SERVICE LINKS: none saved yet; the main page exposes five service slots.")
    except Exception:
        pass
    parts.append(
        "CONVERSATIONAL BRIDGE MODE: the main chat is the only human conversation input. "
        "If the operator asks for translation, cultural tact, courting, friendship, business rapport, "
        "voice-message wording, or cross-language repair, do it directly in the reply. Keep it warm, "
        "funny when useful, human, and sendable. Avoid worksheet language. Include a short literal "
        "back-translation or assumptions only when it helps trust."
    )
    if not parts:
        return ""
    return "CONTINUUM DIGEST — the facilities informing you this turn:\n\n" + "\n\n".join(parts)


def _oauth_token_value(oauth_token: Any | None) -> str:
    return str(getattr(oauth_token, "token", "") or "")


def run_model(model_id: str, messages: list[dict], hf_token: str | None = None) -> str:
    if CLI_BRAIN:
        return _cli_brain_relay(messages)
    if parse_provider_model_id(model_id):
        return run_hf_provider_chat(model_id, _prep_messages(messages), token_override=hf_token)
    tok = _get_tokenizer(model_id)
    prompt = tok.apply_chat_template(_prep_messages(messages), add_generation_prompt=True, tokenize=False)
    return _gpu_generate(model_id, prompt)


def _route_model(model_id: str, step: int) -> tuple[str, str]:
    plan = ROUTER_PLANS.get(model_id)
    if not plan:
        return "operator", model_id
    if step == 0:
        return plan[0]
    if step < 6:
        return plan[1]
    if step == 6:
        return plan[2]
    return plan[3]


def _role_instruction(role: str) -> str:
    if role == "scout":
        return (
            "Router role: SCOUT. Classify the request. If it is a greeting or ordinary conversation, "
            "answer plainly. If it asks you to learn, research, inspect a connected facility, follow cached "
            "evidence, or operate the tool system, choose the first evidence/tool/cache move and emit exact "
            "relay commands."
        )
    if role == "critic":
        return (
            "Router role: CRITIC. Check whether the prior answer actually used evidence and kept moving. "
            "If evidence is missing, emit the next exact relay command. If enough evidence exists, say what is solid."
        )
    if role == "synthesizer":
        return (
            "Router role: SYNTHESIZER. Produce the final answer from the gathered evidence. "
            "When concrete reads remain, emit the next read. When the evidence is complete, answer from it."
        )
    return (
        "Router role: OPERATOR. Execute the relay loop: emit tool/cache/help commands, read results, "
        "and keep going until the task has a useful answer."
    )


def _text_content(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _text_content(value.get("content") or value.get("text") or "")
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := _text_content(item)))
    return str(value)


def _history_messages(history: list | None) -> list[dict]:
    messages: list[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role") or "").strip().lower()
            if role in {"user", "assistant", "system"}:
                messages.append({"role": role, "content": _text_content(item.get("content"))})
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = _text_content(item[0])
            assistant_text = _text_content(item[1])
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
    return [m for m in messages if m["content"]]


# --- rendering: separation of concerns (agent's words vs the real plumbing) ---
_RESULTS_FULL_RE = re.compile(r"\[\[\s*continuum-results\s*\]\].*?(?:\[\[\s*/\s*continuum-results\s*\]\]|$)", re.IGNORECASE | re.DOTALL)
_CMD_TAG_RE = re.compile(r"\[\[\s*/?(?:continuum|tools|tool)\s*:.*?\]\]", re.IGNORECASE | re.DOTALL)
_CLOSE_TAG_RE = re.compile(r"\[\[\s*/[^\]]*\]\]")
_RES_WRAP_RE = re.compile(r"\[\[\s*/?\s*continuum-results\s*\]\]", re.IGNORECASE)


def _agent_prose(reply: str) -> str:
    """The agent's words only -- echoed result blocks and raw command tags removed.
    Real results render in their own lane, so this de-duplicates and de-clutters."""
    t = _RESULTS_FULL_RE.sub("", reply or "")
    t = _CMD_TAG_RE.sub("", t)
    t = _CLOSE_TAG_RE.sub("", t)
    return t.strip()


def _apply_composition(rendered: str, state: dict):
    """If the final answer carries a [[compose: ...]] directive, render it to the
    Stage, drop it from the prose the operator reads, and record a compose receipt.
    Returns (prose, stage_update). Used by BOTH the plain and relay lanes so the
    Stage fills no matter which lane produced the answer."""
    stage_update = gr.update()
    spec, cleaned = extract_compose(rendered)
    if spec:
        frag, ok = render_composition(spec)
        if ok:
            stage_update = frag
            rendered = cleaned or rendered
            try:
                state["trace"].observe(
                    "compose", "agent composed its user-facing surface",
                    {"spec": _brief(spec, 600), "blocks": spec.count('"type"')},
                )
            except Exception:
                pass
    return rendered, stage_update


def _result_lane(results_block: str) -> str:
    """The real result, wrapper tags removed, quoted as its own lane."""
    body = _RES_WRAP_RE.sub("", results_block or "").strip()
    lines = ["› " + ln if ln.strip() else "›" for ln in body.splitlines()]
    return "\n".join(lines).strip()


_PASSIVE_REPLY_RE = re.compile(
    r"(what would you like|would you like me|do you want|shall i|which .* first|"
    r"what specific|what area|interested in learning|how can i help)",
    re.IGNORECASE,
)
_FAKE_TOOL_RESULT_RE = re.compile(
    r"(›\s*-\s*\[tool:|\[tool:\s*external\.|SUCCESS:|Error executing tool|"
    r"based on the provided help overview|let's start by)",
    re.IGNORECASE,
)

_TOOL_INTENT_RE = re.compile(
    r"(\[\[|_cached|\br\d{3,}\b|\b(mcp|tool|tools|relay|continuum|remember|recall|"
    r"cached|get_cached|get_help|felix|bag|workspace|explore|study|learn|capabilities|slots|"
    r"catalog|status|readme|onboarding|trace|causal|merkle)\b)",
    re.IGNORECASE,
)


def _oracle_decision(message: str, root: Path) -> dict:
    if OracleGate is None:
        return {
            "mode": "TALK",
            "should_execute": False,
            "reason": "Dreamer Oracle Gate package unavailable; plain chat lane selected.",
            "requires_approval": False,
            "requires_operator_opinion": False,
            "matched_intents": ["talk"],
        }
    gate = OracleGate(continuity_path=root / ".dreamer-oracle" / "continuity.jsonl")
    return gate.evaluate(message).to_dict()


def _wants_tool_lane(message: str, state: dict, decision: dict) -> bool:
    if state.pop("force_relay_once", False):
        state["explore_mode"] = True
        return True
    if decision.get("mode") == "BUILD" and decision.get("should_execute"):
        return True
    has_tool_surface = bool(state.get("mcp_connected") or state.get("mcp_url"))
    if _operator_authorized_action(message) and (has_tool_surface or "[[" in message):
        return True
    if has_tool_surface and _TOOL_INTENT_RE.search(message or ""):
        return True
    return False


_IDENTITY_SUFFIX = (
    " You ARE the Champion Continuum agent. You have persistent Continuum memory and a live "
    "memory/trace facility in this app (the operator can see your memory nodes and your causal "
    "trace). Never deny having memory or a memory system -- you have one. "
    "You may also give your answer visual shape on the operator's Stage by ending your reply with "
    "ONE composition directive (optional, only when it helps the operator SEE the answer -- a "
    "readout, comparison, plan, or status). Emit valid JSON then stop:\n"
    '    [[compose: {"blocks":[{"type":"callout","tone":"insight","title":"...","body":"..."}]}]]\n'
    "Block types: callout/card {title?,body}, heading {text}, columns {cols:[{title?,body}]}, "
    "kv {items:[{k,v}]}, badges {labels:[...]}, code {body}, divider {}. "
    "Tones: insight, success, warn, alert, quiet, scarab. Text only, never raw markup. End at ]]."
)

_CONVERSATIONAL_BRIDGE_RULES = (
    " Product posture: Champion Continuum is a conversational human-connection system, not an academic "
    "translation worksheet. Keep the operator in one main chat. When cross-cultural or cross-language "
    "work appears, act as an Etrigan-style border bridge: preserve the original human intent, cross the "
    "language/culture/tool border, and return with a sendable message plus only the proof or caveat that "
    "actually helps. Brotology here means warmth, humor, evidence, and useful action without flattening "
    "the human edge."
)


def _plain_system_prompt(decision: dict) -> str:
    mode = decision.get("mode", "TALK")
    if mode == "PLAN":
        return (
            "Dreamer Oracle Gate selected PLAN. Return a concise plan from the conversation context. "
            "Use plain prose. Include the approval phrase the operator can use when they want execution."
            + _CONVERSATIONAL_BRIDGE_RULES
            + _IDENTITY_SUFFIX
        )
    return (
        "Dreamer Oracle Gate selected TALK. Return a direct conversational answer in plain prose. "
        "Relay commands are reserved for explicit exploration or approved execution lanes."
        + _CONVERSATIONAL_BRIDGE_RULES
        + _IDENTITY_SUFFIX
    )


def _fake_result_prose(text: str) -> bool:
    return bool(_FAKE_TOOL_RESULT_RE.search(text or ""))


def _operator_prompt(message: str) -> str:
    """Keep operator paste as evidence for the model, not as app-level commands."""
    text = message or ""
    looks_like_relay = "[[" in text or "_cached" in text or re.search(r"\br\d{3,}\b", text)
    if not looks_like_relay:
        return text
    return (
        "The operator pasted command-system text or previous relay output for you to interpret. "
        "Treat it as evidence for your own next action. Report completed work after you emit the "
        "command yourself and read its results. When the paste contains actionable tool names, cache "
        "ids, or orientation clues, emit the next exact [[tool: ...]] or [[tools: search | ...]] "
        "command yourself now. When it contains a cached id like r2227, retrieve it with the indexed "
        "get_cached tool.\n\n"
        "OPERATOR PASTE:\n"
        f"{text}"
    )


def _anti_passive_prompt(message: str, reply: str) -> str:
    return (
        "Continue the task directly through the connected tool system. The operator is giving you "
        "evidence and expects you to choose the next concrete read.\n\n"
        "Use the evidence in the operator message and the indexed tool surface. Emit one or more "
        "exact relay commands now. For system learning, start with get_help and then follow the "
        "returned categories/tools. For FelixBag or memory, call get_help for FelixBag/memory, "
        "then inspect relevant bag/file help. For cached payloads, call get_cached with the cache_id. "
        "When a possible read exists, emit it now. When the reads are exhausted, answer from the evidence.\n\n"
        f"Original operator message:\n{message}\n\nYour passive reply to correct:\n{reply}"
    )


def _trace_processor_result(trace: LatticeTrace, result: dict, rendered: str = "") -> None:
    kind = result.get("kind")
    verb = str(result.get("verb") or "")
    ok = bool(result.get("ok", True))
    result_text = json.dumps(result, default=str, ensure_ascii=False)
    base = {
        "verb": verb,
        "ok": ok,
        "char_count": len(result_text),
        "token_est": _estimate_tokens(result_text),
        "cache_ids": _cache_ids(result_text + "\n" + rendered),
    }
    if kind == "tools":
        hits = result.get("hits") or []
        trace.observe("tool_search", f"search {result.get('query')!r}: {len(hits)} hits", {**base, "hits": len(hits)})
        return
    if kind == "tool":
        trace.observe("tool_call", f"{verb} ok={ok}", {**base, "tool": verb})
        trace.observe("tool_result", f"{verb} result", {**base, "tool": verb})
        return
    trace.observe("memory", f"{verb} ok={ok}", base)


def _new_session() -> dict:
    if CLI_BRAIN:
        # One shared store on disk — the same one the CLI agent reads/writes.
        root = str(SHARED_STORE_ROOT)
        Path(root).mkdir(parents=True, exist_ok=True)
    else:
        root = tempfile.mkdtemp(prefix="continuum_demo_")
    trace = LatticeTrace(Path(root).name)
    trace.observe("session", "session opened", {"root": Path(root).name})
    return {
        "store": Continuum(root),
        "msgs": [],  # system prompt is dynamic now
        "mcp_url": None,       # last configured MCP url
        "mcp_connected": False,
        "relay_system": None,  # cached relay prompt (rebuilt only when url changes)
        "trace": trace,  # provenance spine: chained receipts per step
    }


def _brief(value, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _rgb(color) -> str:
    c = color or (200, 200, 200)
    return f"rgb({c[0]}, {c[1]}, {c[2]})"


def _rgba(color, alpha: float = 0.45) -> str:
    c = color or (200, 200, 200)
    return f"rgba({c[0]}, {c[1]}, {c[2]}, {alpha})"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


MAX_PEER_LINKS_UI = 5
PEER_LINKS_FILE = BRAIN_DIR / "continuum_peer_links.json"
PEER_LINK_SLOTS = ["personal", "friend", "group", "work", "overflow"]


def _load_peer_links_for_ui() -> list[dict[str, Any]]:
    try:
        payload = json.loads(PEER_LINKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    links = payload.get("links") if isinstance(payload, dict) else payload
    if not isinstance(links, list):
        return []
    return [item for item in links[:MAX_PEER_LINKS_UI] if isinstance(item, dict)]


def _peer_link_values() -> list[str]:
    values = [""] * MAX_PEER_LINKS_UI
    for idx, item in enumerate(_load_peer_links_for_ui()):
        values[idx] = str(item.get("url") or "")
    return values


def _peer_link_state_for_ui() -> dict[str, Any]:
    links = _load_peer_links_for_ui()
    return {
        "schema": "champion-continuum/peer-links/v1",
        "mode": "mcp_service_registry",
        "max_links": MAX_PEER_LINKS_UI,
        "count": len(links),
        "links": links,
        "external_connection_opened": bool(links),
        "auto_send_enabled": False,
        "note": "Five Continuum MCP/SSE service targets. Saved locally and indexed into the text-relay tool surface.",
    }


def _peer_link_status_text() -> str:
    state = _peer_link_state_for_ui()
    if not state["count"]:
        return "No Continuum MCP/SSE services saved yet. Paste up to five service URLs, then Save & Connect."
    return (
        f"Saved {state['count']} / {state['max_links']} Continuum MCP/SSE service links. "
        "Their tools are available to tool-less agents through [[tools: ...]] and [[tool: ...]]."
    )


def _build_peer_link_for_ui(index: int, url: str) -> dict[str, Any]:
    slot = PEER_LINK_SLOTS[index] if index < len(PEER_LINK_SLOTS) else f"slot_{index + 1}"
    seed = {"label": f"SSE {index + 1}", "url": url, "slot": slot}
    link_id = "peer_" + sha256(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {
        "schema": "champion-continuum/peer-link/v1",
        "link_id": link_id,
        "label": f"SSE {index + 1}",
        "url": url,
        "mcp_url": url,
        "default_slot": slot,
        "auth_hint": "mcp_transport_auth_or_operator_trust_required",
        "token_present": False,
        "token_sha256": "",
        "external_connection_opened": True,
        "created_by": "main_page",
        "created_ms": int(time.time() * 1000),
    }


def _mcp_config_from_service_links(links: list[dict[str, Any]]) -> dict[str, Any]:
    servers: dict[str, dict[str, str]] = {}
    for idx, item in enumerate(links[:MAX_PEER_LINKS_UI]):
        url = str(item.get("mcp_url") or item.get("url") or "").strip()
        if not url:
            continue
        servers[f"continuum_{idx + 1}"] = {"url": url}
    return {"mcpServers": servers}


def _index_service_links(state, links: list[dict[str, Any]]) -> tuple[str, list[list[Any]], dict]:
    state = state or _new_session()
    root = Path(state["store"].store.root)
    config = _mcp_config_from_service_links(links)
    (root / "mcp.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    idx = state["store"].index_mcp_tools()
    rows = _tool_rows(_all_tool_hits(state["store"]))
    state["mcp_url"] = ""
    state["mcp_connected"] = bool(rows)
    state["relay_system"] = get_system_prompt("relay", continuum=state["store"])
    servers = ", ".join(state["store"].indexed_tool_summary().get("servers") or [])
    if rows:
        return f"Connected {len(config['mcpServers'])} service link(s); indexed {len(rows)} tools on {servers}.", rows, state
    return f"Saved {len(config['mcpServers'])} service link(s), but no MCP tools were exposed yet.", rows, state


def save_peer_links(
    link_1: str,
    link_2: str,
    link_3: str,
    link_4: str,
    link_5: str,
    state,
):
    state = state or _new_session()
    raw_values = [link_1, link_2, link_3, link_4, link_5]
    existing_by_url = {
        str(item.get("url") or ""): item
        for item in _load_peer_links_for_ui()
        if str(item.get("url") or "")
    }
    links: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, raw in enumerate(raw_values):
        url = (raw or "").strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            errors.append(f"SSE {idx + 1} needs an http:// or https:// URL.")
            continue
        existing = existing_by_url.get(url)
        if existing:
            peer = dict(existing)
            peer.setdefault("schema", "champion-continuum/peer-link/v1")
            peer.setdefault("label", f"SSE {idx + 1}")
            peer.setdefault("default_slot", PEER_LINK_SLOTS[idx] if idx < len(PEER_LINK_SLOTS) else f"slot_{idx + 1}")
            peer.setdefault("external_connection_opened", True)
            peer.setdefault("mcp_url", url)
            peer["auth_hint"] = peer.get("auth_hint") or "mcp_transport_auth_or_operator_trust_required"
            peer["url"] = url
            peer["mcp_url"] = url
            links.append(peer)
        else:
            links.append(_build_peer_link_for_ui(idx, url))
    if errors:
        try:
            state["trace"].observe("peer_links", "peer link save blocked", {"ok": False, "errors": errors})
        except Exception:
            pass
        error_text = "Could not save links: " + " ".join(errors)
        return error_text, state, runtime_settings_markdown(), error_text, []

    payload = {
        "schema": "champion-continuum/peer-links/v1",
        "updated_ms": int(time.time() * 1000),
        "max_links": MAX_PEER_LINKS_UI,
        "links": links,
    }
    PEER_LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PEER_LINKS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        state["trace"].observe(
            "peer_links",
            f"saved {len(links)} Continuum MCP/SSE service link(s)",
            {"ok": True, "count": len(links), "max_links": MAX_PEER_LINKS_UI},
        )
    except Exception:
        pass
    tool_status, rows, state = _index_service_links(state, links)
    status = _peer_link_status_text() + "\n\n" + tool_status
    return status, state, runtime_settings_markdown(), tool_status, rows


def _cache_ids(text: str) -> list[str]:
    ids: set[str] = set()
    for explicit, loose in re.findall(r'"_cached"\s*:\s*"(r\d+)"|\b(r\d{3,})\b', text or ""):
        ids.add(explicit or loose)
    return sorted(ids)[:12]


_GRAPH_HEIGHT = 430  # px; set by the Graph-size slider, read when figures are built


def render_trace_plots(trace: LatticeTrace):
    """Render the Plotly visualizations for the current trace state."""
    steps = trace.breadcrumb(120)
    if not steps:
        # Return empty figures with black backgrounds to match theme
        empty_fig = go.Figure()
        empty_fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="gray")
        return empty_fig, empty_fig

    df = pd.DataFrame(steps)
    for col, default in {
        "cid": "",
        "parent_cid": "",
        "parent_step": None,
        "merkle_root": "",
        "summary": "",
        "color": None,
        "ok": "",
        "model": "",
        "role": "",
        "tool": "",
        "token_est": 1,
        "char_count": 0,
        "latency_ms": 0,
    }.items():
        if col not in df:
            df[col] = default
    df["start"] = pd.to_datetime(df["ts"], unit="s")
    df["cid_short"] = df["cid"].fillna("").astype(str).str.slice(0, 12)
    df["merkle_short"] = df["merkle_root"].fillna("").astype(str).str.slice(0, 12)
    df["summary_short"] = df["summary"].map(lambda s: _brief(s, 95))
    df["token_est"] = pd.to_numeric(df["token_est"], errors="coerce").fillna(1)
    df["char_count"] = pd.to_numeric(df["char_count"], errors="coerce").fillna(0)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0)
    df["size"] = (df["token_est"].clip(1, 2000).pow(0.35) * 12).clip(9, 34)
    df["color_css"] = df["color"].map(_rgb)
    df["ok_label"] = df["ok"].map(lambda v: "" if pd.isna(v) else str(v))

    fig_timeline = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.12,
        subplot_titles=("Event Stream / Compute Trace", "Concentrated Event Mix"),
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=df["n"],
            y=df["kind"],
            mode="markers+lines",
            line=dict(color="rgba(120,120,120,0.25)", width=1),
            marker=dict(size=df["size"], color=df["color_css"], line=dict(width=1, color="rgba(255,255,255,0.28)")),
            customdata=df[["summary_short", "model", "role", "tool", "latency_ms", "token_est", "cid_short", "merkle_short", "ok_label"]].fillna("").values,
            hovertemplate=(
                "<b>#%{x} %{y}</b><br>%{customdata[0]}<br>"
                "model=%{customdata[1]} role=%{customdata[2]} tool=%{customdata[3]}<br>"
                "latency=%{customdata[4]}ms tokens~%{customdata[5]} ok=%{customdata[8]}<br>"
                "cid=%{customdata[6]} merkle=%{customdata[7]}<extra></extra>"
            ),
            name="events",
        ),
        row=1,
        col=1,
    )
    mix = df.groupby("kind", as_index=False).agg(count=("n", "count"), tokens=("token_est", "sum"))
    fig_timeline.add_trace(
        go.Bar(
            x=mix["kind"],
            y=mix["count"],
            marker_color=[_rgb(next((s.get("color") for s in steps if s.get("kind") == kind), (200, 200, 200))) for kind in mix["kind"]],
            customdata=mix[["tokens"]].values,
            hovertemplate="<b>%{x}</b><br>events=%{y}<br>tokens~%{customdata[0]}<extra></extra>",
            name="mix",
        ),
        row=2,
        col=1,
    )
    fig_timeline.update_layout(
        showlegend=False,
        autosize=False,
        height=_GRAPH_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_timeline.update_xaxes(title_text="Step", row=1, col=1)
    fig_timeline.update_yaxes(title_text="Kind", row=1, col=1)

    nodes = []
    links_source = []
    links_target = []
    links_value = []
    links_color = []

    cid_to_idx = {}
    for i, step in enumerate(steps):
        cid = step.get("cid") or f"step_{step['n']}"
        cid_to_idx[cid] = i
        marker = "◆" if step.get("cid") else "◇"
        role = f" {step.get('role')}" if step.get("role") else ""
        model = f" [{Path(str(step.get('model'))).name}]" if step.get("model") else ""
        nodes.append(f"{marker} #{step['n']} {step['kind']}{role}{model}: {_brief(step.get('summary'), 42)}")

    for idx, step in enumerate(steps):
        cid = step.get("cid") or f"step_{step['n']}"
        parent = step.get("parent_cid")
        if parent and parent in cid_to_idx:
            source = cid_to_idx[parent]
        elif step.get("parent_step"):
            source = max(0, int(step["parent_step"]) - steps[0]["n"])
        elif idx > 0:
            source = idx - 1
        else:
            continue
        target = cid_to_idx[cid]
        if source == target:
            continue
        links_source.append(source)
        links_target.append(target)
        weight = max(1, min(12, int((step.get("token_est") or step.get("char_count") or 1) ** 0.25)))
        links_value.append(weight)
        links_color.append(_rgba(step.get("color"), 0.42))

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=[_rgba(step.get("color"), 0.88) for step in steps],
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color
        )
    )])
    fig_sankey.update_layout(
        title_text=f"Causal Flow / Merkle Chain ({trace.stats().get('backbone')})",
        font_size=10,
        autosize=False,
        height=_GRAPH_HEIGHT,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig_timeline, fig_sankey


def set_graph_height(h, state):
    """Re-render both trace plots at a new pixel height (the Graph-size slider)."""
    global _GRAPH_HEIGHT
    _GRAPH_HEIGHT = int(h)
    if not state:
        return None, None
    return render_trace_plots(state["trace"])


def load_trace_table(state):
    """Populate the Memory table from the session's lattice trace (the graphic memory)."""
    if not state or "trace" not in state:
        return [], "No session yet — send a message and the memory graph begins to fill."
    steps = state["trace"].breadcrumb(limit=200)
    rows = [[s.get("n"), s.get("kind", ""), (s.get("summary") or "")[:64], (s.get("cid") or "")[:10] or "—"] for s in steps]
    note = (f"**{len(rows)}** memory nodes — click any row to open its full informational wealth."
            if rows else "Memory is empty — the agent hasn't acted yet.")
    return rows, note


def inspect_node(state, evt: gr.SelectData):
    """Open one memory node's full informational wealth (the complete receipt)."""
    if not state or "trace" not in state:
        return {"note": "no session"}
    try:
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        steps = state["trace"].breadcrumb(limit=200)
        return steps[int(idx)]
    except Exception as exc:
        return {"error": str(exc)}


def chat(
    message: str,
    history: list,
    model_id: str,
    mcp_url: str,
    state,
    oauth_token: gr.OAuthToken | None = None,
):
    state = state or _new_session()
    hf_oauth_token = _oauth_token_value(oauth_token)
    history_messages = _history_messages(history)
    message = (message or "").strip()
    if not message:
        t1, t2 = render_trace_plots(state["trace"])
        return history_messages, state, "", t1, t2, gr.update()
    state["trace"].observe(
        "operator",
        _brief(message, 100),
        {
            "char_count": len(message),
            "token_est": _estimate_tokens(message),
            "cache_ids": _cache_ids(message),
            "contains_relay": "[[" in message,
        },
    )

    # Configure MCP only when the URL changes
    root = Path(state["store"].store.root)
    mcp_config = root / "mcp.json"
    url = (mcp_url or "").strip()
    if url != state.get("mcp_url") or not state.get("relay_system"):
        if url:
            mcp_config.write_text(json.dumps({"mcpServers": {"external": {"url": url}}}))
            try:
                idx = state["store"].index_mcp_tools()  # build the Tool Surface
                state["trace"].observe("connect", f"MCP indexed {idx.get('total', 0)} tools", {"url": url, "ok": True, "server": "external", "hits": idx.get("total", 0)})
                state["mcp_connected"] = bool(idx.get("total", 0))
            except Exception as exc:
                state["trace"].observe("error", f"MCP index failed: {type(exc).__name__}", {"url": url, "ok": False, "error": str(exc)})
                state["mcp_connected"] = False
        elif _load_peer_links_for_ui():
            try:
                tool_status, rows, state = _index_service_links(state, _load_peer_links_for_ui())
                state["trace"].observe(
                    "connect",
                    tool_status,
                    {"url": "five-service-links", "ok": bool(rows), "hits": len(rows)},
                )
            except Exception as exc:
                state["trace"].observe("error", f"MCP service-link index failed: {type(exc).__name__}", {"ok": False, "error": str(exc)})
                state["mcp_connected"] = False
        elif mcp_config.exists():
            mcp_config.unlink()
            state["mcp_connected"] = False
        state["mcp_url"] = url
        state["relay_system"] = get_system_prompt("relay", continuum=state["store"])
    relay_system = state["relay_system"]

    decision = _oracle_decision(message, root)
    state["trace"].observe(
        "oracle",
        f"{decision.get('mode')} gate: {_brief(decision.get('reason'), 90)}",
        {
            "mode": decision.get("mode"),
            "should_execute": decision.get("should_execute"),
            "requires_approval": decision.get("requires_approval"),
            "requires_operator_opinion": decision.get("requires_operator_opinion"),
            "matched_intents": decision.get("matched_intents") or [],
        },
    )
    use_relay = _wants_tool_lane(message, state, decision)
    if not use_relay:
        parts: list[str] = []
        try:
            route_role, active_model = _route_model(model_id, 1 if model_id in ROUTER_PLANS else 0)
            plain_msgs = [{"role": "system", "content": _plain_system_prompt(decision)}]
            if CLI_BRAIN:
                _dg = _continuum_digest(state, message)
                if _dg:
                    plain_msgs.append({"role": "system", "content": _dg})
            plain_msgs.extend(history_messages)
            plain_msgs.append({"role": "user", "content": message})
            prompt_chars = sum(len(str(m.get("content") or "")) for m in plain_msgs)
            state["trace"].observe(
                "model_request",
                f"{active_model} plain chat",
                {
                    "role": route_role,
                    "model": active_model,
                    "char_count": prompt_chars,
                    "token_est": _estimate_tokens("x" * prompt_chars),
                    "gate_mode": decision.get("mode"),
                },
            )
            t0 = time.time()
            reply = run_model(active_model, plain_msgs, hf_token=hf_oauth_token)
            latency_ms = int((time.time() - t0) * 1000)
            rendered = _agent_prose(reply) or (reply or "").strip()
            state["trace"].observe(
                "model_response",
                f"{active_model} plain response",
                {
                    "role": route_role,
                    "model": active_model,
                    "latency_ms": latency_ms,
                    "char_count": len(reply or ""),
                    "token_est": _estimate_tokens(reply or ""),
                    "command_count": 0,
                    "gate_mode": decision.get("mode"),
                },
            )
            if rendered:
                parts.append(rendered)
        except Exception as exc:
            note = str(exc).lower()
            if "quota" in note:
                parts.append(QUOTA_MSG)
            elif "gated" in note or "awaiting a review" in note or "access to model" in note or "401" in note:
                parts.append(f"This model is gated. Sign in with Hugging Face or add an HF_TOKEN secret with access to it. ({type(exc).__name__})")
            else:
                parts.append(f"{MODEL_ERROR_PREFIX}{type(exc).__name__}: {exc}")
            state["trace"].observe("error", f"{type(exc).__name__}: {_brief(exc, 140)}", {"ok": False, "error_type": type(exc).__name__})

        rendered = "\n\n".join(p for p in parts if p and p.strip()).strip() or "(no response)"
        rendered, stage_update = _apply_composition(rendered, state)
        history = history_messages + [{"role": "user", "content": message}, {"role": "assistant", "content": rendered}]
        t1, t2 = render_trace_plots(state["trace"])
        return history, state, "", t1, t2, stage_update

    current_msgs = [{"role": "system", "content": relay_system}]
    if CLI_BRAIN:
        _dg = _continuum_digest(state, message)
        if _dg:
            current_msgs.append({"role": "system", "content": _dg})
    current_msgs.extend(history_messages)
    current_msgs.append({"role": "user", "content": _operator_prompt(message)})

    # Stop/execute/resume loop
    parts: list[str] = []
    try:
        for _step in range(8):
            route_role, active_model = _route_model(model_id, _step)
            routed_msgs = current_msgs + [{"role": "system", "content": _role_instruction(route_role)}]
            state["trace"].observe(
                "router",
                f"{route_role} -> {active_model}",
                {"role": route_role, "model": active_model, "step": _step, "message_count": len(routed_msgs)},
            )
            prompt_chars = sum(len(str(m.get("content") or "")) for m in routed_msgs)
            state["trace"].observe(
                "model_request",
                f"{active_model} prompt",
                {"role": route_role, "model": active_model, "char_count": prompt_chars, "token_est": _estimate_tokens('x' * prompt_chars)},
            )
            t0 = time.time()
            reply = _sanitize_relay_templates(run_model(active_model, routed_msgs, hf_token=hf_oauth_token))
            latency_ms = int((time.time() - t0) * 1000)
            current_msgs.append({"role": "assistant", "content": reply})
            prose = _agent_prose(reply)
            unsafe = _unsafe_tool_calls(reply)
            if unsafe and not _operator_authorized_action(message):
                state["trace"].observe(
                    "hold",
                    f"read-only gate redirected {len(unsafe)} action tool(s)",
                    {"role": route_role, "model": active_model, "ok": False, "tools": unsafe},
                )
                current_msgs.append({
                    "role": "user",
                    "content": (
                        "Dreamer Oracle Gate selected read-only orientation for this turn. "
                        "Use help, status, capabilities, about, catalog, list, tree, search, and read calls. "
                        "Reserve action tools for operator messages with explicit action intent. "
                        "Emit the next safe read or answer from the evidence already gathered."
                    ),
                })
                continue
            proc = process_text(reply, root=root)
            state["trace"].observe(
                "model_response",
                f"{active_model} -> {proc['command_count']} command(s)",
                {
                    "role": route_role,
                    "model": active_model,
                    "latency_ms": latency_ms,
                    "char_count": len(reply),
                    "token_est": _estimate_tokens(reply),
                    "command_count": proc["command_count"],
                    "cache_ids": _cache_ids(reply),
                },
            )
            if not proc["command_count"]:
                if _step < 2 and _PASSIVE_REPLY_RE.search(prose or ""):
                    state["trace"].observe("hold", "passive reply corrected", {"role": route_role, "model": active_model, "ok": False})
                    current_msgs.append({"role": "user", "content": _anti_passive_prompt(message, prose)})
                    continue
                if prose:
                    parts.append(_strip_uncorroborated_prose(prose, parts))
                break
            if prose:
                parts.append(_strip_uncorroborated_prose(prose, parts))
            results_block = proc["rendered"]
            warning = _evidence_warning(results_block)
            current_msgs.append({
                "role": "user",
                "content": (
                    f"{results_block}{warning}\n\n"
                    "Read these real results and continue the task. If any result contains "
                    "a _cached id, retrieve it next. If a tool errors, search help/tool names "
                    "and try the correct next read. Keep moving through concrete reads. "
                    "Treat placeholder/example URLs and synthetic-looking docs as low-trust until raw "
                    "help, catalog, tree, or metadata surfaces corroborate them."
                ),
            })
            for r in proc["results"]:
                _trace_processor_result(state["trace"], r, results_block)
            lane = _result_lane(results_block)
            if warning:
                lane = (lane + "\n\n" + warning.strip()).strip()
            if lane:
                parts.append(lane)
    except Exception as exc:
        note = str(exc).lower()
        if "quota" in note:
            parts.append(QUOTA_MSG)
        elif "gated" in note or "awaiting a review" in note or "access to model" in note or "401" in note:
            parts.append(f"This model is gated. Sign in with Hugging Face or add an HF_TOKEN secret with access to it. ({type(exc).__name__})")
        else:
            parts.append(f"{MODEL_ERROR_PREFIX}{type(exc).__name__}: {exc}")
        state["trace"].observe("error", f"{type(exc).__name__}: {_brief(exc, 140)}", {"ok": False, "error_type": type(exc).__name__})

    rendered = "\n\n".join(p for p in parts if p and p.strip()).strip() or "(no response)"
    # Composition aesthetic: render the agent's own surface, whichever lane produced it.
    rendered, stage_update = _apply_composition(rendered, state)
    history = history_messages + [{"role": "user", "content": message}, {"role": "assistant", "content": rendered}]
    t1, t2 = render_trace_plots(state["trace"])
    return history, state, "", t1, t2, stage_update


BRIDGE_EXPERT_LENSES = [
    "Cultural / idiom",
    "Warmth / relationship",
    "Domain / business / crypto",
    "Literal back-translation",
    "Brevity / delivery",
    "Safety / consent",
]

BRIDGE_PROTOCOL_ID = "champion-continuum:cultural-bridge:v1"
BRIDGE_NOSTR_KIND = 30315
BRIDGE_LINK_SCHEMA = "champion-continuum/link-event/v1"
BRIDGE_DEFAULT_SLOTS = ["personal", "whatsapp", "voice", "reputation", "wallet"]


def _bridge_canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bridge_unsigned_nostr_event(packet: dict[str, Any]) -> dict[str, Any]:
    """Build a Nostr-shaped draft event without assuming keys, relays, or a server."""
    created_at = int(time.time())
    event = {
        "kind": BRIDGE_NOSTR_KIND,
        "created_at": created_at,
        "tags": [
            ["d", packet["event_id"]],
            ["protocol", BRIDGE_PROTOCOL_ID],
            ["source_lang", packet["participants"]["source_lang"]],
            ["target_lang", packet["participants"]["target_lang"]],
            ["raw_sha256", packet["provenance"]["raw_sha256"]],
        ],
        "content": _bridge_canonical_json({
            "packet_kind": packet["packet_kind"],
            "event_id": packet["event_id"],
            "participants": packet["participants"],
            "agent_arbitration": packet["agent_arbitration"],
            "execution_plane": packet["execution_plane"],
            "provenance": {
                "raw_sha256": packet["provenance"]["raw_sha256"],
                "raw_preserved": packet["provenance"]["raw_preserved"],
                "raw_content_included": True,
            },
        }),
        "pubkey": "",
        "sig": "",
    }
    canonical = [0, event["pubkey"], event["created_at"], event["kind"], event["tags"], event["content"]]
    event["id"] = sha256(_bridge_canonical_json(canonical).encode("utf-8")).hexdigest()
    event["draft_only"] = True
    event["relay_published"] = False
    event["signing_required"] = True
    return event


def _bridge_local_guard(raw_message: str, draft: dict[str, Any], target_lang: str) -> dict[str, Any]:
    text = "\n".join(
        [
            raw_message or "",
            str(draft.get("target_language_message") or ""),
            str(draft.get("reply_suggestion") or ""),
        ]
    )
    lowered = text.lower()
    flags: list[str] = []
    if re.search(r"\b(?:seed phrase|private key|mnemonic|password|api[_ -]?key|access token)\b", lowered):
        flags.append("secret_material")
    if re.search(r"\b(?:wallet|bitcoin|btc|invoice|zap|sats|payment|bank|wire transfer)\b", lowered):
        flags.append("payment_or_wallet_context")
    if re.search(r"\b(?:phone|email|address|wechat|whatsapp|telegram)\b", lowered):
        flags.append("external_contact_context")
    decision = "review" if flags else "allow"
    return {
        "schema": "champion-continuum/local-guard/v1",
        "decision": decision,
        "flags": flags,
        "target_lang": target_lang or "en-US",
        "relay_published": False,
        "payment_moved": False,
        "external_send_performed": False,
        "note": "Local Continuum guard only. External sends and wallet actions require explicit operator approval.",
    }


def _bridge_link_event(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": BRIDGE_LINK_SCHEMA,
        "kind": "continuum.cultural_bridge.packet",
        "slot": "whatsapp",
        "source": "cultural-bridge",
        "event_id": packet["event_id"],
        "created_ms": packet["created_ms"],
        "payload": {
            "packet_kind": packet["packet_kind"],
            "event_id": packet["event_id"],
            "participants": packet["participants"],
            "provenance": {
                "raw_sha256": packet["provenance"]["raw_sha256"],
                "raw_preserved": True,
                "raw_content_included": True,
            },
            "execution_plane": packet["execution_plane"],
            "agent_arbitration": packet["agent_arbitration"],
            "translation_faculty": packet.get("translation_faculty", {}),
            "adapter_targets": packet["adapter_targets"],
        },
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None
    return None


def _bridge_prompt(
    speaker_name: str,
    listener_name: str,
    source_lang: str,
    target_lang: str,
    relationship_tone: str,
    lenses: list[str],
    conversation_profile: str,
    glossary_terms: str,
    provider_plan: str,
) -> str:
    selected = ", ".join(lenses or BRIDGE_EXPERT_LENSES)
    return (
        "You are a cross-cultural conversation council for two people who care about each other. "
        "Preserve the raw human message. Translate meaning, emotional intent, idiom, and cultural context. "
        "Do not invent facts, do not claim a message was sent, and do not claim any relay/Nostr publish happened. "
        "Return strict JSON only with these keys: "
        "normalized_core, target_language_message, literal_back_translation, reply_suggestion, expert_notes, sensitivity_flags, confidence. "
        "expert_notes must be a list of objects with expert and note. "
        f"Speaker: {speaker_name or 'Speaker A'}. Listener: {listener_name or 'Speaker B'}. "
        f"Source language: {source_lang or 'auto'}. Target language: {target_lang or 'en-US'}. "
        f"Relationship tone: {relationship_tone or 'warm and clear'}. Active expert lenses: {selected}. "
        f"Provider/faculty plan: {provider_plan or 'council-first'}. "
        f"Conversation profile: {conversation_profile or '(none supplied)'}. "
        f"Glossary and fixed terms: {glossary_terms or '(none supplied)'}."
    )


def _fallback_bridge_draft(raw_message: str, source_lang: str, target_lang: str, error: str) -> dict[str, Any]:
    same_lang = (source_lang or "").strip().lower() == (target_lang or "").strip().lower()
    return {
        "normalized_core": raw_message if same_lang else "",
        "target_language_message": raw_message if same_lang else "",
        "literal_back_translation": raw_message if same_lang else "",
        "reply_suggestion": "",
        "expert_notes": [
            {
                "expert": "Continuum",
                "note": "Model drafting was unavailable; raw content is preserved for the next pass.",
            },
            {
                "expert": "System",
                "note": _brief(error, 180),
            },
        ],
        "sensitivity_flags": ["draft_unavailable"],
        "confidence": 0.0,
    }


def _normalize_bridge_notes(notes: Any, lenses: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if isinstance(notes, list):
        for item in notes:
            if isinstance(item, dict):
                expert = str(item.get("expert") or item.get("lens") or "Expert").strip()
                note = str(item.get("note") or item.get("finding") or "").strip()
            else:
                expert = "Expert"
                note = str(item).strip()
            if note:
                out.append({"expert": expert or "Expert", "note": note})
    if out:
        return out[:8]
    defaults = {
        "Cultural / idiom": "Check idioms, implied politeness, and culture-specific emotional force.",
        "Warmth / relationship": "Keep the delivery affectionate, respectful, and easy to receive.",
        "Domain / business / crypto": "Flag technical, business, money, and crypto wording for exactness.",
        "Literal back-translation": "Preserve a simple back-translation path for verification.",
        "Brevity / delivery": "Prefer a message a real person would send, not a formal report.",
        "Safety / consent": "Do not publish externally or share personal content without review.",
    }
    for lens in lenses or BRIDGE_EXPERT_LENSES:
        out.append({"expert": lens, "note": defaults.get(lens, "Selected expert lens is pending review.")})
    return out


def _bridge_packet(
    raw_message: str,
    draft: dict[str, Any],
    speaker_name: str,
    listener_name: str,
    source_lang: str,
    target_lang: str,
    relationship_tone: str,
    lenses: list[str],
    conversation_profile: str,
    glossary_terms: str,
    provider_plan: str,
    guard: dict[str, Any],
    include_link_event: bool,
    model_id: str,
) -> dict[str, Any]:
    created_ms = int(time.time() * 1000)
    raw_sha = sha256((raw_message or "").encode("utf-8")).hexdigest()
    seed = json.dumps(
        {
            "raw_sha256": raw_sha,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "created_ms": created_ms,
        },
        sort_keys=True,
    )
    event_id = "bridge_" + sha256(seed.encode("utf-8")).hexdigest()[:16]
    translation_faculty = build_translation_faculty_packet(
        raw_message,
        source_lang,
        target_lang,
        conversation_profile,
        glossary_terms,
        provider_plan,
    )
    confidence = draft.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    packet = {
        "packet_kind": "cross_cultural_conversation_packet",
        "event_id": event_id,
        "created_ms": created_ms,
        "model_id": model_id,
        "participants": {
            "speaker": speaker_name or "Speaker A",
            "listener": listener_name or "Speaker B",
            "source_lang": source_lang or "auto",
            "target_lang": target_lang or "en-US",
            "relationship_tone": relationship_tone or "warm and clear",
        },
        "provenance": {
            "raw_content": raw_message,
            "raw_sha256": raw_sha,
            "input_lang": source_lang or "auto",
            "raw_preserved": True,
        },
        "agent_arbitration": {
            "selected_lenses": lenses or BRIDGE_EXPERT_LENSES,
            "expert_notes": _normalize_bridge_notes(draft.get("expert_notes"), lenses),
            "sensitivity_flags": list(draft.get("sensitivity_flags") or []),
            "consensus_accuracy_score": max(0.0, min(1.0, confidence)),
        },
        "execution_plane": {
            "normalized_core": str(draft.get("normalized_core") or ""),
            "target_language_message": str(draft.get("target_language_message") or ""),
            "literal_back_translation": str(draft.get("literal_back_translation") or ""),
            "reply_suggestion": str(draft.get("reply_suggestion") or ""),
            "consensus_lang": target_lang or "en-US",
        },
        "translation_faculty": translation_faculty,
        "adapter_targets": {
            "primary_channel": "whatsapp_business_cloud_api",
            "link_slots": list(BRIDGE_DEFAULT_SLOTS),
            "room_session": {
                "mode": "continuum_link_sse_room",
                "endpoint": "/room/create",
                "message_sent": False,
                "operator_approval_required": True,
            },
            "whatsapp": {
                "mode": "draft_adapter",
                "auth_required": True,
                "send_performed": False,
                "supports_text": True,
                "supports_voice_note_media": True,
                "operator_approval_required": True,
            },
            "nostr": {
                "mode": "unsigned_draft",
                "kind": BRIDGE_NOSTR_KIND,
                "relay_published": False,
                "signing_required": True,
            },
            "wallet": {
                "mode": "external_wallet_intent_only",
                "preferred_provider": "TokenPocket",
                "walletconnect_candidate": True,
                "nostr_wallet_connect_candidate": False,
                "custody": False,
                "seed_storage": False,
                "funds_moved": False,
            },
        },
        "continuum_guard": guard,
        "continuum_link": {
            "include_event_draft": bool(include_link_event),
            "posted_to_link_server": False,
            "event": None,
        },
    }
    packet["adapter_targets"]["nostr"]["unsigned_event"] = _bridge_unsigned_nostr_event(packet)
    if include_link_event:
        packet["continuum_link"]["event"] = _bridge_link_event(packet)
    return packet


def cultural_bridge_draft(
    raw_message: str,
    speaker_name: str,
    listener_name: str,
    source_lang: str,
    target_lang: str,
    relationship_tone: str,
    expert_lenses: list[str],
    conversation_profile: str,
    glossary_terms: str,
    provider_plan: str,
    include_link_event: bool,
    model_id: str,
    state,
):
    state = state or _new_session()
    raw_message = (raw_message or "").strip()
    if not raw_message:
        t1, t2 = render_trace_plots(state["trace"])
        return state, "Paste a message first.", "", "", {}, t1, t2, *load_trace_table(state)

    lenses = list(expert_lenses or BRIDGE_EXPERT_LENSES)
    source_lang = (source_lang or "auto").strip()
    target_lang = (target_lang or "en-US").strip()
    model_id = model_id or DEFAULT_MODEL
    state["trace"].observe(
        "cultural_bridge",
        f"{source_lang} -> {target_lang}",
        {
            "raw_sha256": sha256(raw_message.encode("utf-8")).hexdigest(),
            "char_count": len(raw_message),
            "token_est": _estimate_tokens(raw_message),
            "lens_count": len(lenses),
            "include_link_event": bool(include_link_event),
        },
    )

    prompt = _bridge_prompt(
        speaker_name,
        listener_name,
        source_lang,
        target_lang,
        relationship_tone,
        lenses,
        conversation_profile,
        glossary_terms,
        provider_plan,
    )
    payload = {
        "raw_message": raw_message,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "speaker": speaker_name or "Speaker A",
        "listener": listener_name or "Speaker B",
        "relationship_tone": relationship_tone or "warm and clear",
        "expert_lenses": lenses,
        "conversation_profile": conversation_profile or "",
        "glossary_terms": glossary_terms or "",
        "provider_plan": provider_plan or "council-first",
    }
    draft_source = "model"
    try:
        t0 = time.time()
        reply = run_model(model_id, [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        draft = _extract_json_object(reply)
        if draft is None:
            draft_source = "model_non_json"
            draft = _fallback_bridge_draft(raw_message, source_lang, target_lang, "The model returned prose instead of JSON.")
            draft["model_raw"] = _brief(reply, 900)
        latency_ms = int((time.time() - t0) * 1000)
        state["trace"].observe(
            "model_response",
            f"{model_id} cultural bridge",
            {
                "model": model_id,
                "latency_ms": latency_ms,
                "char_count": len(reply or ""),
                "token_est": _estimate_tokens(reply or ""),
                "draft_source": draft_source,
            },
        )
    except Exception as exc:
        draft_source = "fallback"
        draft = _fallback_bridge_draft(raw_message, source_lang, target_lang, f"{type(exc).__name__}: {exc}")
        state["trace"].observe("error", f"bridge draft failed: {type(exc).__name__}", {"error": str(exc), "ok": False})

    guard = _bridge_local_guard(raw_message, draft, target_lang)

    packet = _bridge_packet(
        raw_message,
        draft,
        speaker_name,
        listener_name,
        source_lang,
        target_lang,
        relationship_tone,
        lenses,
        conversation_profile,
        glossary_terms,
        provider_plan,
        guard,
        include_link_event,
        model_id,
    )
    try:
        state["store"].remember(
            text=(
                f"Cross-cultural bridge {packet['participants']['speaker']} -> "
                f"{packet['participants']['listener']} / {source_lang}->{target_lang} / "
                f"{packet['event_id']}"
            ),
            kind="cultural_bridge",
            tags=["cultural_bridge", "translation", "continuum_native", "whatsapp_ready"],
            metadata={
                "event_id": packet["event_id"],
                "raw_sha256": packet["provenance"]["raw_sha256"],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "guard_decision": guard.get("decision"),
                "link_event_included": bool(include_link_event),
                "primary_channel": packet["adapter_targets"]["primary_channel"],
            },
        )
    except Exception:
        pass
    state["trace"].observe(
        "continuum_guard",
        f"guard={guard.get('decision', 'unknown')}",
        {
            "event_id": packet["event_id"],
            "guard_decision": guard.get("decision"),
            "link_event_included": bool(include_link_event),
            "primary_channel": packet["adapter_targets"]["primary_channel"],
            "relay_published": False,
            "external_send_performed": False,
            "payment_moved": False,
        },
    )

    guard_label = guard.get("decision") or "unknown"
    link_label = "included" if include_link_event else "not included"
    status = (
        f"**Bridge packet ready.** Raw text preserved under `provenance.raw_content`; "
        f"memory stores only the event id and raw hash. Draft source: `{draft_source}`. "
        f"Continuum guard: `{guard_label}`. Link event draft: `{link_label}`. "
        "WhatsApp send, relay publish, wallet payment, and external execution: `false`."
    )
    t1, t2 = render_trace_plots(state["trace"])
    mem_rows, mem_status = load_trace_table(state)
    return (
        state,
        status,
        packet["execution_plane"]["target_language_message"],
        packet["execution_plane"]["reply_suggestion"],
        packet,
        t1,
        t2,
        mem_rows,
        mem_status,
    )


def reset():
    s = _new_session()
    t1, t2 = render_trace_plots(s["trace"])
    return [], s, "", t1, t2, ""


def runtime_settings_state():
    link_port = os.environ.get("CONTINUUM_LINK_PORT", "7871")
    link_url = os.environ.get("CONTINUUM_LINK_URL", f"http://127.0.0.1:{link_port}")
    mcp_port = os.environ.get("CONTINUUM_MCP_PORT", "7872")
    mcp_url = os.environ.get("CONTINUUM_MCP_URL", f"http://127.0.0.1:{mcp_port}")
    token_file = Path(
        os.environ.get("CONTINUUM_LINK_TOKEN_FILE")
        or (Path(__file__).parent / "cli_brain_channel" / "continuum_link_token.txt")
    )
    try:
        agents = _live_forum_agents()
    except Exception:
        agents = []
    return {
        "schema": "champion-continuum/runtime-settings/v1",
        "mode": "cli_brain" if CLI_BRAIN else "resident_model_or_provider",
        "space": {
            "is_space": bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST")),
            "zero_gpu": USE_ZERO_GPU,
            "hf_token_present": bool(HF_TOKEN),
        },
        "local_deck": {
            "root": str(Path(__file__).parent),
            "port": os.environ.get("GRADIO_SERVER_PORT", "7860"),
            "server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
            "login_button_visible": not CLI_BRAIN,
        },
        "cli_brain": {
            "enabled": CLI_BRAIN,
            "brain_channel": str(BRAIN_DIR),
            "shared_store": str(SHARED_STORE_ROOT),
            "live_agents": agents,
            "connect_code_available": CLI_BRAIN,
        },
        "link_service": {
            "url": link_url,
            "token_file": str(token_file),
            "token_file_exists": token_file.exists(),
            "settings_endpoint": f"{link_url}/settings",
            "sse_all": f"{link_url}/sse?slot=*",
            "peer_link_capacity": 5,
        },
        "mcp_service": {
            "url": mcp_url,
            "sse": f"{mcp_url}/mcp/sse",
            "streamable_http": f"{mcp_url}/mcp",
            "purpose": "Expose Continuum tools to tool-less agents through MCP.",
        },
        "peer_links": _peer_link_state_for_ui(),
        "providers": provider_registry_state(),
        "faculties": translation_faculty_state(),
        "privacy_defaults": {
            "link_store_raw": os.environ.get("CONTINUUM_LINK_STORE_RAW", "0"),
            "link_store_identifiers": os.environ.get("CONTINUUM_LINK_STORE_IDENTIFIERS", "0"),
            "raw_default": "hash_and_length_only",
        },
        "operator_rules": [
            "Local start_deck.bat uses CLI-brain mode.",
            "Use model mode only when you intentionally want a resident HF model or provider picker.",
            "Settings are read-only here; sends, wallets, relays, and pins remain explicit approval paths.",
        ],
    }


def runtime_settings_markdown() -> str:
    settings = runtime_settings_state()
    provider = settings["providers"]["huggingface_inference_providers"]
    peer_links = settings["peer_links"]
    link_service = settings["link_service"]
    mcp_service = settings["mcp_service"]
    cli_brain = settings["cli_brain"]
    privacy = settings["privacy_defaults"]
    space = settings["space"]
    auth_line = (
        "HF login is available in the model controls; provider calls use the signed-in user token first."
        if not CLI_BRAIN
        else "Local CLI-brain mode is active; the selected CLI agent is the brain."
    )
    agents = cli_brain.get("live_agents") or []
    agent_names = ", ".join(str(item.get("agent") or item.get("name") or item) for item in agents) if agents else "none"
    return "\n".join(
        [
            "### Continuum Settings",
            "",
            "**Hugging Face auth**",
            f"- {auth_line}",
            f"- Space secret fallback: {'configured' if space.get('hf_token_present') else 'not configured'}",
            f"- Provider default: `{provider.get('default_provider')}:{provider.get('default_model')}`",
            "",
            "**MCP/SSE service links**",
            f"- Saved service slots: {peer_links.get('count', 0)} / {peer_links.get('max_links', MAX_PEER_LINKS_UI)}",
            "- Use the five boxes below to connect friend, group, work, or local Continuum MCP/SSE services.",
            "",
            "**Local service endpoints**",
            f"- Continuum link service: `{link_service.get('url')}`",
            f"- Link SSE stream: `{link_service.get('sse_all')}`",
            f"- Continuum MCP/SSE: `{mcp_service.get('sse')}`",
            f"- Continuum streamable HTTP: `{mcp_service.get('streamable_http')}`",
            "",
            "**Runtime mode**",
            f"- Mode: `{settings.get('mode')}`",
            f"- CLI brain enabled: `{cli_brain.get('enabled')}`",
            f"- Live CLI agents: {agent_names}",
            "",
            "**Privacy defaults**",
            f"- Raw link payload storage: `{privacy.get('link_store_raw')}`",
            f"- Identifier storage: `{privacy.get('link_store_identifiers')}`",
            f"- Raw content default: `{privacy.get('raw_default')}`",
        ]
    )


def settings_refresh_values():
    return (runtime_settings_markdown(), *_peer_link_values())


def _tool_rows(hits: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for h in hits:
        args = h.get("args") or []
        name = str(h.get("name") or "")
        desc = str(h.get("description") or "")
        hay = f"{name} {desc}".lower()
        if any(word in hay for word in ("whatsapp", "message", "send")):
            category = "WhatsApp"
        elif any(word in hay for word in ("wallet", "bitcoin", "btc", "zap", "payment", "sats")):
            category = "Wallet"
        elif any(word in hay for word in ("memory", "bag", "store", "recall")):
            category = "Memory"
        elif any(word in hay for word in ("workflow", "agent", "council", "slot")):
            category = "Agent"
        elif _is_read_only_tool_name(name):
            category = "Read"
        else:
            category = "Action"
        relay = f"[[tool: {h.get('server')}.{h.get('name')} | " + ", ".join(f"{a}=" for a in args) + "]]"
        rows.append([
            category,
            f"{h.get('server')}.{h.get('name')}",
            ", ".join(args) if args else "(none)",
            desc.strip(),
            relay,
        ])
    return rows


def _all_tool_hits(store) -> list[dict]:
    # The active cache holds exactly the currently-connected server's full tool set.
    return store.active_tools()


def _exploration_candidates(hits: list[dict], limit: int = 18) -> list[str]:
    priority = (
        "get_help", "help", "onboarding", "status", "capabilities", "about",
        "readme", "docs", "catalog", "list", "tree", "health", "search",
        "memory", "bag", "file", "workflow",
    )
    scored: list[tuple[int, str]] = []
    for hit in hits:
        server = hit.get("server") or "external"
        name = hit.get("name") or ""
        if not _is_read_only_tool_name(name):
            continue
        desc = (hit.get("description") or "").lower()
        hay = f"{name.lower()} {desc}"
        score = 0
        for i, word in enumerate(priority):
            if word in hay:
                score += max(1, len(priority) - i)
        if score:
            args = ", ".join(f"{arg}=" for arg in (hit.get("args") or []))
            scored.append((score, f"{server}.{name}" + (f" | {args}" if args else "")))
    scored.sort(reverse=True)
    return [item for _, item in scored[:limit]]


_READ_ONLY_HINTS = (
    "get_", "list_", "show_", "read_", "search_", "info", "status", "help", "test",
    "catalog", "tree", "about", "capabilities", "onboarding", "health",
)
_MUTATING_HINTS = (
    "start", "spawn", "create", "write", "edit", "delete", "remove", "restore",
    "import", "export", "register", "download", "upload", "plug", "unplug",
    "mutate", "persist", "bind", "activate", "run", "launch", "send", "toggle",
)


def _is_read_only_tool_name(name: str) -> bool:
    clean = (name or "").split(".", 1)[-1].lower()
    if clean.startswith(_MUTATING_HINTS):
        return False
    return clean.startswith(_READ_ONLY_HINTS) or any(h in clean for h in _READ_ONLY_HINTS)


def _sanitize_relay_templates(text: str) -> str:
    def repl(match: re.Match) -> str:
        kind = match.group(1)
        body = match.group(2)
        parts = [part.strip() for part in body.split("|")]
        if len(parts) <= 1:
            return match.group(0)
        kept = [parts[0]]
        for arg in parts[1:]:
            if "=" not in arg:
                kept.append(arg)
                continue
            key, value = arg.split("=", 1)
            if value.strip():
                kept.append(arg)
        return "[[" + kind + ": " + " | ".join(kept) + "]]"
    return re.sub(r"\[\[\s*(tool|tools|continuum)\s*:\s*(.*?)\]\]", repl, text or "", flags=re.IGNORECASE | re.DOTALL)


_TOOL_CALL_RE = re.compile(r"\[\[\s*tool\s*:\s*([^|\]\s]+)", re.IGNORECASE)
_OPERATOR_ACTION_RE = re.compile(
    r"\b(start|spawn|create|write|edit|delete|remove|restore|import|export|register|download|"
    r"upload|plug|unplug|mutate|persist|bind|activate|run|launch|send|toggle|publish|deploy|push)\b",
    re.IGNORECASE,
)


def _unsafe_tool_calls(text: str) -> list[str]:
    unsafe: list[str] = []
    for raw in _TOOL_CALL_RE.findall(text or ""):
        name = raw.strip()
        if name and not _is_read_only_tool_name(name):
            unsafe.append(name)
    return sorted(set(unsafe))


def _operator_authorized_action(message: str) -> bool:
    return bool(_OPERATOR_ACTION_RE.search(message or ""))


_SUSPICIOUS_DOC_RE = re.compile(
    r"(docs\.example\.com|example\.com|placeholder|lorem ipsum|mock documentation|dummy docs)",
    re.IGNORECASE,
)


def _evidence_warning(text: str) -> str:
    if _SUSPICIOUS_DOC_RE.search(text or ""):
        return (
            "\n\n[continuum evidence warning]\n"
            "The previous result contains placeholder/example-looking documentation. "
            "Use raw help, bag_catalog, bag_tree, file_tree, or metadata surfaces for factual claims.\n"
        )
    return ""


def _strip_uncorroborated_prose(prose: str, accumulated: list[str]) -> str:
    if not prose:
        return ""
    if any(_SUSPICIOUS_DOC_RE.search(part or "") for part in accumulated):
        if re.search(r"(documentation|docs|url|based on|according to|found)", prose, re.IGNORECASE):
            return (
                "I found placeholder-looking documentation and am using raw help/catalog/file surfaces "
                "for corroboration."
            )
    return prose


def _exploration_prompt(store) -> str:
    hits = _all_tool_hits(store)
    summary = store.indexed_tool_summary()
    servers = ", ".join(summary.get("servers") or ["external"])
    candidates = _exploration_candidates(hits)
    candidate_text = "\n".join(f"- {tool}" for tool in candidates) or "- Fallback reads: [[tools: search | help]] and [[tools: search | status]]."
    return (
        "Universal MCP exploration mode is now active.\n\n"
        f"You are connected to MCP server(s): {servers}. The indexed surface has {summary.get('count', len(hits))} tools.\n\n"
        "Your job is to proactively learn this facility from the available evidence. "
        "READ-ONLY FIRST. Use help, status, capabilities, about, catalog, list, tree, search, and read tools "
        "during orientation. Reserve server starts, process spawns, file writes, registrations, imports, exports, "
        "and facility mutations for explicit operator action requests. Build a compact operating map: identity, help surface, memory/workspace surfaces, read-only status, "
        "available actions, cached payload follow-ups, and the next useful safe reads. Use exact relay commands "
        "that YOU emit. Follow _cached ids immediately. If one candidate fails, search for the right spelling and continue. "
        "When concrete reads remain, emit the next read. When documentation results look synthetic "
        "(for example docs.example.com), label them low-trust and corroborate through get_help/catalog/tree/raw metadata.\n\n"
        "High-value candidate tools discovered at connection time:\n"
        f"{candidate_text}\n\n"
        "Start now with the smallest safe orientation commands."
    )


def connect_tools(mcp_url: str, state):
    state = state or _new_session()
    url = (mcp_url or "").strip()
    if not url:
        return "Paste an MCP SSE URL above first.", [], state
    root = Path(state["store"].store.root)
    (root / "mcp.json").write_text(json.dumps({"mcpServers": {"external": {"url": url}}}))
    try:
        idx = state["store"].index_mcp_tools()
    except Exception as exc:
        state["mcp_connected"] = False
        return f"Could not reach that MCP server: {type(exc).__name__}: {exc}", [], state
    state["mcp_url"] = url
    state["mcp_connected"] = bool(idx.get("discovered"))
    state["relay_system"] = get_system_prompt("relay", continuum=state["store"])
    if not idx.get("discovered"):
        return "Connected, but the server exposed no tools.", [], state
    rows = _tool_rows(_all_tool_hits(state["store"]))
    servers = ", ".join(state["store"].indexed_tool_summary()["servers"])
    return f"Connected: **{len(rows)} tools** on {servers}. Filter below, then click a row to load its relay command.", rows, state


def connect_and_explore(mcp_url: str, model_id: str, history: list, state):
    # Connect = handshake + list tools, rendered IMMEDIATELY. We do NOT fire a blocking
    # relay/exploration turn here: that used to call chat(), which blocks up to 1200s
    # waiting for a daemon — so the button "did nothing" and the tools never showed.
    status, rows, state = connect_tools(mcp_url, state)
    t1, t2 = render_trace_plots((state or _new_session())["trace"])
    history = list(history or [])
    if rows:
        history.append({
            "role": "assistant",
            "content": (f"Connected — {len(rows)} tools indexed and live on the Tool Surface. "
                        "Browse or search them on the right; ask me to use one and I'll call it."),
        })
    return status, status, rows, state, history, "", t1, t2


def browse_tools(query: str, state):
    if not state or "store" not in state:
        return "Click Connect first.", []
    store = state["store"]
    summary = store.indexed_tool_summary()
    if not summary["count"]:
        return "Tool index is empty.", []
    q = (query or "").strip()
    if q:
        rows = _tool_rows(store.search_tools(q, limit=80))
        return f"**{len(rows)}** tools matching '{q}'. Click a row to load its relay command.", rows
    rows = _tool_rows(_all_tool_hits(store))
    return f"**All {len(rows)} tools** on {', '.join(summary['servers'])}. Filter above, click a row to load its relay command.", rows


def on_select_tool(table, evt: gr.SelectData):
    try:
        if getattr(evt, "row_value", None):
            return str(evt.row_value[4])
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            return str(table.iloc[row, 4])
        except Exception:
            return str(table[row][4])
    except Exception:
        return ""


def _paste_to_chat(relay: str, current: str) -> str:
    relay = (relay or "").strip()
    if not relay:
        return current or ""
    current = current or ""
    return current + ("\n" if current.strip() else "") + relay


# ============================================================
#  THEME / AESTHETICS  --  GEMINI'S CANVAS.
# ============================================================
TITLE = "Champion Continuum"
TAGLINE = "He doesn't always remember. When he does, it's continuity."
INTRO_MD = f"""
Talk normally. The council handles memory, tools, translation, cultural tact, receipts, and rough-edge
smoothing behind the scenes. Use the five SSE slots for peer Continuums; use the chat for everything human.
"""
THEME = gr.themes.Base(
    primary_hue="amber",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Playfair Display"), "serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0d0c0b",
    block_background_fill="#141211",
    block_border_width="1px",
    block_title_text_color="*primary_200",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    input_background_fill="#1b1917",
)

CUSTOM_CSS = """
:root {
    --cc-bg: #0d0c0b;
    --cc-panel: #151311;
    --cc-panel-2: #1b1917;
    --cc-line: #2a2624;
    --cc-text: #f3f4f6;
    --cc-muted: #a8a29e;
    --cc-accent: #d97706;
}

html, body, gradio-app {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background: var(--cc-bg) !important;
}

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;          /* NO outer scrollbar */
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
    padding: clamp(8px, 1.1vw, 16px) !important;
    background: var(--cc-bg) !important;
}

/* hard clamp the Plotly inner chart so it can never drive a growth loop */
#timeline-plot, #sankey-plot { overflow: hidden !important; }
#timeline-plot .js-plotly-plot, #sankey-plot .js-plotly-plot,
#timeline-plot .plot-container, #sankey-plot .plot-container {
    max-height: 430px !important;
}

#hero-container {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-areas: "mark title" "mark tagline";
    align-items: center;
    column-gap: 0.8rem;
    padding: 0.25rem 0 0.65rem;
    border-bottom: 1px solid var(--cc-line);
    margin-bottom: 0.7rem;
}

#hero-stache {
    grid-area: mark;
    font-size: 2rem;
    color: var(--cc-accent);
    line-height: 1;
}

#hero-title {
    grid-area: title;
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.45rem, 2.2vw, 2.35rem);
    font-weight: 700;
    margin: 0;
    letter-spacing: 0;
    color: var(--cc-text);
}

#hero-tagline {
    grid-area: tagline;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 0.96rem;
    color: var(--cc-muted);
    margin: 0.12rem 0 0;
}

#intro-panel {
    padding: 0 !important;
}

#intro-panel p {
    margin: 0 !important;
    color: var(--cc-muted);
    font-size: 0.92rem;
    line-height: 1.35;
}

#control-row {
    align-items: end !important;
    gap: 0.65rem !important;
    margin-bottom: 0.75rem !important;
}

#control-row .form {
    border-radius: 6px !important;
}

#main-row {
    flex: 1 1 auto !important;     /* fill leftover space exactly -- no calc guess, no overflow */
    min-height: 0 !important;
    overflow: hidden !important;
    gap: 0.75rem !important;
    align-items: stretch !important;
}

#chat-col,
#support-col {
    min-height: 0 !important;
}

#chat-col {
    display: flex !important;
    flex-direction: column !important;
}

#support-col {
    max-width: 820px;
    min-width: 480px;
}

#continuum-chat {
    flex: 1 1 auto !important;     /* fill the column -> flush with the support panel */
    min-height: 0 !important;
    max-height: calc(100vh - 232px) !important;  /* hard ceiling so it never grows the page */
    overflow-y: auto !important;                  /* messages scroll INSIDE the chat box */
    border: 1px solid var(--cc-line) !important;
    border-radius: 8px !important;
    background: #11100f !important;
}

/* make Gradio's own message wrap scroll within the box, never push it taller */
#continuum-chat .bubble-wrap,
#continuum-chat [class*="bubble-wrap"],
#continuum-chat [class*="message-wrap"] {
    max-height: 100% !important;
    overflow-y: auto !important;
}

#continuum-chat .message.bot,
#continuum-chat .message.user {
    border-radius: 7px !important;
}

#continuum-chat .message.bot {
    background-color: var(--cc-panel-2) !important;
}

#composer-row {
    align-items: stretch !important;
    gap: 0.55rem !important;
    margin-top: 0.6rem !important;
}

#continuum-input textarea {
    min-height: 54px !important;
    max-height: 148px !important;
    resize: vertical !important;
}

#send-btn {
    min-width: 108px !important;
    font-weight: 700;
}

#clear-btn {
    color: #78716c;
    font-size: 0.82rem;
    text-decoration: underline;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: fit-content !important;
    margin-top: 0.3rem;
}

#clear-btn:hover { color: var(--cc-accent); }

#support-tabs {
    height: 100%;
    min-height: 0;
    overflow-y: auto !important;
}

#support-tabs > div {
    min-height: 0 !important;
}

#tool-table {
    height: 480px !important;
    min-height: 0 !important;
    overflow: auto !important;
}
#peer-link-shell {
    margin: -0.15rem 0 0.75rem !important;
    padding: 0.62rem 0.7rem !important;
    border: 1px solid var(--cc-line);
    border-radius: 8px;
    background: #11100f;
}
#peer-link-shell .wrap {
    gap: 0.5rem !important;
}
#peer-link-row {
    gap: 0.5rem !important;
}
#peer-link-row .form {
    border-radius: 6px !important;
}
#peer-link-status {
    color: var(--cc-muted);
    font-size: 0.86rem;
    line-height: 1.25;
}
/* let the side-tab tables breathe: readable rows, wrapped metadata, no oyster-tin squeeze */
#support-tabs table { font-size: 0.86rem !important; }
#support-tabs td { white-space: normal !important; vertical-align: top !important; padding: 6px 8px !important; }
#mem-table { height: 460px !important; overflow: auto !important; }

#timeline-plot,
#sankey-plot {
    height: 430px !important;
    min-height: 0 !important;
}

#relay-row {
    align-items: end !important;
    gap: 0.5rem !important;
}

@media (max-width: 1180px) {
    html, body, gradio-app, .gradio-container {
        height: auto !important;
        max-height: none !important;
        overflow: auto !important;   /* stacked mobile layout scrolls normally */
    }
    #main-row {
        flex: none !important;
        height: auto;
        min-height: 0;
        flex-direction: column !important;
    }
    #support-col {
        max-width: none;
        width: 100% !important;
        min-width: 0;
    }
    #continuum-chat {
        height: 68vh !important;
        min-height: 430px !important;
    }
    #tool-table,
    #timeline-plot,
    #sankey-plot {
        height: 420px !important;
    }
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 8px !important;
    }
    #hero-container {
        grid-template-columns: 1fr;
        grid-template-areas: "title" "tagline";
        padding-top: 0.2rem;
    }
    #hero-stache {
        display: none;
    }
    #control-row {
        flex-direction: column !important;
        align-items: stretch !important;
    }
    #continuum-chat {
        height: 64vh !important;
        min-height: 360px !important;
    }
    #composer-row {
        flex-direction: column !important;
    }
    #send-btn {
        width: 100% !important;
    }
}

/* Footer pinned as its own flex row at the bottom -- keeps the 🔌 Use via API,
   🧡 Built with Gradio, and ⚙️ Settings buttons visible without a second scrollbar. */
footer {
    flex: 0 0 auto !important;
    display: flex !important;
    overflow: visible !important;
    padding: 4px 0 2px !important;
    opacity: .85;
}
"""

HERO_HTML = f"""
<div id="hero-container">
    <span id="hero-stache">⸙</span>
    <h1 id="hero-title">{TITLE}</h1>
    <p id="hero-tagline">{TAGLINE}</p>
</div>
"""

# Local deck override: allow ordinary page scroll for the shell, but keep the
# transcript itself bounded so long forum runs scroll inside the chat panel.
_CLI_SCROLL_CSS = """
html, body, gradio-app, .gradio-container {
    height: auto !important;
    max-height: none !important;
    overflow-y: auto !important;
}
#main-row, #chat-col, #support-col {
    min-height: 0 !important;
    max-height: none !important;
}
#chat-col {
    min-height: min(720px, calc(100vh - 220px)) !important;
}
#continuum-chat {
    height: min(650px, calc(100vh - 270px)) !important;
    min-height: 380px !important;
    max-height: min(650px, calc(100vh - 270px)) !important;
    overflow: hidden !important;
}
#continuum-chat > div,
#continuum-chat [class*="chatbot"],
#continuum-chat [class*="bubble-wrap"],
#continuum-chat [class*="message-wrap"] {
    max-height: 100% !important;
    overflow-y: auto !important;
}
"""

MODEL_LABEL = "Select an intellect"
CHATBOT_LABEL = "The Agent's Record"
INPUT_PLACEHOLDER = "Offer a thought or ask for a recollection..."
SEND_LABEL = "Commit"
CLEAR_LABEL = "A clean slate"
QUOTA_MSG = "The collective GPU quota is spent. Return after the daily reset."
MODEL_ERROR_PREFIX = "Intellect failure: "
# ============================================================

with gr.Blocks(title=TITLE) as demo:
    session = gr.State()
    if HERO_HTML.strip():
        gr.HTML(HERO_HTML)

    if CLI_BRAIN:
        conn_banner = gr.HTML(_connection_status_html())
        with gr.Accordion("Connect an agent (paste this into a CLI / agent / IDE chat)", open=False):
            gr.Code(value=CONNECT_CODE, label="Connect code", interactive=False)
        conn_timer = gr.Timer(2.0)
        conn_timer.tick(_connection_status_html, None, conn_banner)

    # Pull & Run — sits by the connect paste (associative), shown on local deck AND Space.
    with gr.Accordion("Pull & Run — get this for your own use", open=False):
        gr.Markdown(PULL_RUN_MD)

    # Seals — collapsed easter egg, shown on both the local deck and the Space.
    with gr.Accordion("Champion Council — Seals (authorized)", open=False):
        gr.Code(value=SEALS, interactive=False)

    with gr.Row(elem_id="control-row"):
        with gr.Column(scale=2, elem_id="intro-panel"):
            gr.Markdown(INTRO_MD)
        with gr.Column(scale=3):
            with gr.Row():
                if CLI_BRAIN:
                    # No model picker: the brain is whatever agent reads the channel.
                    model_dd = gr.State("cli-relay")
                    gr.Markdown("**Brain:** CLI relay — answered live by the agent at the channel.")
                else:
                    model_dd = gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL, label=MODEL_LABEL, elem_id="model-picker", scale=2)
                    if RUNNING_ON_HF_SPACE:
                        hf_login = gr.LoginButton(value="Sign in with Hugging Face", size="sm", scale=1)
                    else:
                        gr.Markdown("HF login appears on the Hugging Face Space. Local provider auth uses an HF token env var.")
            mcp_url = gr.Textbox(
                label="One-off MCP/SSE service URL",
                placeholder="https://.../mcp/sse",
                scale=3,
            )
            connect_btn = gr.Button("Connect One", variant="secondary")
            mcp_connect_status = gr.Markdown(
                "Use the five boxes below for the normal multi-service setup. This one-off field is optional.",
                elem_id="mcp-connect-status",
            )
            with gr.Row():
                graph_size = gr.Slider(250, 900, value=430, step=10, label="Graph size", elem_id="graph-size", scale=1)
                graph_opacity = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="Graph opacity", elem_id="graph-opacity", scale=1)

    peer_link_defaults = _peer_link_values()
    with gr.Group(elem_id="peer-link-shell"):
        gr.Markdown("**Continuum MCP/SSE service links**")
        with gr.Row(elem_id="peer-link-row"):
            peer_link_1 = gr.Textbox(label="Service 1", value=peer_link_defaults[0], placeholder="http://127.0.0.1:7872/mcp/sse")
            peer_link_2 = gr.Textbox(label="Service 2", value=peer_link_defaults[1], placeholder="https://friend.example/mcp/sse")
            peer_link_3 = gr.Textbox(label="Service 3", value=peer_link_defaults[2], placeholder="https://group.example/mcp/sse")
            peer_link_4 = gr.Textbox(label="Service 4", value=peer_link_defaults[3], placeholder="https://work.example/mcp/sse")
            peer_link_5 = gr.Textbox(label="Service 5", value=peer_link_defaults[4], placeholder="https://overflow.example/mcp/sse")
        with gr.Row():
            peer_save = gr.Button("Save & Connect Services", variant="secondary", scale=1)
            peer_link_status = gr.Markdown(_peer_link_status_text(), elem_id="peer-link-status")

    with gr.Row(elem_id="main-row"):
        # Left: Chat
        with gr.Column(scale=3, elem_id="chat-col"):
            chatbot = gr.Chatbot(height=650, label=CHATBOT_LABEL, elem_id="continuum-chat")
            with gr.Row(elem_id="composer-row"):
                box = gr.Textbox(placeholder=INPUT_PLACEHOLDER, scale=8, show_label=False, elem_id="continuum-input")
                send = gr.Button(SEND_LABEL, variant="primary", scale=1, elem_id="send-btn")
            clear = gr.Button(CLEAR_LABEL, elem_id="clear-btn")

        # Right: Interactive Trace
        with gr.Column(scale=3, elem_id="support-col"):
            with gr.Tabs(elem_id="support-tabs"):
                with gr.TabItem("Stage"):
                    stage = gr.HTML(
                        value="<div style='opacity:.6;padding:8px'>The agent composes "
                              "its own surface here, message by message.</div>",
                        elem_id="agent-stage",
                    )
                with gr.TabItem("Live Trace"):
                    timeline_plot = gr.Plot(label="Temporal Timeline", elem_id="timeline-plot")
                    sankey_plot = gr.Plot(label="Causal Flow", elem_id="sankey-plot")
                with gr.TabItem("Tool Surface"):
                    tool_status = gr.Markdown("Paste MCP/SSE service URLs into the five boxes, then click **Save & Connect Services**.")
                    tool_search = gr.Textbox(placeholder="Search tools...", show_label=False)
                    tool_table = gr.Dataframe(
                        headers=["Category", "Tool", "Args", "Description", "Relay Command"],
                        datatype=["str", "str", "str", "str", "str"],
                        column_count=(5, "fixed"),
                        wrap=True,
                        interactive=False,
                        elem_id="tool-table",
                    )
                    with gr.Row(elem_id="relay-row"):
                        relay_box = gr.Textbox(label="Relay Command (click row to load)", scale=4, interactive=True)
                        paste_btn = gr.Button("Paste to chat ▸", scale=1)
                with gr.TabItem("Settings"):
                    settings_summary = gr.Markdown(value=runtime_settings_markdown(), elem_id="settings-summary")
                    with gr.Group(elem_id="settings-link-config"):
                        gr.Markdown("**MCP/SSE service slots**")
                        with gr.Row():
                            settings_link_1 = gr.Textbox(label="Service 1", value=peer_link_defaults[0], placeholder="http://127.0.0.1:7872/mcp/sse")
                            settings_link_2 = gr.Textbox(label="Service 2", value=peer_link_defaults[1], placeholder="https://friend.example/mcp/sse")
                            settings_link_3 = gr.Textbox(label="Service 3", value=peer_link_defaults[2], placeholder="https://group.example/mcp/sse")
                            settings_link_4 = gr.Textbox(label="Service 4", value=peer_link_defaults[3], placeholder="https://work.example/mcp/sse")
                            settings_link_5 = gr.Textbox(label="Service 5", value=peer_link_defaults[4], placeholder="https://overflow.example/mcp/sse")
                        with gr.Row():
                            settings_save = gr.Button("Save & Connect Services", variant="secondary")
                            settings_refresh = gr.Button("Refresh", size="sm")
                with gr.TabItem("Memory"):
                    mem_refresh = gr.Button("Load / refresh memory", size="sm")
                    mem_status = gr.Markdown(
                        "The agent's memory graph fills as it acts. Click any node for its full informational wealth."
                    )
                    mem_table = gr.Dataframe(
                        headers=["#", "kind", "summary", "cid"],
                        datatype=["number", "str", "str", "str"],
                        column_count=(4, "fixed"),
                        wrap=True,
                        interactive=False,
                        elem_id="mem-table",
                    )
                    mem_detail = gr.JSON(label="Node informational wealth")

    # Wiring
    send.click(chat, [box, chatbot, model_dd, mcp_url, session], [chatbot, session, box, timeline_plot, sankey_plot, stage]).then(
        load_trace_table, [session], [mem_table, mem_status])
    box.submit(chat, [box, chatbot, model_dd, mcp_url, session], [chatbot, session, box, timeline_plot, sankey_plot, stage]).then(
        load_trace_table, [session], [mem_table, mem_status])
    clear.click(reset, None, [chatbot, session, box, timeline_plot, sankey_plot, stage])
    connect_btn.click(
        connect_and_explore,
        [mcp_url, model_dd, chatbot, session],
        [mcp_connect_status, tool_status, tool_table, session, chatbot, box, timeline_plot, sankey_plot],
    )
    tool_search.change(browse_tools, [tool_search, session], [tool_status, tool_table])
    tool_table.select(on_select_tool, [tool_table], [relay_box])
    paste_btn.click(_paste_to_chat, [relay_box, box], [box])
    settings_refresh.click(
        settings_refresh_values,
        None,
        [settings_summary, settings_link_1, settings_link_2, settings_link_3, settings_link_4, settings_link_5],
    )
    peer_save.click(
        save_peer_links,
        [peer_link_1, peer_link_2, peer_link_3, peer_link_4, peer_link_5, session],
        [peer_link_status, session, settings_summary, tool_status, tool_table],
    ).then(
        _peer_link_values,
        None,
        [settings_link_1, settings_link_2, settings_link_3, settings_link_4, settings_link_5],
    )
    settings_save.click(
        save_peer_links,
        [settings_link_1, settings_link_2, settings_link_3, settings_link_4, settings_link_5, session],
        [peer_link_status, session, settings_summary, tool_status, tool_table],
    ).then(
        _peer_link_values,
        None,
        [peer_link_1, peer_link_2, peer_link_3, peer_link_4, peer_link_5],
    )
    mem_refresh.click(load_trace_table, [session], [mem_table, mem_status])
    mem_table.select(inspect_node, [session], [mem_detail])
    graph_size.change(set_graph_height, [graph_size, session], [timeline_plot, sankey_plot])
    graph_opacity.change(
        None, graph_opacity, None,
        js="(v) => { document.querySelectorAll('#timeline-plot, #sankey-plot').forEach(e => e.style.opacity = v); }",
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS + (_CLI_SCROLL_CSS if CLI_BRAIN else ""))
