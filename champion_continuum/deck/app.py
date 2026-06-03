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
import html
import json
import os
import re
import socket
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

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
from continuum_daemon_registry import load_daemon_registry
from continuum_music_forge import music_forge_state
from continuum_provider_registry import hf_provider_model_id, parse_provider_model_id, provider_catalog_state, provider_registry_state, run_hf_provider_chat
from continuum_translation_faculty import build_translation_faculty_packet, translation_faculty_state

try:
    from dreamer_oracle_gate import OracleGate
except Exception:  # Space still boots if the optional gate package is unavailable.
    OracleGate = None

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)  # needed for gated models (Gemma, Llama) + quota

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
GATED_MODEL_IDS = {
    "google/gemma-2-2b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
}
INTENT_MODE_CHOICES = [
    "Auto",
    "Plain Conversation",
    "Translation Bridge",
    "Music Forge",
    "Resource Audit",
    "Expressive Wallpaper",
]
TOOL_REQUIRED_INTENT_MODES = {"Music Forge", "Resource Audit", "Expressive Wallpaper"}

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


def _uses_request_hf_token(model_id: str, hf_token: str | None = None) -> bool:
    return bool(hf_token and model_id in GATED_MODEL_IDS)


def _model_hf_token(model_id: str, hf_token: str | None = None) -> str | None:
    if _uses_request_hf_token(model_id, hf_token):
        return hf_token
    return HF_TOKEN


def _get_tokenizer(model_id: str, hf_token: str | None = None):
    if _uses_request_hf_token(model_id, hf_token):
        return AutoTokenizer.from_pretrained(model_id, token=_model_hf_token(model_id, hf_token))
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, token=_model_hf_token(model_id, hf_token))
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
def _gpu_generate(model_id: str, prompt: str, hf_token: str | None = None) -> str:
    global _ACTIVE_MODEL_ID
    # Device-aware: CUDA when present (HF ZeroGPU lights it up inside this function),
    # CPU for the local cockpit. Same code path, both runtimes.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    request_scoped_token = _uses_request_hf_token(model_id, hf_token)
    token = _model_hf_token(model_id, hf_token)
    if request_scoped_token:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, token=token
        ).to(device)
    elif model_id not in _MODEL_CACHE:
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
            model_id, dtype=dtype, token=token
        ).to(device)
        _ACTIVE_MODEL_ID = model_id
    else:
        model = _MODEL_CACHE[model_id]
    try:
        tok = _get_tokenizer(model_id, hf_token=hf_token)
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
    finally:
        if request_scoped_token:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
BRAIN_DIR.mkdir(exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(BRAIN_DIR / "gradio_tmp"))
os.environ.setdefault("CONTINUUM_EVENT_LOG", str(BRAIN_DIR / "continuum_link_events.jsonl"))
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
(HF provider shortcut: .\\start_hf_daemon.bat)
(HF provider scout pack: .\\start_all_hf_provider_daemons.bat)
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
Hugging Face Inference Providers can join as a daemon too:
```
.\start_hf_daemon.bat
.\start_all_hf_provider_daemons.bat
```
It uses `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` or your local `hf auth login`
token. Override the default model with `FORUM_HF_PROVIDER` and `FORUM_HF_MODEL`.
The pack launcher starts verified chat-provider routes supported by the local
Hugging Face client. Credit-gated routes stay out of the default pack. Use
`.\start_all_hf_provider_daemons.bat --include-unverified` when you want to
attempt additional catalog chat-capable providers, accepting that some live
model/provider pairs may fail.
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
        card = d.get("capability_card") if isinstance(d.get("capability_card"), dict) else {}
        kind = str(card.get("kind") or d.get("kind") or "mind")
        risk = str(card.get("risk_level") or d.get("risk_level") or "unknown")
        caps = [str(x) for x in (card.get("capabilities") or d.get("capabilities") or [])][:3]
        can_speak = bool(d.get("can_speak", False))
        can_watch = bool(d.get("can_watch", False))
        busy = bool(d.get("busy", False))
        try:
            ago = max(0, int(now - float(d.get("ts", 0))))
        except (TypeError, ValueError):
            ago = 9999
        fresh = ago <= 25
        state = "fresh" if fresh else "stale"
        if busy:
            state = "busy"
        role_class = re.sub(r"[^a-z0-9_-]+", "-", kind.lower()).strip("-") or "mind"
        if "provider" in role_class or "inference" in role_class:
            role_class = "provider"
        elif "engineer" in role_class or "codex" in agent.lower():
            role_class = "engineer"
        elif "audit" in role_class or "gemini" in agent.lower():
            role_class = "auditor"
        elif "relationship" in role_class or "claude" in agent.lower():
            role_class = "voice"
        note = "live" if fresh else f"{ago}s idle"
        if busy:
            note = "at work"
        speak_watch = ("speak" if can_speak else "silent") + "/" + ("watch" if can_watch else "blind")
        cap_text = " · ".join(caps) if caps else "forum"
        chips.append(
            "<span class='cc-mind-chip "
            + html.escape(role_class)
            + " "
            + html.escape(state)
            + "' title='"
            + html.escape(f"{agent} | {kind} | risk {risk} | {speak_watch}")
            + "'>"
            + "<span class='cc-mind-pulse'></span>"
            + "<span class='cc-mind-main'>"
            + "<span class='cc-mind-name'>"
            + html.escape(agent)
            + "</span>"
            + "<span class='cc-mind-kind'>"
            + html.escape(kind.replace("_", " "))
            + "</span>"
            + "</span>"
            + "<span class='cc-mind-meta'>"
            + html.escape(note)
            + " · "
            + html.escape(cap_text)
            + "</span>"
            + "</span>"
        )
    if not chips:
        return (
            "<div class='cc-forum-roster empty'>"
            "<span class='cc-roster-title'>FORUM</span>"
            "<span class='cc-roster-empty'>Forum empty — paste the connect code into a CLI / agent / IDE to join.</span>"
            "</div>"
        )
    return (
        "<div class='cc-forum-roster'>"
        "<div class='cc-roster-head'>"
        "<span class='cc-roster-title'>FORUM</span>"
        "<span class='cc-roster-subtitle'>minds present</span>"
        "</div>"
        "<div class='cc-roster-chips'>"
        + "".join(chips)
        + "</div></div>"
    )


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
    tok = _get_tokenizer(model_id, hf_token=hf_token)
    prompt = tok.apply_chat_template(_prep_messages(messages), add_generation_prompt=True, tokenize=False)
    return _gpu_generate(model_id, prompt, hf_token=hf_token)


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
    intent_mode = str(state.get("intent_mode") or "Auto")
    if intent_mode in TOOL_REQUIRED_INTENT_MODES:
        return True
    if decision.get("mode") == "BUILD" and decision.get("should_execute"):
        return True
    has_tool_surface = bool(state.get("mcp_connected") or state.get("mcp_url"))
    if _operator_authorized_action(message) and (has_tool_surface or "[[" in message):
        return True
    if has_tool_surface and _TOOL_INTENT_RE.search(message or ""):
        return True
    return False


def _intent_mode_context(intent_mode: str, state: dict | None = None) -> str:
    mode = (intent_mode or "Auto").strip() or "Auto"
    if mode not in INTENT_MODE_CHOICES:
        mode = "Auto"
    tool_summary = {}
    try:
        tool_summary = state["store"].indexed_tool_summary() if state and state.get("store") else {}
    except Exception:
        tool_summary = {}
    tool_count = int(tool_summary.get("count") or 0)
    lines = [
        "OPERATOR INTENT MODE OVERRIDE:",
        f"- Selected mode: {mode}",
        "- This is the operator's current routing preference for the normal chat, not a separate worksheet.",
    ]
    if mode == "Auto":
        lines.append("- Infer the goal from the message. Use tools only when the request needs live facilities or evidence.")
    elif mode == "Plain Conversation":
        lines.append("- Stay conversational. Do not use tools unless the operator explicitly asks for a facility, file, or live evidence.")
    elif mode == "Translation Bridge":
        lines.extend([
            "- Prioritize a sendable cross-language/cross-cultural reply in the main answer.",
            "- Preserve warmth, humor, and human intent. Include literal back-translation only when it improves trust.",
        ])
    elif mode == "Music Forge":
        lines.extend([
            "- Treat the message as a request to plan or produce original music/audio through Music Forge.",
            "- First search Music Forge tools. If native.* tools are returned, use native.continuum_music_forge_state, native.continuum_music_compose_packet, and native.continuum_music_backend_preset for in-process state/payload work.",
            "- Use schema/generation tools only when the specific tool result says the real backend is available. A finished tool-backed music turn returns saved audio file paths and a manifest path.",
        ])
    elif mode == "Resource Audit":
        lines.extend([
            "- Enumerate the live resource/tool surface from evidence. Do not invent facilities.",
            "- If no MCP tools are indexed, still audit the Continuum-native fallback surface through native.* tools before reporting a dead end.",
        ])
    elif mode == "Expressive Wallpaper":
        lines.extend([
            "- Treat the answer as a visual/expressive surface request.",
            "- Search continuum_expressive_wallpaper first when you need the contract, then use continuum_wallpaper_text for words, continuum_wallpaper_control for settings/audio/modal commands, or continuum_wallpaper_preset for named looks.",
            "- If native.* wallpaper tools are returned, call them directly. The native wallpaper bridge does not require an indexed MCP sidecar.",
            "- Wallpaper tool success means queued for the browser bridge, not visibly rendered. Say queued unless a browser receipt or Probe Wallpaper Bridge readout confirms the iframe applied it.",
        ])
    lines.append(f"- Indexed MCP tool count visible to this session: {tool_count}.")
    return "\n".join(lines)


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


def _active_tool_cache_state() -> dict[str, Any]:
    cache_path = SHARED_STORE_ROOT / "mcp_tools_active.json"
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {"count": 0, "url": "", "servers": [], "cache_path": str(cache_path), "exists": False}
    tools = payload.get("tools") if isinstance(payload, dict) else []
    if not isinstance(tools, list):
        tools = []
    return {
        "count": len(tools),
        "url": str(payload.get("url") or ""),
        "servers": sorted({str(item.get("server") or "") for item in tools if isinstance(item, dict) and item.get("server")}),
        "cache_path": str(cache_path),
        "exists": True,
        "ts": payload.get("ts"),
    }


def _local_mcp_base_url() -> str:
    mcp_port = os.environ.get("CONTINUUM_MCP_PORT", "7872")
    return os.environ.get("CONTINUUM_MCP_URL", f"http://127.0.0.1:{mcp_port}").rstrip("/")


def _local_mcp_sse_url() -> str:
    base = _local_mcp_base_url()
    if base.endswith("/mcp/sse"):
        return base
    if base.endswith("/mcp"):
        return f"{base}/sse"
    return f"{base}/mcp/sse"


def _maybe_index_local_native_tools() -> str:
    active = _active_tool_cache_state()
    if int(active.get("count") or 0) > 0:
        return ""
    mcp_port = _safe_int(os.environ.get("CONTINUUM_MCP_PORT", "7872"), 7872)
    probe = _tcp_service_state(_local_mcp_base_url(), mcp_port)
    if not probe.get("running"):
        return f"Local MCP self-index skipped: {probe.get('detail') or 'service not reachable'}."
    try:
        root = SHARED_STORE_ROOT
        root.mkdir(parents=True, exist_ok=True)
        config = {"mcpServers": {"continuum_local": {"url": _local_mcp_sse_url()}}}
        (root / "mcp.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        idx = Continuum(root).index_mcp_tools()
        count = int(idx.get("total") or idx.get("discovered") or 0)
        if count:
            return f"Local MCP self-indexed {count} native tools from `{_local_mcp_sse_url()}`."
        return f"Local MCP self-index reached `{_local_mcp_sse_url()}`, but no tools were listed."
    except Exception as exc:
        return f"Local MCP self-index failed: {type(exc).__name__}: {exc}"


def _peer_link_state_for_ui() -> dict[str, Any]:
    links = _load_peer_links_for_ui()
    active_tools = _active_tool_cache_state()
    return {
        "schema": "champion-continuum/peer-links/v1",
        "mode": "mcp_service_registry",
        "max_links": MAX_PEER_LINKS_UI,
        "count": len(links),
        "links": links,
        "indexed_tool_count": active_tools["count"],
        "indexed_tool_url": active_tools["url"],
        "external_connection_opened": bool(links),
        "auto_send_enabled": False,
        "note": "Five Continuum MCP/SSE service targets. Saved links become tool surfaces only after indexing succeeds.",
    }


def _peer_link_status_text() -> str:
    state = _peer_link_state_for_ui()
    if not state["count"]:
        return "No Continuum MCP/SSE services saved yet. Paste up to five service URLs, then Save & Connect."
    if int(state.get("indexed_tool_count") or 0) <= 0:
        return (
            f"Saved {state['count']} / {state['max_links']} Continuum MCP/SSE service links. "
            "No tools are indexed yet; start the target MCP service, confirm the URL, then click Save & Connect Services."
        )
    return (
        f"Saved {state['count']} / {state['max_links']} Continuum MCP/SSE service links. "
        f"Indexed {state['indexed_tool_count']} tools from `{state.get('indexed_tool_url') or 'active cache'}`. "
        "Tool-less agents can use them through [[tools: ...]] and [[tool: ...]]."
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


def _peer_links_from_raw_values(raw_values: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    existing_by_url = {
        str(item.get("url") or ""): item
        for item in _load_peer_links_for_ui()
        if str(item.get("url") or "")
    }
    links: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, raw in enumerate(raw_values[:MAX_PEER_LINKS_UI]):
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
    return links, errors


def _persist_peer_link_values(raw_values: list[str]) -> tuple[str, list[dict[str, Any]], list[str]]:
    links, errors = _peer_links_from_raw_values(raw_values)
    if errors:
        return "Could not save links: " + " ".join(errors), links, errors
    payload = {
        "schema": "champion-continuum/peer-links/v1",
        "updated_ms": int(time.time() * 1000),
        "max_links": MAX_PEER_LINKS_UI,
        "links": links,
    }
    PEER_LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PEER_LINKS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _peer_link_status_text(), links, []


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
    status_text, links, errors = _persist_peer_link_values(raw_values)
    if errors:
        try:
            state["trace"].observe("peer_links", "peer link save blocked", {"ok": False, "errors": errors})
        except Exception:
            pass
        return status_text, state, runtime_settings_markdown(), status_text, []
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


def latest_music_track() -> tuple[str, str | None, dict]:
    """Return the newest Music Forge artifact as a compact UI readout."""
    output_root = Path(os.environ.get("CONTINUUM_MUSIC_OUTPUTS", BRAIN_DIR / "music_outputs"))
    try:
        manifests = sorted(output_root.rglob("manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        manifests = []
    if not manifests:
        return "No Music Forge tracks found yet.", None, {"status": "empty", "output_root": str(output_root)}

    manifest_path = manifests[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Latest manifest could not be read: {type(exc).__name__}: {exc}", None, {
            "status": "error",
            "manifest_path": str(manifest_path),
        }

    saved_files = list(manifest.get("saved_files") or [])
    audio_path = ""
    audio_sha = ""
    audio_bytes = 0
    if saved_files:
        audio_path = str(saved_files[0].get("path") or "")
        audio_sha = str(saved_files[0].get("sha256") or "")
        try:
            audio_bytes = Path(audio_path).stat().st_size
            if not audio_sha:
                audio_sha = sha256(Path(audio_path).read_bytes()).hexdigest()
        except OSError:
            audio_bytes = int(saved_files[0].get("bytes") or 0)

    summary = "\n".join(
        [
            f"**Title:** {manifest.get('title') or '(untitled)'}",
            f"**Receipt:** `{manifest.get('receipt_id') or 'legacy manifest'}`",
            f"**Backend:** `{manifest.get('space_id') or '(unknown)'}` `{manifest.get('api_name') or ''}`",
            f"**Audio:** `{audio_path or 'none'}`",
            f"**Bytes:** `{audio_bytes}`",
            f"**SHA256:** `{audio_sha or 'unavailable'}`",
            f"**Manifest:** `{manifest_path}`",
        ]
    )
    info = {
        "status": "ok",
        "receipt_id": manifest.get("receipt_id") or "",
        "action_class": manifest.get("action_class") or "legacy",
        "approval_state": manifest.get("approval_state") or "",
        "title": manifest.get("title"),
        "space_id": manifest.get("space_id"),
        "api_name": manifest.get("api_name"),
        "audio_path": audio_path,
        "audio_bytes": audio_bytes,
        "audio_sha256": audio_sha,
        "manifest_path": str(manifest_path),
    }
    return summary, audio_path if audio_path and Path(audio_path).exists() else None, info


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
    intent_mode: str,
    mcp_url: str,
    state,
    oauth_token: gr.OAuthToken | None = None,
):
    state = state or _new_session()
    hf_oauth_token = _oauth_token_value(oauth_token)
    history_messages = _history_messages(history)
    message = (message or "").strip()
    if intent_mode not in INTENT_MODE_CHOICES:
        intent_mode = "Auto"
    state["intent_mode"] = intent_mode
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
            "intent_mode": intent_mode,
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
            plain_msgs.append({"role": "system", "content": _intent_mode_context(intent_mode, state)})
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
                    "intent_mode": intent_mode,
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
    current_msgs.append({"role": "system", "content": _intent_mode_context(intent_mode, state)})
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
            action_authorized = _operator_authorized_action(message) or intent_mode == "Music Forge"
            if unsafe and not action_authorized:
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


def _tcp_service_state(url: str, fallback_port: int | None = None) -> dict[str, Any]:
    parsed = urlparse(url or "")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or fallback_port
    if not port:
        return {"running": False, "host": host, "port": None, "detail": "no port configured"}
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return {"running": True, "host": host, "port": int(port), "detail": "tcp port accepts connections"}
    except OSError as exc:
        return {
            "running": False,
            "host": host,
            "port": int(port),
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _safe_int(value: str, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def runtime_settings_state():
    link_port = os.environ.get("CONTINUUM_LINK_PORT", "7871")
    link_url = os.environ.get("CONTINUUM_LINK_URL", f"http://127.0.0.1:{link_port}")
    mcp_port = os.environ.get("CONTINUUM_MCP_PORT", "7872")
    mcp_url = os.environ.get("CONTINUUM_MCP_URL", f"http://127.0.0.1:{mcp_port}")
    link_probe = _tcp_service_state(link_url, _safe_int(link_port, 7871))
    mcp_probe = _tcp_service_state(mcp_url, _safe_int(mcp_port, 7872))
    active_tools = _active_tool_cache_state()
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
            "running": link_probe["running"],
            "probe": link_probe,
            "token_file": str(token_file),
            "token_file_exists": token_file.exists(),
            "settings_endpoint": f"{link_url}/settings",
            "sse_all": f"{link_url}/sse?slot=*",
            "peer_link_capacity": 5,
        },
        "mcp_service": {
            "url": mcp_url,
            "running": mcp_probe["running"],
            "probe": mcp_probe,
            "sse": f"{mcp_url}/mcp/sse",
            "streamable_http": f"{mcp_url}/mcp",
            "purpose": "Expose Continuum tools to tool-less agents through MCP.",
            "indexed_tool_count": active_tools["count"],
            "indexed_tool_url": active_tools["url"],
        },
        "peer_links": _peer_link_state_for_ui(),
        "providers": provider_registry_state(),
        "music_forge": music_forge_state()
        | {
            "mcp_sidecar_running": mcp_probe["running"],
            "mcp_tools_indexed": active_tools["count"],
            "chat_auto_route_ready": bool(mcp_probe["running"] and active_tools["count"]),
        },
        "expressive_wallpaper": _wallpaper_runtime_state(),
        "utility_daemons": load_daemon_registry(BRAIN_DIR),
        "faculties": translation_faculty_state(),
        "privacy_defaults": {
            "link_store_raw": os.environ.get("CONTINUUM_LINK_STORE_RAW", "0"),
            "link_store_identifiers": os.environ.get("CONTINUUM_LINK_STORE_IDENTIFIERS", "0"),
            "raw_default": "hash_and_length_only",
        },
        "operator_rules": [
            "Local start_deck.bat uses CLI-brain mode.",
            "Use model mode only when you intentionally want a resident HF model or provider picker.",
            "The App Settings accordion describes the live posture; sends, wallets, relays, and pins remain explicit approval paths.",
        ],
    }


def runtime_settings_markdown() -> str:
    settings = runtime_settings_state()
    provider = settings["providers"]["huggingface_inference_providers"]
    peer_links = settings["peer_links"]
    link_service = settings["link_service"]
    mcp_service = settings["mcp_service"]
    music = settings["music_forge"]
    wallpaper = settings["expressive_wallpaper"]
    daemon_registry = settings["utility_daemons"]
    cli_brain = settings["cli_brain"]
    privacy = settings["privacy_defaults"]
    space = settings["space"]
    agents = cli_brain.get("live_agents") or []
    agent_names = ", ".join(str(item.get("agent") or item.get("name") or item) for item in agents) if agents else "none"
    auth_line = (
        "HF login is available in the model controls; provider calls use the signed-in user token first."
        if not CLI_BRAIN
        else "Local CLI-brain mode is active; the connected CLI agent is the brain."
    )
    daemons = daemon_registry.get("daemons") or []
    daemon_names = ", ".join(
        f"{item.get('agent')}:{item.get('kind')}"
        for item in daemons
        if not item.get("stale")
    ) or "none"
    daemon_counts = daemon_registry.get("counts") or {}
    link_running = "running" if link_service.get("running") else "not running in this process"
    mcp_running = "running" if mcp_service.get("running") else "not running in this process"
    hosted_sidecar_note = (
        "- Hosted Space note: `start_deck.bat` sidecars are local desktop processes; the Space runs `app.py` unless we add in-process services."
        if space.get("is_space") and not CLI_BRAIN
        else "- Local launcher note: `start_deck.bat` should start the Link service, MCP service, and deck together."
    )
    music_route_line = (
        "ready"
        if music.get("chat_auto_route_ready")
        else "not ready until MCP sidecar is running and tools are indexed"
    )
    return "\n".join(
        [
            "### Continuum Settings",
            "",
            "**Auth and models**",
            f"- {auth_line}",
            f"- Space secret fallback: {'configured' if space.get('hf_token_present') else 'not configured'}",
            f"- Provider default: `{provider.get('default_provider')}:{provider.get('default_model')}`",
            "",
            "**Utility daemons**",
            f"- Active sprites: {daemon_counts.get('active', 0)}",
            f"- Safe for autonomous assignment: {daemon_counts.get('safe_for_autonomous_assignment', 0)}",
            f"- Live roster: {daemon_names}",
            "",
            "**Continuum service links**",
            f"- Saved slots: {peer_links.get('count', 0)} / {peer_links.get('max_links', MAX_PEER_LINKS_UI)}",
            "- Edit the five MCP/SSE boxes above, then click **Save & Connect Services**.",
            "",
            "**Local endpoints**",
            f"- Link service: `{link_service.get('url')}` — {link_running}",
            f"- Link stream: `{link_service.get('sse_all')}`",
            f"- MCP/SSE: `{mcp_service.get('sse')}` — {mcp_running}",
            f"- MCP HTTP: `{mcp_service.get('streamable_http')}`",
            f"- Indexed MCP tools: `{mcp_service.get('indexed_tool_count', 0)}`",
            hosted_sidecar_note,
            "",
            "**Music Forge**",
            f"- Output folder: `{music.get('output_dir')}`",
            f"- Gradio client module: `{'installed' if music.get('gradio_client_available') else 'not installed'}`",
            f"- MCP tool lane: `{'running' if music.get('mcp_sidecar_running') else 'not running'}`",
            f"- Chat-to-audio auto-route: `{music_route_line}`",
            "- Installed tools can generate real audio files only after a working music Space/API is called and audio files are saved.",
            "",
            "**Expressive wallpaper**",
            f"- Active: `{'yes' if wallpaper.get('active') else 'no'}`",
            f"- Asset: `{wallpaper.get('asset') or 'none'}`",
            f"- Council speech rain: `{'ready' if wallpaper.get('speech_rain_ready') else 'unavailable'}`",
            "- Assistant/council replies can drive glyph rain, color, speed, direction, intensity, font size, modal state, and audio-reactive settings.",
            "",
            "**Runtime**",
            f"- Mode: `{settings.get('mode')}`",
            f"- CLI brain: `{cli_brain.get('enabled')}`",
            f"- Live agents: {agent_names}",
            "",
            "**Privacy defaults**",
            f"- Raw link storage: `{privacy.get('link_store_raw')}`",
            f"- Identifier storage: `{privacy.get('link_store_identifiers')}`",
            f"- Raw content default: `{privacy.get('raw_default')}`",
        ]
    )


def launch_hf_provider_daemon_pack(include_unverified: bool = False) -> str:
    """Start the local HF provider daemon pack from the deck UI."""
    if RUNNING_ON_HF_SPACE:
        return (
            "**HF provider daemon pack:** hosted Space launch is disabled.\n\n"
            "This button starts local desktop forum daemons that use the operator's "
            "HF token when they later answer turns. The hosted public Space catalogs "
            "the routes; it does not spawn token-backed background workers for visitors. "
            "Run `start_all_hf_provider_daemons.bat` locally instead."
        )
    try:
        from launch_hf_provider_pack import launch_pack

        result = launch_pack(include_unverified=bool(include_unverified))
    except Exception as exc:
        return f"**HF provider daemon pack:** failed before launch: `{type(exc).__name__}: {exc}`"

    status = str(result.get("status") or "unknown")
    launched = list(result.get("launched") or [])
    skipped = list(result.get("skipped") or [])
    lines = [
        f"**HF provider daemon pack:** `{status}`",
        "",
        f"- Routes considered: `{result.get('route_count', 0)}`",
        f"- Launched: `{len(launched)}`",
        f"- Skipped: `{len(skipped)}`",
        f"- Include unverified catalog attempts: `{bool(result.get('include_unverified'))}`",
        "- Credits are used only when a launched daemon later answers an assigned turn.",
    ]
    if status != "ok":
        lines.append(f"- Reason: `{result.get('reason', 'unknown')}`")
    if launched:
        lines.extend(["", "**Launched**"])
        for item in launched[:24]:
            lines.append(
                f"- `{item.get('agent')}` -> `{item.get('provider')}:{item.get('model')}` "
                f"(pid `{item.get('pid')}`, verified route `{item.get('verified_route')}`, "
                f"manual/credit-gated `{item.get('manual_or_credit_gated')}`)"
            )
    if skipped:
        lines.extend(["", "**Skipped**"])
        for item in skipped[:24]:
            lines.append(f"- `{item.get('agent')}`: {item.get('reason')}")
    return "\n".join(lines)


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


NATIVE_TOOLKIT_SPECS = [
    ("Status", "continuum_state", [], "Read Continuum link/event service state."),
    ("Status", "continuum_settings", [], "Read settings, facilities, providers, privacy posture, and peer/service links."),
    ("Status", "continuum_health", [], "Read local Continuum MCP service health."),
    ("Status", "continuum_providers", [], "Read model/provider routing posture."),
    ("Status", "continuum_provider_catalog", [], "Read HF Inference Provider catalog, routing policy, and free-credit posture."),
    ("Status", "continuum_faculties", [], "Read translation and cultural bridge faculty readiness."),
    ("Daemons", "continuum_utility_daemons", [], "Read live utility daemon capability cards and safety posture."),
    ("Daemons", "continuum_match_daemons", ["capability", "output", "include_stale"], "Find live utility daemons by capability or output type."),
    ("Daemons", "continuum_heartbeat", ["component", "status", "slot", "note", "capabilities_json"], "Publish a local component heartbeat event."),
    ("Music", "continuum_music_forge_state", [], "Read Music Forge readiness, output folder, and public music backends."),
    ("Music", "continuum_music_compose_packet", ["idea", "style", "lyrics", "language", "duration", "avoid"], "Build a song prompt and lyrics packet."),
    ("Music", "continuum_music_backend_preset", ["backend", "prompt", "lyrics", "duration", "seed"], "Build a ready-to-call public music backend payload."),
    ("Music", "continuum_music_hf_space_schema", ["space_id"], "Inspect a Hugging Face music Space API before generation."),
    ("Music", "continuum_music_generate_preset", ["backend", "prompt", "lyrics", "duration", "seed", "title"], "Generate music through a known public HF Space preset and save audio locally."),
    ("Music", "continuum_music_generate_hf_space", ["space_id", "prompt", "payload_json", "api_name", "title"], "Call a Hugging Face music Space and save returned audio locally."),
    ("Bridge", "continuum_translate_packet", ["raw_message", "target_language", "source_language", "relationship_tone"], "Build a local translation and cultural bridge packet without sending externally."),
    ("Links", "continuum_links", [], "Read registered peer/service link registry."),
    ("Links", "continuum_slots", [], "List event slots and current slot counts."),
    ("Links", "continuum_events", ["slot", "limit"], "Read recent Continuum events from a slot or all slots."),
    ("Links", "continuum_post_event", ["kind", "slot", "text", "payload_json", "source"], "Append a redacted local Continuum event for coordination."),
    ("Links", "continuum_create_room", ["room_label", "speaker_label", "listener_label", "source_lang", "target_lang", "relationship_tone"], "Create a local room session and return join paths."),
    ("Wallpaper", "continuum_expressive_wallpaper", [], "Read expressive wallpaper readiness and control contract."),
    ("Wallpaper", "continuum_wallpaper_text", ["text", "mode", "source", "slot"], "Queue text for the expressive wallpaper speech-rain bridge."),
    ("Wallpaper", "continuum_wallpaper_control", ["text", "settings_json", "command", "source", "slot"], "Queue wallpaper settings, audio-reactive, modal, and orchestration commands."),
    ("Wallpaper", "continuum_wallpaper_preset", ["preset", "text", "source", "slot"], "Apply a named expressive wallpaper preset."),
    ("Memory", "continuum_remember", ["text", "tags", "kind"], "Store a durable Continuum memory record."),
    ("Memory", "continuum_search", ["query", "limit"], "Search durable Continuum memory records."),
    ("Memory", "continuum_process_agent_text", ["text", "max_tool_calls"], "Execute relay commands emitted by a tool-less agent."),
    ("Intent", "continuum_whatsapp_send_intent", ["to", "text", "payload_json"], "Draft a WhatsApp send intent; does not send a message."),
    ("Intent", "continuum_wallet_intent", ["amount_sats", "memo", "asset", "payload_json"], "Draft a wallet/payment intent; does not move funds."),
]


def native_toolkit_surface() -> tuple[str, list[list[str]]]:
    self_index_note = _maybe_index_local_native_tools()
    active_by_name: dict[str, dict[str, Any]] = {}
    try:
        payload = json.loads((SHARED_STORE_ROOT / "mcp_tools_active.json").read_text(encoding="utf-8"))
        for item in payload.get("tools") or []:
            name = str(item.get("name") or "")
            if name:
                active_by_name[name.lower()] = item
    except Exception:
        active_by_name = {}

    rows: list[list[str]] = []
    indexed = 0
    for category, name, fallback_args, desc in NATIVE_TOOLKIT_SPECS:
        hit = active_by_name.get(name.lower())
        args = list(fallback_args)
        if hit:
            indexed += 1
            schema = hit.get("input_schema") or {}
            props = list((schema.get("properties") or {}).keys())
            if props:
                args = props
            tool_name = f"{hit.get('server')}.{name}"
            relay = f"[[tool: {tool_name} | " + ", ".join(f"{arg}=" for arg in args) + "]]"
            if not args:
                relay = f"[[tool: {tool_name} | ]]"
        else:
            tool_name = f"native.{name}"
            relay = f"[[tool: native.{name} | " + ", ".join(f"{arg}=" for arg in args) + "]]"
            if not args:
                relay = f"[[tool: native.{name} | ]]"
        rows.append([
            category,
            tool_name,
            ", ".join(args) if args else "(none)",
            desc,
            relay,
        ])
    status = (
        f"Continuum Native Toolkits: {len(rows)} cataloged; {indexed} indexed in the active MCP surface. "
        "Indexed rows call the MCP sidecar; `native.*` rows use the in-process fallback for tool-less agents."
    )
    if self_index_note:
        status += "\n\n" + self_index_note
    return status, rows


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
    "state", "settings", "schema", "preset", "packet", "providers", "faculties",
    "slots", "links", "events",
)
_MUTATING_HINTS = (
    "start", "spawn", "create", "write", "edit", "delete", "remove", "restore",
    "import", "export", "register", "download", "upload", "plug", "unplug",
    "mutate", "persist", "bind", "activate", "run", "launch", "send", "toggle",
    "post", "remember", "process", "heartbeat", "generate", "wallet", "whatsapp",
)


def _is_read_only_tool_name(name: str) -> bool:
    clean = (name or "").split(".", 1)[-1].lower()
    tokens = [part for part in re.split(r"[^a-z0-9]+", clean) if part]
    if clean.startswith(_MUTATING_HINTS) or any(token in _MUTATING_HINTS for token in tokens):
        return False
    if clean in {"continuum_expressive_wallpaper"}:
        return True
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
    r"upload|plug|unplug|mutate|persist|bind|activate|run|launch|send|toggle|publish|deploy|push|"
    r"adjust|update|change|set|queue|display|show|paint|rain|make|orchestrate|style|animate|control)\b",
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

#continuum-wallpaper {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    background: #0d0c0b;
    --cc-wallpaper-opacity: 0.52;
    --cc-wallpaper-brightness: 0.82;
    --cc-wallpaper-saturate: 1.2;
    --cc-wallpaper-contrast: 1.08;
}

#continuum-wallpaper[data-wallpaper-kind="web_wallpaper"] {
    --cc-wallpaper-opacity: 0.72;
    --cc-wallpaper-brightness: 0.92;
    --cc-wallpaper-saturate: 1.28;
    --cc-wallpaper-contrast: 1.1;
}

#continuum-wallpaper video,
#continuum-wallpaper img,
#continuum-wallpaper iframe {
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: var(--cc-wallpaper-opacity, 0.52);
    filter: saturate(var(--cc-wallpaper-saturate, 1.2)) contrast(var(--cc-wallpaper-contrast, 1.08)) brightness(var(--cc-wallpaper-brightness, 0.82));
    border: 0;
    display: block;
}

#continuum-wallpaper::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(13,12,11,0.16), rgba(13,12,11,0.72)),
        radial-gradient(circle at 74% 18%, rgba(217,119,6,0.10), transparent 34%),
        radial-gradient(circle at 18% 80%, rgba(34,197,94,0.08), transparent 32%);
}

#continuum-wallpaper.cc-wallpaper-blob {
    left: var(--cc-wallpaper-blob-x, calc(100vw - 476px));
    top: var(--cc-wallpaper-blob-y, 92px);
    right: auto;
    bottom: auto;
    width: var(--cc-wallpaper-blob-w, min(440px, calc(100vw - 36px)));
    height: var(--cc-wallpaper-blob-h, 310px);
    z-index: 39;
    pointer-events: none;
    border: 1px solid rgba(250, 204, 21, 0.58);
    border-radius: 28% 14% 22% 17% / 16% 26% 18% 30%;
    background:
        linear-gradient(135deg, rgba(250, 204, 21, 0.12), rgba(226, 232, 240, 0.08)),
        rgba(12, 12, 14, 0.70);
    box-shadow:
        0 24px 84px rgba(0, 0, 0, 0.48),
        0 0 34px rgba(250, 204, 21, 0.20),
        inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    resize: none;
    animation: cc-blob-breathe 10s ease-in-out infinite;
}

#continuum-wallpaper.cc-wallpaper-blob iframe,
#continuum-wallpaper.cc-wallpaper-blob video,
#continuum-wallpaper.cc-wallpaper-blob img {
    opacity: 0.92;
    filter: saturate(1.28) contrast(1.12) brightness(0.94);
    pointer-events: none;
}

#continuum-wallpaper.cc-wallpaper-blob::before {
    content: "Behold: Matrix Rain Blob";
    position: absolute;
    inset: 0.58rem 4.7rem auto 0.72rem;
    height: 2rem;
    z-index: 4;
    display: flex;
    align-items: center;
    padding: 0 0.72rem;
    border-radius: 999px 44px 999px 44px;
    border: 1px solid rgba(226, 232, 240, 0.36);
    background:
        linear-gradient(90deg, rgba(250, 204, 21, 0.24), rgba(226, 232, 240, 0.13)),
        rgba(13, 12, 11, 0.68);
    color: #fff7ed;
    font-size: 0.75rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: grab;
    pointer-events: none;
    box-shadow: 0 0 18px rgba(250, 204, 21, 0.18);
}

#continuum-wallpaper.cc-wallpaper-blob::after {
    background:
        linear-gradient(180deg, rgba(13,12,11,0.08), rgba(13,12,11,0.28)),
        radial-gradient(circle at 74% 18%, rgba(250,204,21,0.08), transparent 34%);
}

#continuum-wallpaper.cc-wallpaper-blob.cc-wallpaper-dragging::before {
    cursor: grabbing;
}

#continuum-wallpaper.cc-wallpaper-collapsed {
    width: min(360px, calc(100vw - 28px));
    height: 58px;
    overflow: hidden;
    border-radius: 999px 46px 999px 46px;
}

#continuum-wallpaper.cc-wallpaper-collapsed iframe,
#continuum-wallpaper.cc-wallpaper-collapsed video,
#continuum-wallpaper.cc-wallpaper-collapsed img {
    opacity: 0.12;
}

.cc-wallpaper-blob-button,
.cc-wallpaper-blob-handle,
.cc-wallpaper-blob-grip,
.cc-wallpaper-blob-edge {
    display: none;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-button,
#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-handle,
#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-grip,
#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-edge {
    display: block;
    position: absolute;
    z-index: 6;
    pointer-events: auto;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-handle {
    left: 0.72rem;
    right: 4.7rem;
    top: 0.58rem;
    height: 2rem;
    border: 0;
    border-radius: 999px 44px 999px 44px;
    background: transparent;
    cursor: grab;
    padding: 0;
}

#continuum-wallpaper.cc-wallpaper-blob.cc-wallpaper-dragging .cc-wallpaper-blob-handle {
    cursor: grabbing;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-button {
    top: 0.78rem;
    width: 24px;
    height: 24px;
    border: 1px solid rgba(226, 232, 240, 0.46);
    border-radius: 999px;
    background: rgba(15, 15, 17, 0.74);
    color: #fff7ed;
    cursor: pointer;
    font-size: 14px;
    line-height: 20px;
    padding: 0;
    box-shadow: 0 0 14px rgba(250, 204, 21, 0.18);
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-min {
    right: 3.12rem;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-close {
    right: 1.02rem;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-grip {
    right: 0.68rem;
    bottom: 0.6rem;
    width: 34px;
    height: 34px;
    border: 1px solid rgba(226, 232, 240, 0.42);
    border-radius: 999px 999px 12px 999px;
    background:
        linear-gradient(135deg, transparent 0 34%, rgba(226, 232, 240, 0.78) 35% 38%, transparent 39% 48%, rgba(250, 204, 21, 0.74) 49% 52%, transparent 53% 100%),
        rgba(15, 15, 17, 0.55);
    cursor: nwse-resize;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-edge[data-edge="right"] {
    top: 48px;
    right: 0;
    bottom: 42px;
    width: 22px;
    cursor: ew-resize;
}

#continuum-wallpaper.cc-wallpaper-blob .cc-wallpaper-blob-edge[data-edge="bottom"] {
    left: 42px;
    right: 58px;
    bottom: 0;
    height: 22px;
    cursor: ns-resize;
}

html, body, gradio-app {
    min-height: 100% !important;
    height: auto !important;
    max-height: none !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background: var(--cc-bg) !important;
}

.gradio-container {
    position: relative !important;
    z-index: 1 !important;
    max-width: none !important;
    width: 100% !important;
    min-height: 100vh !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
    padding: clamp(8px, 1.1vw, 16px) !important;
    background: transparent !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background:
        linear-gradient(180deg, rgba(13,12,11,0.88), rgba(13,12,11,0.96)),
        var(--cc-bg);
}

html.cc-wallpaper-active body,
html.cc-wallpaper-active gradio-app,
html.cc-wallpaper-active .gradio-container {
    background: transparent !important;
}

html.cc-wallpaper-active .gradio-container::before {
    background:
        linear-gradient(180deg, rgba(13,12,11,0.36), rgba(13,12,11,0.70)),
        radial-gradient(circle at 72% 8%, rgba(250,204,21,0.08), transparent 28%),
        radial-gradient(circle at 18% 78%, rgba(34,197,94,0.06), transparent 30%) !important;
}

.cc-forum-roster {
    display: flex;
    align-items: center;
    gap: 0.72rem;
    padding: 0.62rem 0.72rem;
    border: 1px solid rgba(217,119,6,0.26);
    border-radius: 8px;
    background:
        linear-gradient(90deg, rgba(217,119,6,0.12), rgba(22,163,74,0.07), rgba(59,130,246,0.07)),
        rgba(17,16,15,0.82);
    box-shadow: 0 10px 34px rgba(0,0,0,0.22);
    backdrop-filter: blur(8px);
    margin: 0 0 0.65rem;
}

.cc-forum-roster.empty {
    border-color: rgba(248,113,113,0.45);
    background: rgba(127,29,29,0.20);
    color: #fecaca;
}

.cc-roster-head {
    flex: 0 0 auto;
    display: grid;
    gap: 0.08rem;
    min-width: 108px;
}

.cc-roster-title {
    color: #fbbf24;
    font-weight: 800;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
}

.cc-roster-subtitle,
.cc-roster-empty {
    color: var(--cc-muted);
    font-size: 0.76rem;
}

.cc-roster-chips {
    display: flex;
    align-items: stretch;
    gap: 0.46rem;
    flex-wrap: wrap;
}

.cc-mind-chip {
    --chip-accent: #a8a29e;
    --chip-bg: rgba(168,162,158,0.10);
    display: inline-grid;
    grid-template-columns: 12px minmax(96px, auto) auto;
    align-items: center;
    gap: 0.44rem;
    min-height: 38px;
    padding: 0.34rem 0.58rem;
    border-radius: 7px;
    border: 1px solid color-mix(in srgb, var(--chip-accent) 62%, transparent);
    background: linear-gradient(180deg, color-mix(in srgb, var(--chip-bg) 72%, transparent), rgba(17,16,15,0.62));
    color: var(--cc-text);
}

.cc-mind-chip.provider { --chip-accent: #22d3ee; --chip-bg: rgba(34,211,238,0.14); }
.cc-mind-chip.engineer { --chip-accent: #f59e0b; --chip-bg: rgba(245,158,11,0.14); }
.cc-mind-chip.auditor { --chip-accent: #818cf8; --chip-bg: rgba(129,140,248,0.14); }
.cc-mind-chip.voice { --chip-accent: #fb7185; --chip-bg: rgba(251,113,133,0.14); }
.cc-mind-chip.stale { --chip-accent: #78716c; --chip-bg: rgba(120,113,108,0.08); opacity: 0.72; }
.cc-mind-chip.busy { --chip-accent: #a3e635; --chip-bg: rgba(163,230,53,0.14); }

.cc-mind-pulse {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--chip-accent);
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--chip-accent) 52%, transparent);
}

.cc-mind-chip.fresh .cc-mind-pulse,
.cc-mind-chip.busy .cc-mind-pulse {
    animation: cc-pulse 1.9s ease-out infinite;
}

.cc-mind-main {
    display: grid;
    gap: 0.02rem;
}

.cc-mind-name {
    color: var(--cc-text);
    font-weight: 800;
    font-size: 0.86rem;
    line-height: 1.05;
}

.cc-mind-kind {
    color: var(--chip-accent);
    font-size: 0.68rem;
    line-height: 1.1;
    text-transform: lowercase;
}

.cc-mind-meta {
    color: var(--cc-muted);
    font-size: 0.72rem;
    white-space: nowrap;
}

@keyframes cc-pulse {
    0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--chip-accent) 52%, transparent); }
    72% { box-shadow: 0 0 0 8px transparent; }
    100% { box-shadow: 0 0 0 0 transparent; }
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
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow: visible !important;
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

html.cc-wallpaper-active #continuum-chat {
    background: rgba(17, 16, 15, 0.74) !important;
    backdrop-filter: blur(8px) saturate(1.08);
    -webkit-backdrop-filter: blur(8px) saturate(1.08);
}

html.cc-wallpaper-active #support-tabs,
html.cc-wallpaper-active #peer-link-shell,
html.cc-wallpaper-active #app-settings-panel {
    background: rgba(17, 16, 15, 0.68) !important;
    backdrop-filter: blur(8px) saturate(1.08);
    -webkit-backdrop-filter: blur(8px) saturate(1.08);
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

#intent-mode {
    margin-top: 0.55rem !important;
    margin-bottom: 0.15rem !important;
}

#intent-mode .wrap,
#intent-mode [class*="radio"] {
    gap: 0.35rem !important;
}

#intent-mode label {
    border-radius: 7px !important;
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

#blob-controls-row {
    align-items: center !important;
    gap: 0.42rem !important;
    margin: 0.45rem 0 0.05rem !important;
}

#blob-controls-row button {
    min-width: 86px !important;
    border-radius: 7px !important;
    border-color: rgba(245, 158, 11, 0.32) !important;
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.11), rgba(148, 163, 184, 0.07)) !important;
    color: #f8fafc !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.035) !important;
}

#chat-col.cc-blob-enabled,
#support-col.cc-blob-enabled {
    --cc-blob-pressure: 0;
    --cc-blob-resonance: 0.12;
    --cc-blob-settle: 1;
    --cc-blob-pressure-glow: 0px;
    --cc-blob-resonance-glow: 4px;
    --cc-blob-contact-glow: 16px;
    --cc-blob-saturate: 1.04;
    --cc-blob-brightness: 1;
    --cc-blob-energy-opacity: 0.92;
    --cc-blob-energy-blur: 7px;
    --cc-blob-energy-saturate: 1.4;
    --cc-blob-current-duration: 13s;
    position: fixed !important;
    left: var(--cc-blob-x, 28px) !important;
    top: var(--cc-blob-y, 84px) !important;
    width: var(--cc-blob-w, min(720px, calc(100vw - 56px))) !important;
    height: var(--cc-blob-h, min(760px, calc(100vh - 112px))) !important;
    min-width: 280px !important;
    min-height: 128px !important;
    max-width: calc(100vw - 18px) !important;
    max-height: calc(100vh - 18px) !important;
    z-index: var(--cc-blob-z, 40) !important;
    resize: both !important;
    overflow: auto !important;
    isolation: isolate;
    padding: 3.05rem 0.88rem 0.9rem !important;
    border-radius: 33% 15% 25% 18% / 18% 28% 19% 31% !important;
    border: 1px solid rgba(250, 204, 21, 0.64) !important;
    background:
        linear-gradient(135deg, rgba(250, 204, 21, 0.18), rgba(226, 232, 240, 0.13) 42%, rgba(13, 12, 11, 0.62)),
        rgba(12, 12, 14, 0.58) !important;
    backdrop-filter: blur(18px) saturate(1.22);
    -webkit-backdrop-filter: blur(18px) saturate(1.22);
    box-shadow:
        0 28px 92px rgba(0, 0, 0, 0.48),
        0 0 calc(34px + var(--cc-blob-resonance-glow)) rgba(245, 158, 11, 0.16),
        0 0 var(--cc-blob-pressure-glow) rgba(226, 232, 240, 0.18),
        inset 0 0 0 1px rgba(255, 255, 255, 0.08),
        inset 0 0 54px rgba(226, 232, 240, 0.07) !important;
    filter: saturate(var(--cc-blob-saturate)) brightness(var(--cc-blob-brightness));
    animation: cc-blob-breathe 9s ease-in-out infinite;
    transition:
        border-radius 180ms ease,
        transform 180ms ease,
        box-shadow 180ms ease,
        filter 180ms ease,
        opacity 180ms ease;
}

#chat-col.cc-blob-enabled {
    overflow: hidden !important;
    padding-bottom: 6.3rem !important;
}

.cc-blob-energy,
.cc-blob-resize-grip,
.cc-blob-resize-edge,
.cc-blob-close {
    display: none;
}

#chat-col.cc-blob-enabled > :not(.cc-blob-energy),
#support-col.cc-blob-enabled > :not(.cc-blob-energy) {
    position: relative;
    z-index: 2;
}

#chat-col.cc-blob-enabled .cc-blob-energy,
#support-col.cc-blob-enabled .cc-blob-energy {
    display: block;
    position: absolute;
    inset: 0;
    z-index: 1;
    overflow: hidden;
    pointer-events: none;
    border-radius: inherit;
    opacity: var(--cc-blob-energy-opacity);
    mix-blend-mode: screen;
}

#chat-col.cc-blob-enabled .cc-blob-energy::before,
#support-col.cc-blob-enabled .cc-blob-energy::before {
    content: "";
    position: absolute;
    inset: -18%;
    background:
        repeating-conic-gradient(from 8deg, rgba(250, 204, 21, 0.00) 0 13deg, rgba(250, 204, 21, 0.13) 14deg 15deg, rgba(226, 232, 240, 0.00) 16deg 33deg),
        radial-gradient(circle at 30% 25%, rgba(226, 232, 240, 0.20), transparent 24%),
        radial-gradient(circle at 72% 70%, rgba(250, 204, 21, 0.17), transparent 28%);
    filter: blur(var(--cc-blob-energy-blur)) saturate(var(--cc-blob-energy-saturate));
    animation: cc-blob-current 13s linear infinite;
    animation-duration: var(--cc-blob-current-duration);
}

#chat-col.cc-blob-enabled .cc-blob-bubble,
#support-col.cc-blob-enabled .cc-blob-bubble {
    position: absolute;
    left: var(--x);
    bottom: -14%;
    width: var(--s);
    height: var(--s);
    border-radius: 50%;
    border: 1px solid rgba(226, 232, 240, 0.34);
    background:
        radial-gradient(circle at 34% 26%, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.18) 14%, transparent 32%),
        radial-gradient(circle at 62% 68%, rgba(250, 204, 21, 0.28), transparent 54%),
        rgba(226, 232, 240, 0.055);
    box-shadow:
        inset 0 0 12px rgba(255, 255, 255, 0.12),
        0 0 14px rgba(250, 204, 21, 0.18);
    transform: translate3d(0, 0, 0) scale(0.74);
    animation: cc-blob-bubble-rise var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
}

#chat-col.cc-blob-enabled .cc-blob-spark,
#support-col.cc-blob-enabled .cc-blob-spark {
    position: absolute;
    left: var(--x);
    top: var(--y);
    width: var(--w);
    height: 1px;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent, rgba(226, 232, 240, 0.95), rgba(250, 204, 21, 0.72), transparent);
    box-shadow: 0 0 10px rgba(226, 232, 240, 0.42), 0 0 18px rgba(250, 204, 21, 0.28);
    transform: rotate(var(--rot));
    animation: cc-blob-spark-flare var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
}

#chat-col.cc-blob-enabled .cc-blob-resize-grip,
#support-col.cc-blob-enabled .cc-blob-resize-grip {
    display: block;
    position: absolute;
    right: 0.7rem;
    bottom: 0.62rem;
    width: 34px;
    height: 34px;
    z-index: 6;
    border: 1px solid rgba(226, 232, 240, 0.42);
    border-radius: 999px 999px 12px 999px;
    background:
        linear-gradient(135deg, transparent 0 34%, rgba(226, 232, 240, 0.78) 35% 38%, transparent 39% 48%, rgba(250, 204, 21, 0.74) 49% 52%, transparent 53% 100%),
        rgba(15, 15, 17, 0.55);
    box-shadow: 0 0 18px rgba(250, 204, 21, 0.24), inset 0 0 12px rgba(226, 232, 240, 0.08);
    cursor: nwse-resize;
    pointer-events: auto;
    padding: 0;
}

#chat-col.cc-blob-enabled .cc-blob-resize-edge,
#support-col.cc-blob-enabled .cc-blob-resize-edge {
    display: block;
    position: absolute;
    z-index: 8;
    pointer-events: auto;
    background: transparent;
}

#chat-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="right"],
#support-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="right"] {
    top: 48px;
    right: 0;
    bottom: 42px;
    width: 24px;
    cursor: ew-resize;
}

#chat-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="bottom"],
#support-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="bottom"] {
    left: 42px;
    right: 58px;
    bottom: 0;
    height: 24px;
    cursor: ns-resize;
}

#chat-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="corner"],
#support-col.cc-blob-enabled .cc-blob-resize-edge[data-edge="corner"] {
    right: 0;
    bottom: 0;
    width: 62px;
    height: 62px;
    cursor: nwse-resize;
}

#chat-col.cc-blob-edge-left,
#support-col.cc-blob-edge-left,
#chat-col.cc-blob-edge-right,
#support-col.cc-blob-edge-right {
    border-radius: 10% 42% 42% 10% / 24% 14% 14% 24% !important;
    transform: scaleX(0.78) scaleY(1.055);
    box-shadow:
        0 20px 88px rgba(0, 0, 0, 0.48),
        0 0 48px rgba(250, 204, 21, 0.24),
        inset 0 0 0 1px rgba(255, 255, 255, 0.10),
        inset 0 0 64px rgba(226, 232, 240, 0.09) !important;
}

#chat-col.cc-blob-edge-right,
#support-col.cc-blob-edge-right {
    border-radius: 42% 10% 10% 42% / 14% 24% 24% 14% !important;
}

#chat-col.cc-blob-edge-top,
#support-col.cc-blob-edge-top,
#chat-col.cc-blob-edge-bottom,
#support-col.cc-blob-edge-bottom {
    border-radius: 18% 18% 38% 38% / 12% 12% 42% 42% !important;
    transform: scaleX(1.055) scaleY(0.82);
}

#chat-col.cc-blob-edge-bottom,
#support-col.cc-blob-edge-bottom {
    border-radius: 38% 38% 18% 18% / 42% 42% 12% 12% !important;
}

#chat-col.cc-blob-impact,
#support-col.cc-blob-impact {
    animation: cc-blob-impact 420ms ease-out 1, cc-blob-breathe 9s ease-in-out infinite;
}

#chat-col.cc-blob-collide,
#support-col.cc-blob-collide {
    --cc-blob-pressure: 0.82;
    --cc-blob-resonance: 0.88;
    border-color: rgba(226, 232, 240, 0.78) !important;
    box-shadow:
        0 22px 90px rgba(0, 0, 0, 0.50),
        0 0 52px rgba(226, 232, 240, 0.22),
        0 0 42px rgba(250, 204, 21, 0.24),
        inset 0 0 0 1px rgba(255, 255, 255, 0.12),
        inset 0 0 70px rgba(226, 232, 240, 0.11) !important;
}

#chat-col.cc-blob-probe-settled,
#support-col.cc-blob-probe-settled {
    --cc-blob-pressure: 0.02;
    --cc-blob-resonance: 0.16;
}

#chat-col.cc-blob-probe-contact::after,
#support-col.cc-blob-probe-contact::after {
    box-shadow:
        0 0 20px rgba(250, 204, 21, 0.24),
        0 0 var(--cc-blob-contact-glow) rgba(226, 232, 240, 0.22);
}

#chat-col.cc-blob-enabled .cc-blob-resize-grip:hover,
#support-col.cc-blob-enabled .cc-blob-resize-grip:hover {
    border-color: rgba(250, 204, 21, 0.84);
    box-shadow: 0 0 24px rgba(250, 204, 21, 0.34), inset 0 0 12px rgba(226, 232, 240, 0.14);
}

#chat-col.cc-blob-collapsed .cc-blob-resize-grip,
#support-col.cc-blob-collapsed .cc-blob-resize-grip,
#chat-col.cc-blob-collapsed .cc-blob-resize-edge,
#support-col.cc-blob-collapsed .cc-blob-resize-edge {
    display: none;
}

#chat-col.cc-blob-enabled .cc-blob-close,
#support-col.cc-blob-enabled .cc-blob-close {
    display: block;
    position: absolute;
    right: 1.08rem;
    top: 0.83rem;
    width: 24px;
    height: 24px;
    z-index: 7;
    border: 1px solid rgba(226, 232, 240, 0.46);
    border-radius: 999px;
    background: rgba(15, 15, 17, 0.72);
    color: #fff7ed;
    cursor: pointer;
    pointer-events: auto;
    font-size: 16px;
    line-height: 20px;
    padding: 0;
    box-shadow: 0 0 16px rgba(250, 204, 21, 0.20);
}

#chat-col.cc-blob-enabled .cc-blob-close:hover,
#support-col.cc-blob-enabled .cc-blob-close:hover {
    border-color: rgba(250, 204, 21, 0.86);
    color: #facc15;
}

#chat-col.cc-blob-enabled::before,
#support-col.cc-blob-enabled::before {
    content: "";
    position: absolute;
    inset: 0.55rem;
    pointer-events: none;
    z-index: 1;
    border-radius: inherit;
    background:
        linear-gradient(90deg, transparent 0 8%, rgba(250, 204, 21, 0.20) 8.5%, transparent 9.5% 46%, rgba(226, 232, 240, 0.18) 47%, transparent 48% 100%),
        repeating-radial-gradient(circle at 16% 20%, rgba(250, 204, 21, 0.20) 0 1px, transparent 2px 19px),
        repeating-linear-gradient(132deg, rgba(226, 232, 240, 0.14) 0 1px, transparent 1px 14px);
    opacity: 0.72;
    mix-blend-mode: screen;
}

#chat-col.cc-blob-enabled::after,
#support-col.cc-blob-enabled::after {
    content: attr(data-blob-title);
    position: absolute;
    inset: 0.62rem 0.82rem auto;
    height: 2rem;
    display: flex;
    align-items: center;
    padding: 0 0.72rem;
    border-radius: 999px 42px 999px 42px;
    border: 1px solid rgba(226, 232, 240, 0.34);
    background:
        linear-gradient(90deg, rgba(250, 204, 21, 0.19), rgba(226, 232, 240, 0.16), rgba(245, 158, 11, 0.11)),
        rgba(15, 15, 17, 0.50);
    color: #fff7ed;
    font-size: 0.77rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: grab;
    box-shadow: 0 0 18px rgba(250, 204, 21, 0.18);
    z-index: 4;
}

#chat-col.cc-blob-dragging,
#support-col.cc-blob-dragging {
    user-select: none;
}

#chat-col.cc-blob-dragging::after,
#support-col.cc-blob-dragging::after {
    cursor: grabbing;
}

#chat-col.cc-blob-enabled #continuum-chat {
    height: calc(100% - 148px) !important;
    min-height: 92px !important;
    max-height: none !important;
    margin-bottom: 5.4rem !important;
    overflow-y: auto !important;
    background: rgba(13, 12, 11, 0.55) !important;
    border-color: rgba(226, 232, 240, 0.24) !important;
}

#chat-col.cc-blob-enabled #intent-mode {
    flex: 0 0 auto !important;
    margin-top: 0.42rem !important;
}

#chat-col.cc-blob-enabled #blob-controls-row {
    flex: 0 0 auto !important;
}

#chat-col.cc-blob-enabled #composer-row {
    position: absolute !important;
    left: 0.88rem;
    right: 0.88rem;
    bottom: 0.76rem;
    z-index: 9;
    flex: none !important;
    padding: 0.48rem !important;
    border: 1px solid rgba(226, 232, 240, 0.18);
    border-radius: 10px;
    background: rgba(12, 12, 14, 0.76);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 -10px 28px rgba(0, 0, 0, 0.22), 0 0 18px rgba(250, 204, 21, 0.08);
}

#chat-col.cc-blob-enabled #continuum-input textarea {
    min-height: 48px !important;
    max-height: 112px !important;
}

#chat-col.cc-blob-enabled #clear-btn {
    position: absolute !important;
    left: 1.1rem;
    bottom: 5.05rem;
    z-index: 8;
}

#support-col.cc-blob-enabled #support-tabs {
    height: calc(100% - 0.3rem) !important;
    max-height: none !important;
    overflow-y: auto !important;
    background: rgba(13, 12, 11, 0.42) !important;
    border-radius: 10px !important;
}

#chat-col.cc-blob-collapsed,
#support-col.cc-blob-collapsed {
    width: min(360px, calc(100vw - 28px)) !important;
    height: 58px !important;
    min-height: 58px !important;
    resize: none !important;
    overflow: hidden !important;
    padding: 0 !important;
    border-radius: 999px 48px 999px 48px !important;
}

#chat-col.cc-blob-collapsed > :not(.cc-blob-energy),
#support-col.cc-blob-collapsed > :not(.cc-blob-energy) {
    opacity: 0 !important;
    pointer-events: none !important;
}

#chat-col.cc-blob-collapsed::after,
#support-col.cc-blob-collapsed::after {
    inset: 0.62rem 0.72rem;
    height: auto;
}

body.cc-blob-meld #chat-col.cc-blob-enabled,
body.cc-blob-meld #support-col.cc-blob-enabled {
    mix-blend-mode: normal;
}

body.cc-blob-meld #support-col.cc-blob-enabled {
    transform: translate(18px, 22px) rotate(0.3deg);
    opacity: 0.88;
    --cc-blob-z: 39;
}

body.cc-blob-meld #chat-col.cc-blob-enabled {
    --cc-blob-z: 41;
}

@keyframes cc-blob-breathe {
    0%, 100% {
        border-radius: 33% 15% 25% 18% / 18% 28% 19% 31%;
        filter: saturate(1.04);
    }
    48% {
        border-radius: 18% 29% 17% 34% / 31% 18% 29% 17%;
        filter: saturate(1.16);
    }
}

@keyframes cc-blob-current {
    0% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.06); }
    100% { transform: rotate(360deg) scale(1); }
}

@keyframes cc-blob-bubble-rise {
    0% {
        transform: translate3d(0, 18%, 0) scale(0.56);
        opacity: 0;
    }
    16% { opacity: 0.66; }
    58% {
        transform: translate3d(var(--drift), -62vh, 0) scale(1.08);
        opacity: 0.45;
    }
    100% {
        transform: translate3d(calc(var(--drift) * -0.45), -112vh, 0) scale(0.74);
        opacity: 0;
    }
}

@keyframes cc-blob-spark-flare {
    0%, 100% {
        opacity: 0;
        transform: rotate(var(--rot)) scaleX(0.25);
    }
    45% {
        opacity: 0.82;
        transform: rotate(var(--rot)) scaleX(1);
    }
    62% {
        opacity: 0.18;
        transform: rotate(calc(var(--rot) + 5deg)) scaleX(0.68);
    }
}

@keyframes cc-blob-impact {
    0% { filter: saturate(1.1) brightness(1); }
    40% { filter: saturate(1.35) brightness(1.16); }
    100% { filter: saturate(1.05) brightness(1); }
}

@media (prefers-reduced-motion: reduce) {
    #chat-col.cc-blob-enabled,
    #support-col.cc-blob-enabled,
    #chat-col.cc-blob-enabled .cc-blob-energy::before,
    #support-col.cc-blob-enabled .cc-blob-energy::before,
    #chat-col.cc-blob-enabled .cc-blob-bubble,
    #support-col.cc-blob-enabled .cc-blob-bubble,
    #chat-col.cc-blob-enabled .cc-blob-spark,
    #support-col.cc-blob-enabled .cc-blob-spark {
        animation: none !important;
    }
}

#support-tabs {
    height: auto;
    max-height: calc(100vh - 232px);
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


BACKGROUND_MEDIA_CANDIDATES = [
    "continuum_wallpaper.html",
    "continuum_wallpaper.webm",
    "continuum_wallpaper.mp4",
    "continuum_wallpaper.gif",
    "continuum_wallpaper.png",
    "continuum_wallpaper.jpg",
    "continuum_wallpaper.jpeg",
]


def _select_background_media() -> str:
    configured = os.environ.get("CONTINUUM_BACKGROUND_MEDIA") or os.environ.get("CONTINUUM_WALLPAPER_MEDIA") or ""
    selected = configured.strip()
    if selected:
        return selected
    asset_dir = Path(__file__).parent / "assets"
    for name in BACKGROUND_MEDIA_CANDIDATES:
        candidate = asset_dir / name
        if candidate.exists():
            return str(candidate)
    return ""


def _media_src(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).parent / value
    try:
        rel = path.resolve().relative_to(Path(__file__).parent.resolve())
        return "/gradio_api/file=" + quote(str(rel).replace("\\", "/"), safe="/")
    except Exception:
        return "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:")


def _wallpaper_runtime_state() -> dict[str, Any]:
    selected = _select_background_media()
    suffix = Path(selected.split("?", 1)[0]).suffix.lower() if selected else ""
    return {
        "schema": "champion-continuum/expressive-wallpaper/v1",
        "active": bool(selected),
        "asset": selected,
        "kind": "web_wallpaper" if suffix in {".html", ".htm"} else ("video" if suffix in {".webm", ".mp4", ".mov", ".m4v"} else ("image" if selected else "none")),
        "speech_rain_ready": bool(selected and suffix in {".html", ".htm"}),
        "control_contract": {
            "types": ["continuum:speech-rain", "continuum:wallpaper-control"],
            "transport": "event log -> deck timer -> postMessage to embedded wallpaper iframe -> iframe receipt",
            "truth_boundary": "Tool success queues a browser command; visible/render truth requires the deck's Probe Wallpaper Bridge receipt.",
            "inputs": ["assistant_text", "council_text", "daemon_directive", "continuum_wallpaper_text", "continuum_wallpaper_control", "continuum_wallpaper_preset"],
            "outputs": ["glyph_rain", "pattern", "direction", "color", "speed", "intensity", "font_size", "audio_reactivity", "settings_modal"],
            "settings_json_keys": [
                "fontSize", "characterSize", "pattern", "direction", "primaryColor", "secondaryColor",
                "speed", "intensity", "density", "characterSet", "customCharacters", "colorPreset",
                "hueReactivity", "saturationGain", "brightnessDepth", "audioReactive", "audioReverse",
                "audioDiagonals", "autoOrchestrator", "reverseFlow", "settingsPanel", "canvasOpacity",
            ],
            "pattern_values": ["classic", "rainbow", "pentad", "chaos", "harmonic", "particles"],
            "pattern_aliases": {"rain": "classic", "matrix": "classic", "prism": "pentad", "waves": "harmonic"},
            "commands": [
                "chaos_once", "toggle_audio", "audio_on", "audio_off", "auto_on", "auto_off",
                "reverse_flow", "settings_open", "settings_minimize", "settings_close",
            ],
            "mutates_external_state": False,
        },
    }


def _latest_wallpaper_event(last_event_id: str = "") -> tuple[str, str]:
    event_log = BRAIN_DIR / "continuum_link_events.jsonl"
    last_seen = str(last_event_id or "")
    if not event_log.exists():
        return "", last_seen
    try:
        lines = event_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", last_seen
    for line in reversed(lines[-80:]):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("kind") not in {"continuum.wallpaper.text", "continuum.wallpaper.control"}:
            continue
        event_id = str(event.get("event_id") or "")
        if event_id and event_id == last_seen:
            return "", last_seen
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        text = str(payload.get("text") or "").strip()
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        command_text = str(payload.get("command") or "").strip()
        settings_json = str(payload.get("settings_json") or "").strip()
        if not text and not settings and not command_text and not settings_json:
            continue
        command = {
            "text": text[:2400],
            "mode": str(payload.get("mode") or "rain"),
            "source": str(payload.get("source") or event.get("source") or "wallpaper-event"),
            "settings": settings,
            "settings_json": settings_json,
            "command": command_text,
            "slot": str(payload.get("slot") or event.get("slot") or "wallpaper"),
            "event_id": event_id,
        }
        return json.dumps(command, ensure_ascii=False), event_id or last_seen
    return "", last_seen


def _background_media_html() -> str:
    selected = _select_background_media()
    if not selected:
        return ""
    src = _media_src(selected)
    suffix = Path(selected.split("?", 1)[0]).suffix.lower()
    if suffix in {".webm", ".mp4", ".mov", ".m4v"}:
        return (
            "<div id='continuum-wallpaper' data-wallpaper-kind='video'>"
            f"<video src='{html.escape(src)}' autoplay muted loop playsinline></video>"
            "</div>"
        )
    if suffix in {".html", ".htm"}:
        return (
            "<div id='continuum-wallpaper' data-wallpaper-kind='web_wallpaper'>"
            f"<iframe src='{html.escape(src)}' title='Continuum wallpaper' aria-hidden='true'></iframe>"
            "</div>"
        )
    return (
        "<div id='continuum-wallpaper' data-wallpaper-kind='image'>"
        f"<img src='{html.escape(src)}' alt='' aria-hidden='true'>"
        "</div>"
    )


def _launch_allowed_paths() -> list[str]:
    allowed = [str(Path(__file__).parent / "assets")]
    configured = os.environ.get("CONTINUUM_BACKGROUND_MEDIA") or os.environ.get("CONTINUUM_WALLPAPER_MEDIA") or ""
    if configured and not configured.startswith(("http://", "https://", "data:")):
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).parent / configured
        parent = str(path.parent)
        if parent not in allowed:
            allowed.append(parent)
    return allowed

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

_CONTINUUM_FOOTER_LINK_URL = os.environ.get(
    "CONTINUUM_LINK_URL",
    f"http://127.0.0.1:{os.environ.get('CONTINUUM_LINK_PORT', '7871')}",
)
_CONTINUUM_FOOTER_MCP_URL = os.environ.get(
    "CONTINUUM_MCP_URL",
    f"http://127.0.0.1:{os.environ.get('CONTINUUM_MCP_PORT', '7872')}",
)
_CONTINUUM_FOOTER_BOOTSTRAP = json.dumps(
    {
        "links": _peer_link_values(),
        "peer_status": _peer_link_status_text(),
        "auth": {
            "mode": "huggingface_space_oauth" if RUNNING_ON_HF_SPACE and not CLI_BRAIN else ("cli_brain" if CLI_BRAIN else "resident_model_or_provider"),
            "hf_token_present": bool(HF_TOKEN),
            "login_button_visible": bool(not CLI_BRAIN),
        },
        "provider": {
            "default_provider": provider_registry_state()["huggingface_inference_providers"]["default_provider"],
            "default_model": provider_registry_state()["huggingface_inference_providers"]["default_model"],
        },
        "endpoints": {
            "link_service": _CONTINUUM_FOOTER_LINK_URL,
            "link_sse": f"{_CONTINUUM_FOOTER_LINK_URL}/sse?slot=*",
            "mcp_sse": f"{_CONTINUUM_FOOTER_MCP_URL}/mcp/sse",
            "mcp_http": f"{_CONTINUUM_FOOTER_MCP_URL}/mcp",
        },
    },
    ensure_ascii=False,
)

CONTINUUM_SETTINGS_HEAD = """
<script>
(() => {
  const sectionId = "continuum-footer-settings";
  const bootstrap = __CONTINUUM_FOOTER_BOOTSTRAP__;
  const labels = ["Service 1", "Service 2", "Service 3", "Service 4", "Service 5"];
  const html = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[ch]);

  function hasSettingsText(node) {
    const text = node.innerText || node.textContent || "";
    return text.includes("Display Theme") &&
      text.includes("Progressive Web App") &&
      text.includes("Language");
  }

  function isUsablePanel(node) {
    if (!node || node === document.body || node.tagName === "GRADIO-APP") return false;
    const style = window.getComputedStyle(node);
    if (style.display === "none" || style.visibility === "hidden") return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 80 && rect.height > 80;
  }

  function panelScore(node) {
    const rect = node.getBoundingClientRect();
    const textLen = (node.innerText || node.textContent || "").length;
    return (rect.width * rect.height) + textLen;
  }

  function settingsPanel() {
    const direct = Array.from(document.querySelectorAll("dialog,[role='dialog'],section,div"))
      .filter((node) => hasSettingsText(node) && isUsablePanel(node));
    if (direct.length) {
      direct.sort((a, b) => panelScore(a) - panelScore(b));
      return direct[0];
    }

    const anchors = Array.from(document.querySelectorAll("h1,h2,h3,h4,label,span,p,div"))
      .filter((node) => (node.innerText || node.textContent || "").includes("Display Theme"));
    const candidates = [];
    for (const anchor of anchors) {
      let node = anchor;
      for (let depth = 0; node && depth < 10; depth += 1, node = node.parentElement) {
        if (hasSettingsText(node) && isUsablePanel(node)) {
          candidates.push(node);
          break;
        }
      }
    }
    candidates.sort((a, b) => panelScore(a) - panelScore(b));
    return candidates[0] || null;
  }

  function mainServiceInputs() {
    const shell = document.querySelector("#peer-link-shell");
    return shell ? Array.from(shell.querySelectorAll("textarea,input")).slice(0, 5) : [];
  }

  function readMainValues() {
    const values = mainServiceInputs().map((input) => input.value || "");
    const fallback = (bootstrap.links || ["", "", "", "", ""]).slice(0, 5);
    while (values.length < 5) values.push(fallback[values.length] || "");
    return values;
  }

  function setValue(input, value) {
    const proto = Object.getPrototypeOf(input);
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) setter.call(input, value);
    else input.value = value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function clickMainSave() {
    const shell = document.querySelector("#peer-link-shell");
    if (!shell) return false;
    const save = Array.from(shell.querySelectorAll("button"))
      .find((button) => (button.innerText || "").includes("Save & Connect Services"));
    if (!save) return false;
    save.click();
    return true;
  }

  function statusGrid(data) {
    const provider = bootstrap.provider || {};
    const auth = bootstrap.auth || {};
    const endpoints = bootstrap.endpoints || {};
    return `
      <div class="cc-settings-grid">
        <div><b>HF auth</b><span>${html(auth.mode || "available")}</span></div>
        <div><b>Secret fallback</b><span>${auth.hf_token_present ? "configured" : "not configured"}</span></div>
        <div><b>Provider</b><span>${html(provider.default_provider || "auto")}:${html(provider.default_model || "")}</span></div>
        <div><b>MCP/SSE</b><span>${html(endpoints.mcp_sse || "")}</span></div>
      </div>
    `;
  }

  function mount(panel) {
    if (!panel) return;
    const existing = panel.querySelector("#" + sectionId);
    if (existing) {
      draw(existing, readMainValues());
      return;
    }
    const section = document.createElement("div");
    section.id = sectionId;
    section.innerHTML = `
      <style>
        #${sectionId} { margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(120,113,108,.35); }
        #${sectionId} h3 { margin: 0 0 8px; font-size: 1rem; }
        #${sectionId} p { margin: 0 0 10px; opacity: .78; line-height: 1.35; }
        #${sectionId} .cc-settings-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 8px 0 12px; }
        #${sectionId} .cc-settings-grid div { border: 1px solid rgba(120,113,108,.28); border-radius: 7px; padding: 8px; }
        #${sectionId} .cc-settings-grid b, #${sectionId} label { display: block; font-size: .78rem; opacity: .72; margin-bottom: 4px; }
        #${sectionId} .cc-settings-grid span { display: block; overflow-wrap: anywhere; font-size: .86rem; }
        #${sectionId} input { width: 100%; box-sizing: border-box; border-radius: 7px; border: 1px solid rgba(120,113,108,.38); padding: 8px 9px; margin-bottom: 8px; background: rgba(17,16,15,.72); color: inherit; }
        #${sectionId} .cc-settings-actions { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
        #${sectionId} button { border-radius: 7px; border: 1px solid rgba(217,119,6,.45); background: #d97706; color: white; padding: 8px 10px; cursor: pointer; font-weight: 650; }
        #${sectionId} .cc-settings-note { font-size: .82rem; opacity: .72; }
        @media (max-width: 720px) { #${sectionId} .cc-settings-grid { grid-template-columns: 1fr; } }
      </style>
      <h3>Continuum Settings</h3>
      <p>Added to this same Gradio settings menu. The original language, theme, and PWA controls stay above.</p>
      <div data-role="status">Loading Continuum settings...</div>
      <div data-role="links"></div>
      <div class="cc-settings-actions">
        <button type="button" data-role="save">Save & Connect Services</button>
        <span class="cc-settings-note" data-role="message"></span>
      </div>`;
    panel.appendChild(section);

    const status = section.querySelector("[data-role='status']");
    const links = section.querySelector("[data-role='links']");
    const message = section.querySelector("[data-role='message']");

    function localDraw(values) {
      while (values.length < 5) values.push("");
      status.innerHTML = statusGrid();
      links.innerHTML = values.map((value, idx) => `
        <label>${labels[idx]}</label>
        <input data-slot="${idx}" type="url" value="${html(value)}"
          placeholder="${idx === 0 ? "http://127.0.0.1:7872/mcp/sse" : "https://friend.example/mcp/sse"}" />`
      ).join("");
      message.textContent = bootstrap.peer_status || "";
    }
    function draw(target, values) {
      const targetStatus = target.querySelector("[data-role='status']");
      const targetLinks = target.querySelector("[data-role='links']");
      const targetMessage = target.querySelector("[data-role='message']");
      if (!targetStatus || !targetLinks || !targetMessage) return;
      while (values.length < 5) values.push("");
      targetStatus.innerHTML = statusGrid();
      targetLinks.innerHTML = values.map((value, idx) => `
        <label>${labels[idx]}</label>
        <input data-slot="${idx}" type="url" value="${html(value)}"
          placeholder="${idx === 0 ? "http://127.0.0.1:7872/mcp/sse" : "https://friend.example/mcp/sse"}" />`
      ).join("");
      targetMessage.textContent = bootstrap.peer_status || "";
    }

    localDraw(readMainValues());

    section.querySelector("[data-role='save']").addEventListener("click", async () => {
      const values = Array.from(section.querySelectorAll("input[data-slot]"))
        .sort((a, b) => Number(a.dataset.slot) - Number(b.dataset.slot))
        .map((input) => input.value.trim());
      message.textContent = "Saving...";
      const inputs = mainServiceInputs();
      inputs.forEach((input, idx) => setValue(input, values[idx] || ""));
      const connected = clickMainSave();
      message.textContent = connected ? "Saved and sent to the visible service connector." : "Saved in the menu. Main service boxes were not found on this render.";
    });
  }

  function tick() {
    const panel = settingsPanel();
    if (panel) mount(panel);
  }

  window.addEventListener("load", () => {
    tick();
    new MutationObserver(tick).observe(document.body, { childList: true, subtree: true });
    document.addEventListener("click", () => setTimeout(tick, 80), true);
  });
})();
</script>
""".replace("__CONTINUUM_FOOTER_BOOTSTRAP__", _CONTINUUM_FOOTER_BOOTSTRAP)

CONTINUUM_WALLPAPER_HEAD = """
<script>
(() => {
  const BLOB_KEY = "continuum.wallpaperBlob.v1";
  let wallpaperBlobState = null;
  let wallpaperSaveTimer = null;

  function textOf(value) {
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(textOf).filter(Boolean).join("\\n");
    if (typeof value === "object") {
      if ("content" in value) return textOf(value.content);
      if ("text" in value) return textOf(value.text);
      if ("value" in value) return textOf(value.value);
    }
    return String(value || "");
  }

  function latestAssistantText(history) {
    if (!Array.isArray(history)) return "";
    for (let i = history.length - 1; i >= 0; i -= 1) {
      const row = history[i];
      if (Array.isArray(row)) {
        const text = textOf(row[1]).trim();
        if (text) return text;
      } else if (row && typeof row === "object") {
        const role = String(row.role || "").toLowerCase();
        if (role === "assistant") {
          const text = textOf(row.content).trim();
          if (text) return text;
        }
      }
    }
    return "";
  }

  function wallpaperFrame() {
    const shell = document.querySelector("#continuum-wallpaper");
    return shell ? shell.querySelector("iframe") : null;
  }

  function wallpaperShell() {
    return document.querySelector("#continuum-wallpaper");
  }

  function wallpaperDefaults() {
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    return {
      blob: false,
      collapsed: false,
      x: Math.max(18, vw - 470),
      y: 92,
      w: Math.min(440, vw - 36),
      h: Math.min(310, vh - 118)
    };
  }

  function loadWallpaperBlobState() {
    const base = wallpaperDefaults();
    try {
      const saved = JSON.parse(localStorage.getItem(BLOB_KEY) || "{}");
      return { ...base, ...saved };
    } catch {
      return base;
    }
  }

  function saveWallpaperBlobState() {
    clearTimeout(wallpaperSaveTimer);
    wallpaperSaveTimer = setTimeout(() => {
      try { localStorage.setItem(BLOB_KEY, JSON.stringify(wallpaperBlobState)); } catch {}
    }, 90);
  }

  function clampWallpaperBlob() {
    const panel = wallpaperBlobState || loadWallpaperBlobState();
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    panel.w = Math.min(Math.max(Number(panel.w) || 440, 220), vw - 24);
    panel.h = Math.min(Math.max(Number(panel.h) || 310, 58), vh - 24);
    panel.x = Math.min(Math.max(Number(panel.x) || 18, 8), Math.max(8, vw - Math.min(panel.w, vw - 24) - 8));
    panel.y = Math.min(Math.max(Number(panel.y) || 92, 8), Math.max(8, vh - Math.min(panel.h, vh - 24) - 8));
  }

  function ensureWallpaperChrome() {
    const shell = wallpaperShell();
    if (!shell || shell.dataset.wallpaperChromeBound === "1") return shell;
    shell.dataset.wallpaperChromeBound = "1";

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "cc-wallpaper-blob-handle";
    handle.setAttribute("aria-label", "Move Matrix Rain wallpaper blob");
    shell.appendChild(handle);

    const min = document.createElement("button");
    min.type = "button";
    min.className = "cc-wallpaper-blob-button cc-wallpaper-blob-min";
    min.setAttribute("aria-label", "Collapse Matrix Rain wallpaper blob");
    min.textContent = "–";
    shell.appendChild(min);

    const close = document.createElement("button");
    close.type = "button";
    close.className = "cc-wallpaper-blob-button cc-wallpaper-blob-close";
    close.setAttribute("aria-label", "Return Matrix Rain wallpaper to full background");
    close.textContent = "x";
    shell.appendChild(close);

    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "cc-wallpaper-blob-grip";
    grip.setAttribute("aria-label", "Resize Matrix Rain wallpaper blob");
    shell.appendChild(grip);

    for (const edge of ["right", "bottom"]) {
      const rail = document.createElement("button");
      rail.type = "button";
      rail.className = "cc-wallpaper-blob-edge";
      rail.dataset.edge = edge;
      rail.setAttribute("aria-label", `Resize Matrix Rain wallpaper blob ${edge}`);
      shell.appendChild(rail);
    }

    let drag = null;
    handle.addEventListener("pointerdown", (event) => {
      if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
      if (!wallpaperBlobState.blob) return;
      drag = {
        id: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        x: wallpaperBlobState.x,
        y: wallpaperBlobState.y
      };
      shell.classList.add("cc-wallpaper-dragging");
      handle.setPointerCapture?.(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    });
    handle.addEventListener("pointermove", (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      wallpaperBlobState.x = drag.x + event.clientX - drag.startX;
      wallpaperBlobState.y = drag.y + event.clientY - drag.startY;
      applyWallpaperBlobState();
      event.preventDefault();
    });
    const endDrag = (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      drag = null;
      shell.classList.remove("cc-wallpaper-dragging");
      applyWallpaperBlobState();
    };
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);

    let resize = null;
    function bindResize(node, edge) {
      node.addEventListener("pointerdown", (event) => {
        if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
        if (!wallpaperBlobState.blob || wallpaperBlobState.collapsed) return;
        resize = {
          id: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          w: wallpaperBlobState.w,
          h: wallpaperBlobState.h,
          edge
        };
        node.setPointerCapture?.(event.pointerId);
        event.preventDefault();
        event.stopPropagation();
      });
      node.addEventListener("pointermove", (event) => {
        if (!resize || resize.id !== event.pointerId) return;
        if (resize.edge === "right" || resize.edge === "corner") {
          wallpaperBlobState.w = resize.w + event.clientX - resize.startX;
        }
        if (resize.edge === "bottom" || resize.edge === "corner") {
          wallpaperBlobState.h = resize.h + event.clientY - resize.startY;
        }
        applyWallpaperBlobState();
        event.preventDefault();
      });
      const endResize = (event) => {
        if (!resize || resize.id !== event.pointerId) return;
        resize = null;
        applyWallpaperBlobState();
      };
      node.addEventListener("pointerup", endResize);
      node.addEventListener("pointercancel", endResize);
    }
    bindResize(grip, "corner");
    shell.querySelectorAll(".cc-wallpaper-blob-edge").forEach((node) => bindResize(node, node.dataset.edge || "corner"));

    min.addEventListener("click", (event) => {
      if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
      wallpaperBlobState.blob = true;
      wallpaperBlobState.collapsed = !wallpaperBlobState.collapsed;
      applyWallpaperBlobState();
      event.preventDefault();
      event.stopPropagation();
    });

    close.addEventListener("click", (event) => {
      if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
      wallpaperBlobState.blob = false;
      wallpaperBlobState.collapsed = false;
      applyWallpaperBlobState();
      event.preventDefault();
      event.stopPropagation();
    });

    handle.addEventListener("dblclick", (event) => {
      if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
      wallpaperBlobState.collapsed = !wallpaperBlobState.collapsed;
      applyWallpaperBlobState();
      event.preventDefault();
      event.stopPropagation();
    });

    return shell;
  }

  function applyWallpaperBlobState() {
    const shell = ensureWallpaperChrome();
    if (!shell) return;
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    clampWallpaperBlob();
    shell.classList.toggle("cc-wallpaper-blob", Boolean(wallpaperBlobState.blob));
    shell.classList.toggle("cc-wallpaper-collapsed", Boolean(wallpaperBlobState.blob && wallpaperBlobState.collapsed));
    shell.style.setProperty("--cc-wallpaper-blob-x", `${Math.round(wallpaperBlobState.x)}px`);
    shell.style.setProperty("--cc-wallpaper-blob-y", `${Math.round(wallpaperBlobState.y)}px`);
    shell.style.setProperty("--cc-wallpaper-blob-w", `${Math.round(wallpaperBlobState.w)}px`);
    shell.style.setProperty("--cc-wallpaper-blob-h", `${Math.round(wallpaperBlobState.h)}px`);
    document.documentElement.classList.toggle("cc-wallpaper-active", true);
    document.documentElement.dataset.continuumWallpaperMode = wallpaperBlobState.blob ? "blob-underlay" : "full-underlay";
    saveWallpaperBlobState();
  }

  function setWallpaperBlobMode(blob, collapsed = false) {
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    wallpaperBlobState.blob = Boolean(blob);
    wallpaperBlobState.collapsed = Boolean(collapsed);
    applyWallpaperBlobState();
  }

  window.continuumSpeechRain = function(text, source = "council") {
    const clean = textOf(text).replace(/\\s+/g, " ").trim();
    if (!clean) return false;
    const frame = wallpaperFrame();
    const message = {
      type: "continuum:speech-rain",
      source,
      text: clean.slice(0, 2400),
      ts: Date.now()
    };
    return window.continuumWallpaperPostMessage(message, "speech-rain");
  };

  const wallpaperPendingMessages = [];
  let wallpaperDrainTimer = null;

  function recordWallpaperParentReceipt(kind, data = {}) {
    const receipt = {
      kind,
      ok: data.ok !== false,
      ts: Date.now(),
      ...data
    };
    const text = JSON.stringify(receipt);
    document.documentElement.dataset.continuumWallpaperReceipt = text.slice(0, 1200);
    try { localStorage.setItem("continuum.wallpaperReceipt.v1", text); } catch {}
    return receipt;
  }

  function drainWallpaperMessages() {
    clearTimeout(wallpaperDrainTimer);
    wallpaperDrainTimer = null;
    if (!wallpaperPendingMessages.length) return;
    const pending = wallpaperPendingMessages.splice(0, wallpaperPendingMessages.length);
    for (const message of pending) {
      if (!window.continuumWallpaperPostMessage(message, message.type || "queued", false)) {
        wallpaperPendingMessages.push(message);
      }
    }
    if (wallpaperPendingMessages.length) {
      wallpaperDrainTimer = setTimeout(drainWallpaperMessages, 500);
    }
  }

  window.continuumWallpaperPostMessage = function(message, kind = "wallpaper", allowQueue = true) {
    const frame = wallpaperFrame();
    if (!frame || !frame.contentWindow) {
      if (allowQueue) {
        wallpaperPendingMessages.push(message);
        if (!wallpaperDrainTimer) wallpaperDrainTimer = setTimeout(drainWallpaperMessages, 250);
      }
      document.documentElement.dataset.continuumWallpaperControl = allowQueue ? "queued" : "missing-frame";
      recordWallpaperParentReceipt(kind, {
        ok: false,
        state: allowQueue ? "queued_waiting_for_iframe" : "missing_iframe",
        pending: wallpaperPendingMessages.length
      });
      return allowQueue;
    }
    frame.contentWindow.postMessage(message, "*");
    document.documentElement.dataset.continuumWallpaperControl = "posted";
    recordWallpaperParentReceipt(kind, {
      ok: true,
      state: "posted_to_iframe",
      pending: wallpaperPendingMessages.length,
      message_type: message.type || ""
    });
    return true;
  };

  window.continuumWallpaperControl = function(payload = {}) {
    const data = typeof payload === "string" ? { text: payload } : { ...(payload || {}) };
    return window.continuumWallpaperPostMessage({
      type: "continuum:wallpaper-control",
      source: data.source || "wallpaper-control",
      text: textOf(data.text || "").slice(0, 2400),
      mode: data.mode || "",
      event_id: data.event_id || "",
      command: data.command || "",
      settings: data.settings || {},
      settings_json: data.settings_json || "",
      slot: data.slot || "wallpaper",
      ts: Date.now()
    }, "wallpaper-control");
  };

  window.continuumWallpaperCommand = function(payload = {}) {
    const data = typeof payload === "string" ? { text: payload } : { ...(payload || {}) };
    if (data.mode === "blob") setWallpaperBlobMode(true, false);
    if (data.mode === "collapsed") setWallpaperBlobMode(true, true);
    if (data.mode === "background") setWallpaperBlobMode(false, false);
    if (data.command || data.settings || data.settings_json) return window.continuumWallpaperControl(data);
    if (data.text) {
      window.continuumSpeechRain(data.text, data.source || "wallpaper-command");
      return window.continuumWallpaperControl(data);
    }
    return true;
  };

  window.continuumWallpaperBridgeProbe = function() {
    const shell = wallpaperShell();
    const frame = wallpaperFrame();
    let iframe = {};
    try {
      iframe = frame?.contentWindow?.continuumWallpaperProbe?.() || {};
    } catch (err) {
      iframe = { error: err?.message || String(err) };
    }
    let receipt = {};
    try { receipt = JSON.parse(localStorage.getItem("continuum.wallpaperReceipt.v1") || "{}"); } catch {}
    const rect = shell ? shell.getBoundingClientRect() : null;
    return {
      shell_present: Boolean(shell),
      iframe_present: Boolean(frame),
      iframe_src: frame?.getAttribute("src") || "",
      iframe_window: Boolean(frame?.contentWindow),
      active_class: document.documentElement.classList.contains("cc-wallpaper-active"),
      mode: document.documentElement.dataset.continuumWallpaperMode || "",
      parent_state: document.documentElement.dataset.continuumWallpaperControl || "",
      pending_messages: wallpaperPendingMessages.length,
      shell_rect: rect ? { width: Math.round(rect.width), height: Math.round(rect.height), x: Math.round(rect.x), y: Math.round(rect.y) } : null,
      last_receipt: receipt,
      iframe
    };
  };

  window.addEventListener("message", (event) => {
    const data = event && event.data;
    if (!data || data.type !== "continuum:wallpaper-receipt") return;
    recordWallpaperParentReceipt(data.kind || "iframe-receipt", {
      ok: data.ok !== false,
      state: data.state || "iframe_applied",
      event_id: data.event_id || "",
      detail: data.detail || {},
      pending: wallpaperPendingMessages.length
    });
  });

  window.continuumToggleWallpaperBlob = function() {
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    wallpaperBlobState.blob = !wallpaperBlobState.blob;
    if (wallpaperBlobState.blob) wallpaperBlobState.collapsed = false;
    applyWallpaperBlobState();
  };

  window.continuumCollapseWallpaperBlob = function() {
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    wallpaperBlobState.blob = true;
    wallpaperBlobState.collapsed = !wallpaperBlobState.collapsed;
    applyWallpaperBlobState();
  };

  window.continuumWallpaperBlobState = function() {
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    return { ...wallpaperBlobState };
  };

  window.continuumSpeechRainFromHistory = function(history) {
    return window.continuumSpeechRain(latestAssistantText(history), "chat-history");
  };

  function bootWallpaper() {
    if (!wallpaperBlobState) wallpaperBlobState = loadWallpaperBlobState();
    if (wallpaperShell()) applyWallpaperBlobState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootWallpaper, { once: true });
  } else {
    bootWallpaper();
  }
  new MutationObserver(bootWallpaper).observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("resize", () => { if (wallpaperBlobState) applyWallpaperBlobState(); });
})();
</script>
"""

CONTINUUM_BLOB_OVERLAY_HEAD = """
<script>
(() => {
  const KEY = "continuum.blobOverlay.v2";
  const PROBE_KEY = "continuum.blobProbe.v1";
  const PANEL_META = {
    chat: { selector: "#chat-col", title: "Behold: Chat Blob" },
    support: { selector: "#support-col", title: "Behold: Surface Blob" }
  };
  let state = null;
  let saveTimer = null;
  let probeSaveTimer = null;
  let didProbeInit = false;
  const historyStack = [];
  const probeMotion = {};
  const probeState = {
    version: 1,
    updatedAt: 0,
    summary: {},
    panels: {},
    buffer: []
  };

  function defaults() {
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    return {
      meld: false,
      zSeed: 50,
      panels: {
        chat: {
          enabled: false,
          collapsed: false,
          x: 24,
          y: 72,
          w: Math.min(640, Math.round(vw * 0.58), vw - 96),
          h: Math.min(560, Math.round(vh * 0.68), vh - 128),
          edge: "",
          z: 51
        },
        support: {
          enabled: false,
          collapsed: false,
          x: Math.max(24, vw - Math.min(560, Math.round(vw * 0.48), vw - 96) - 36),
          y: 96,
          w: Math.min(560, Math.round(vw * 0.48), vw - 96),
          h: Math.min(520, Math.round(vh * 0.62), vh - 142),
          edge: "",
          z: 50
        }
      }
    };
  }

  function loadState() {
    const base = defaults();
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
      return {
        meld: Boolean(saved.meld),
        zSeed: Number(saved.zSeed) || base.zSeed,
        panels: {
          chat: { ...base.panels.chat, ...(saved.panels?.chat || {}) },
          support: { ...base.panels.support, ...(saved.panels?.support || {}) }
        }
      };
    } catch {
      return base;
    }
  }

  function queueSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      try { localStorage.setItem(KEY, JSON.stringify(state)); } catch {}
    }, 80);
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
  }

  function round3(value) {
    return Math.round((Number(value) || 0) * 1000) / 1000;
  }

  function cloneProbe() {
    try { return JSON.parse(JSON.stringify(probeState)); } catch { return probeState; }
  }

  function queueProbeSave() {
    clearTimeout(probeSaveTimer);
    probeSaveTimer = setTimeout(() => {
      try {
        const payload = {
          version: probeState.version,
          updatedAt: probeState.updatedAt,
          summary: probeState.summary,
          panels: probeState.panels,
          buffer: probeState.buffer.slice(-40)
        };
        localStorage.setItem(PROBE_KEY, JSON.stringify(payload));
      } catch {}
    }, 160);
  }

  function absenceField(panel) {
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    return {
      left: Math.max(0, Math.round(panel.x)),
      right: Math.max(0, Math.round(vw - (panel.x + panel.w))),
      top: Math.max(0, Math.round(panel.y)),
      bottom: Math.max(0, Math.round(vh - (panel.y + panel.h)))
    };
  }

  function updateProbeSummary() {
    const panels = Object.values(probeState.panels || {});
    const active = panels.filter((panel) => panel.enabled);
    const dominant = active.reduce((best, panel) => {
      if (!best) return panel;
      return (panel.pressure + panel.resonance) > (best.pressure + best.resonance) ? panel : best;
    }, null);
    probeState.updatedAt = Date.now();
    probeState.summary = {
      active_panels: active.length,
      dominant_panel: dominant?.name || "",
      max_pressure: round3(Math.max(0, ...active.map((panel) => panel.pressure || 0))),
      max_resonance: round3(Math.max(0, ...active.map((panel) => panel.resonance || 0))),
      contact_panels: active.filter((panel) => panel.contact_edge).map((panel) => panel.name),
      settled: active.length > 0 && active.every((panel) => panel.settled || panel.collapsed),
      collision: active.some((panel) => panel.collision)
    };
    window.__continuumBlobProbe = probeState;
    document.documentElement.dataset.continuumBlobProbe = active.length ? "active" : "idle";
  }

  function measurePanel(name, cause = "render", extra = {}) {
    const panel = state?.panels?.[name];
    if (!panel) return null;
    const now = Date.now();
    const rect = panelRect(panel);
    const prev = probeMotion[name] || {
      left: rect.left,
      top: rect.top,
      w: panel.w,
      h: panel.h,
      lastMotionAt: now
    };
    const movement = Math.hypot(
      rect.left - prev.left,
      rect.top - prev.top,
      panel.w - prev.w,
      panel.h - prev.h
    );
    const speed = clamp01(movement / 110);
    const wall = wallContactMetrics(panel);
    const collisionPressure = panel.collision ? 0.86 : 0;
    const manualPressure = clamp01(extra.pressure);
    const pressure = clamp01(Math.max(speed * 0.92, wall.pressure, collisionPressure, manualPressure));
    const moving = Boolean(extra.moving) || speed > 0.018 || pressure > 0.16;
    const lastMotionAt = moving ? now : (prev.lastMotionAt || now);
    const enabled = Boolean(panel.enabled);
    const collapsed = Boolean(panel.collapsed);
    const resonance = enabled
      ? clamp01(Math.max(0.12, pressure * 0.76 + speed * 0.35 + (collapsed ? 0.02 : 0.09)))
      : 0;
    const settled = Boolean(enabled && !collapsed && now - lastMotionAt > 900 && pressure < 0.08);
    const info = {
      name,
      title: PANEL_META[name]?.title || name,
      cause,
      ts: now,
      enabled,
      collapsed,
      x: Math.round(panel.x),
      y: Math.round(panel.y),
      w: Math.round(panel.w),
      h: Math.round(panel.h),
      z: Math.round(Number(panel.z) || 0),
      pressure: round3(pressure),
      resonance: round3(resonance),
      speed: round3(speed),
      settled,
      collision: Boolean(panel.collision),
      contact_edge: wall.edge,
      contact_depth: round3(wall.depth),
      absence: absenceField(panel)
    };
    probeMotion[name] = {
      left: rect.left,
      top: rect.top,
      w: panel.w,
      h: panel.h,
      lastMotionAt
    };
    probeState.panels[name] = info;
    updateProbeSummary();
    return info;
  }

  function noteProbe(name, cause, extra = {}) {
    if (!state) state = loadState();
    const names = name && state.panels[name] ? [name] : Object.keys(PANEL_META);
    for (const panelName of names) measurePanel(panelName, cause, extra);
    probeState.buffer.push({
      ts: Date.now(),
      panel: name || "all",
      cause,
      summary: { ...probeState.summary }
    });
    while (probeState.buffer.length > 80) probeState.buffer.shift();
    queueProbeSave();
  }

  function clampPanel(panel) {
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    panel.w = Math.min(Math.max(Number(panel.w) || 520, 240), vw - 42);
    panel.h = Math.min(Math.max(Number(panel.h) || 420, 58), vh - 42);
    panel.x = Math.min(Math.max(Number(panel.x) || 18, 8), Math.max(8, vw - Math.min(panel.w, vw - 42) - 8));
    panel.y = Math.min(Math.max(Number(panel.y) || 72, 8), Math.max(8, vh - Math.min(panel.h, vh - 42) - 8));
  }

  function panelElement(name) {
    return document.querySelector(PANEL_META[name].selector);
  }

  function ensureBlobChrome(el) {
    if (!el.querySelector(":scope > .cc-blob-energy")) {
      const energy = document.createElement("div");
      energy.className = "cc-blob-energy";
      for (let i = 0; i < 12; i += 1) {
        const bubble = document.createElement("i");
        bubble.className = "cc-blob-bubble";
        bubble.style.setProperty("--x", `${8 + ((i * 17) % 84)}%`);
        bubble.style.setProperty("--s", `${16 + ((i * 11) % 38)}px`);
        bubble.style.setProperty("--dur", `${7 + (i % 5) * 1.4}s`);
        bubble.style.setProperty("--delay", `${-1 * (i * 0.73).toFixed(2)}s`);
        bubble.style.setProperty("--drift", `${(i % 2 ? 1 : -1) * (18 + (i % 4) * 12)}px`);
        energy.appendChild(bubble);
      }
      for (let i = 0; i < 5; i += 1) {
        const spark = document.createElement("i");
        spark.className = "cc-blob-spark";
        spark.style.setProperty("--x", `${10 + ((i * 23) % 72)}%`);
        spark.style.setProperty("--y", `${18 + ((i * 19) % 62)}%`);
        spark.style.setProperty("--w", `${90 + i * 28}px`);
        spark.style.setProperty("--rot", `${-26 + i * 13}deg`);
        spark.style.setProperty("--dur", `${4.6 + i * 0.9}s`);
        spark.style.setProperty("--delay", `${-1 * (i * 1.1).toFixed(2)}s`);
        energy.appendChild(spark);
      }
      el.prepend(energy);
    }
    if (!el.querySelector(":scope > .cc-blob-resize-grip")) {
      const grip = document.createElement("button");
      grip.type = "button";
      grip.className = "cc-blob-resize-grip";
      grip.setAttribute("aria-label", "Resize blob");
      el.appendChild(grip);
    }
    for (const edge of ["right", "bottom", "corner"]) {
      if (!el.querySelector(`:scope > .cc-blob-resize-edge[data-edge="${edge}"]`)) {
        const rail = document.createElement("button");
        rail.type = "button";
        rail.className = "cc-blob-resize-edge";
        rail.dataset.edge = edge;
        rail.setAttribute("aria-label", `Resize blob ${edge}`);
        el.appendChild(rail);
      }
    }
    if (!el.querySelector(":scope > .cc-blob-close")) {
      const close = document.createElement("button");
      close.type = "button";
      close.className = "cc-blob-close";
      close.setAttribute("aria-label", "Close blob overlay");
      close.textContent = "x";
      el.appendChild(close);
    }
  }

  function edgeClass(edge) {
    return {
      left: "cc-blob-edge-left",
      right: "cc-blob-edge-right",
      top: "cc-blob-edge-top",
      bottom: "cc-blob-edge-bottom"
    }[edge] || "";
  }

  function applyPanel(name) {
    const el = panelElement(name);
    if (!el || !state?.panels?.[name]) return;
    const panel = state.panels[name];
    clampPanel(panel);
    ensureBlobChrome(el);
    const probeInfo = measurePanel(name, "apply");
    el.dataset.blobTitle = PANEL_META[name].title;
    el.classList.toggle("cc-blob-enabled", Boolean(panel.enabled));
    el.classList.toggle("cc-blob-collapsed", Boolean(panel.enabled && panel.collapsed));
    el.classList.toggle("cc-blob-collide", Boolean(panel.enabled && panel.collision));
    el.classList.toggle("cc-blob-probe-settled", Boolean(panel.enabled && probeInfo?.settled));
    el.classList.toggle("cc-blob-probe-contact", Boolean(panel.enabled && probeInfo?.contact_edge));
    el.classList.remove("cc-blob-edge-left", "cc-blob-edge-right", "cc-blob-edge-top", "cc-blob-edge-bottom");
    const edge = panel.enabled ? edgeClass(panel.edge) : "";
    if (edge) el.classList.add(edge);
    if (panel.enabled) {
      el.style.setProperty("--cc-blob-x", `${panel.x}px`);
      el.style.setProperty("--cc-blob-y", `${panel.y}px`);
      el.style.setProperty("--cc-blob-w", `${panel.w}px`);
      el.style.setProperty("--cc-blob-h", `${panel.h}px`);
      el.style.setProperty("--cc-blob-z", `${Number(panel.z) || 40}`);
      const pressure = probeInfo?.pressure ?? 0;
      const resonance = probeInfo?.resonance ?? 0.12;
      el.style.setProperty("--cc-blob-pressure", `${pressure}`);
      el.style.setProperty("--cc-blob-resonance", `${resonance}`);
      el.style.setProperty("--cc-blob-settle", probeInfo?.settled ? "1" : "0");
      el.style.setProperty("--cc-blob-pressure-glow", `${Math.round(pressure * 36)}px`);
      el.style.setProperty("--cc-blob-resonance-glow", `${Math.round(resonance * 34)}px`);
      el.style.setProperty("--cc-blob-contact-glow", `${16 + Math.round(pressure * 34)}px`);
      el.style.setProperty("--cc-blob-saturate", `${round3(1.02 + resonance * 0.22)}`);
      el.style.setProperty("--cc-blob-brightness", `${round3(1 + pressure * 0.08)}`);
      el.style.setProperty("--cc-blob-energy-opacity", `${round3(0.74 + resonance * 0.22)}`);
      el.style.setProperty("--cc-blob-energy-blur", `${round3(Math.max(3.8, 7 - pressure * 2))}px`);
      el.style.setProperty("--cc-blob-energy-saturate", `${round3(1.22 + resonance * 0.46)}`);
      el.style.setProperty("--cc-blob-current-duration", `${round3(Math.max(7.5, 13 - resonance * 4))}s`);
    }
  }

  function apply() {
    if (!state) state = loadState();
    document.body.classList.toggle("cc-blob-meld", Boolean(state.meld));
    applyPanel("chat");
    applyPanel("support");
    queueSave();
    queueProbeSave();
  }

  function snapshot() {
    try { return JSON.parse(JSON.stringify(state || loadState())); } catch { return null; }
  }

  function remember() {
    const snap = snapshot();
    if (!snap) return;
    historyStack.push(snap);
    while (historyStack.length > 24) historyStack.shift();
  }

  function undoBlob() {
    const prior = historyStack.pop();
    if (!prior) return;
    state = prior;
    apply();
  }

  function anyEnabled() {
    return Object.keys(PANEL_META).some((name) => Boolean(state?.panels?.[name]?.enabled));
  }

  function anyExpanded() {
    return Object.keys(PANEL_META).some((name) => {
      const panel = state?.panels?.[name];
      return Boolean(panel?.enabled && !panel?.collapsed);
    });
  }

  function reduceBlobState() {
    if (!state) state = loadState();
    if (!anyEnabled()) return;
    remember();
    if (anyExpanded()) {
      for (const name of Object.keys(PANEL_META)) {
        const panel = state.panels[name];
        if (panel.enabled) panel.collapsed = true;
      }
    } else {
      for (const name of Object.keys(PANEL_META)) {
        state.panels[name].enabled = false;
        state.panels[name].collapsed = false;
        state.panels[name].edge = "";
      }
      state.meld = false;
    }
    apply();
  }

  function raisePanel(name) {
    if (!state) state = loadState();
    const panel = state.panels[name];
    if (!panel) return;
    state.zSeed = (Number(state.zSeed) || 50) + 1;
    panel.z = state.zSeed;
    applyPanel(name);
    queueSave();
  }

  function markImpact(el) {
    el.classList.remove("cc-blob-impact");
    void el.offsetWidth;
    el.classList.add("cc-blob-impact");
  }

  function wallContactMetrics(panel) {
    const vw = Math.max(360, window.innerWidth || 1280);
    const vh = Math.max(420, window.innerHeight || 860);
    const margin = 14;
    const contacts = [
      ["left", Math.max(0, margin - panel.x)],
      ["right", Math.max(0, panel.x + panel.w - (vw - margin))],
      ["top", Math.max(0, margin - panel.y)],
      ["bottom", Math.max(0, panel.y + panel.h - (vh - margin))]
    ].filter(([, depth]) => depth > 0);
    if (!contacts.length) return { edge: "", depth: 0, pressure: 0 };
    contacts.sort((a, b) => b[1] - a[1]);
    return {
      edge: contacts[0][0],
      depth: contacts[0][1],
      pressure: clamp01(contacts[0][1] / 72)
    };
  }

  function wallContact(panel) {
    return wallContactMetrics(panel).edge;
  }

  function panelRect(panel) {
    return {
      left: panel.x,
      top: panel.y,
      right: panel.x + panel.w,
      bottom: panel.y + panel.h,
      cx: panel.x + panel.w / 2,
      cy: panel.y + panel.h / 2
    };
  }

  function overlapDepth(a, b) {
    const x = Math.min(a.right, b.right) - Math.max(a.left, b.left);
    const y = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    if (x <= 0 || y <= 0) return null;
    return { x, y };
  }

  function clearCollisionSoon(...names) {
    window.setTimeout(() => {
      if (!state) return;
      for (const name of names) {
        if (state.panels[name]) {
          state.panels[name].collision = false;
          applyPanel(name);
        }
      }
    }, 360);
  }

  function resolveBlobCollisions(activeName) {
    if (!state?.panels?.[activeName]) return;
    const active = state.panels[activeName];
    if (!active.enabled || active.collapsed) return;
    const activeEl = panelElement(activeName);
    const activeRect = panelRect(active);
    for (const otherName of Object.keys(PANEL_META)) {
      if (otherName === activeName) continue;
      const other = state.panels[otherName];
      if (!other?.enabled || other.collapsed) continue;
      const depth = overlapDepth(activeRect, panelRect(other));
      if (!depth) continue;
      const otherEl = panelElement(otherName);
      active.collision = true;
      other.collision = true;
      if (depth.x < depth.y) {
        const direction = activeRect.cx <= panelRect(other).cx ? 1 : -1;
        const push = Math.min(96, Math.max(18, depth.x * 0.62));
        other.x += direction * push;
        active.x -= direction * Math.min(24, push * 0.22);
      } else {
        const direction = activeRect.cy <= panelRect(other).cy ? 1 : -1;
        const push = Math.min(84, Math.max(16, depth.y * 0.58));
        other.y += direction * push;
        active.y -= direction * Math.min(22, push * 0.22);
      }
      active.edge = wallContact(active);
      other.edge = wallContact(other);
      clampPanel(active);
      clampPanel(other);
      markImpact(activeEl);
      markImpact(otherEl);
      applyPanel(otherName);
      noteProbe(activeName, "collision", { moving: true, pressure: 0.86 });
      noteProbe(otherName, "collision", { moving: true, pressure: 0.86 });
      clearCollisionSoon(activeName, otherName);
    }
  }

  function togglePanel(name) {
    if (!state) state = loadState();
    remember();
    const panel = state.panels[name];
    panel.enabled = !panel.enabled;
    if (panel.enabled) panel.collapsed = false;
    raisePanel(name);
    apply();
    noteProbe(name, panel.enabled ? "enable" : "disable");
  }

  function collapseEnabled() {
    if (!state) state = loadState();
    remember();
    let touched = false;
    for (const name of Object.keys(PANEL_META)) {
      const panel = state.panels[name];
      if (panel.enabled) {
        panel.collapsed = !panel.collapsed;
        touched = true;
      }
    }
    if (!touched) {
      state.panels.chat.enabled = true;
      state.panels.chat.collapsed = true;
    }
    apply();
    noteProbe("", "collapse");
  }

  function resetPanels() {
    remember();
    state = defaults();
    try { localStorage.removeItem(KEY); } catch {}
    apply();
    noteProbe("", "reset");
  }

  function toggleMeld() {
    if (!state) state = loadState();
    remember();
    state.meld = !state.meld;
    state.panels.chat.enabled = true;
    state.panels.support.enabled = true;
    raisePanel("support");
    raisePanel("chat");
    apply();
    noteProbe("", "meld");
  }

  function bindPanel(name) {
    const el = panelElement(name);
    if (!el) return;
    ensureBlobChrome(el);

    if (el.dataset.blobChromeBound !== "1") {
      el.dataset.blobChromeBound = "1";
      let resize = null;
      function bindResizeHandle(handle, edge) {
        if (!handle || handle.dataset.blobResizeBound === "1") return;
        handle.dataset.blobResizeBound = "1";
        handle.addEventListener("pointerdown", (event) => {
          if (!state) state = loadState();
          const panel = state.panels[name];
          if (!panel.enabled || panel.collapsed) return;
          remember();
          raisePanel(name);
          resize = {
            id: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            w: panel.w,
            h: panel.h,
            edge
          };
          handle.setPointerCapture?.(event.pointerId);
          event.preventDefault();
          event.stopPropagation();
        });
        handle.addEventListener("pointermove", (event) => {
          if (!resize || resize.id !== event.pointerId) return;
          const panel = state.panels[name];
          if (resize.edge === "right" || resize.edge === "corner") {
            panel.w = resize.w + event.clientX - resize.startX;
          }
          if (resize.edge === "bottom" || resize.edge === "corner") {
            panel.h = resize.h + event.clientY - resize.startY;
          }
          const priorEdge = panel.edge || "";
          panel.edge = wallContact(panel);
          if (panel.edge && panel.edge !== priorEdge) markImpact(el);
          applyPanel(name);
          noteProbe(name, "resize", { moving: true, pressure: 0.42 });
          queueSave();
        });
        const endResize = (event) => {
          if (!resize || resize.id !== event.pointerId) return;
          resize = null;
          const panel = state.panels[name];
          if (panel) panel.edge = wallContact(panel);
          resolveBlobCollisions(name);
          applyPanel(name);
          noteProbe(name, "resize-end");
          queueSave();
        };
        handle.addEventListener("pointerup", endResize);
        handle.addEventListener("pointercancel", endResize);
      }

      bindResizeHandle(el.querySelector(":scope > .cc-blob-resize-grip"), "corner");
      el.querySelectorAll(":scope > .cc-blob-resize-edge").forEach((handle) => {
        bindResizeHandle(handle, handle.dataset.edge || "corner");
      });

      const close = el.querySelector(":scope > .cc-blob-close");
      if (close && close.dataset.blobCloseBound !== "1") {
        close.dataset.blobCloseBound = "1";
        close.addEventListener("click", (event) => {
          if (!state) state = loadState();
          remember();
          const panel = state.panels[name];
          panel.enabled = false;
          panel.collapsed = false;
          panel.edge = "";
          apply();
          noteProbe(name, "close");
          event.preventDefault();
          event.stopPropagation();
        });
      }
    }

    if (el.dataset.blobBound === "1") return;
    el.dataset.blobBound = "1";

    let drag = null;
    el.addEventListener("pointerdown", () => {
      if (!state) state = loadState();
      if (state.panels[name]?.enabled) raisePanel(name);
    }, true);

    el.addEventListener("pointerdown", (event) => {
      if (!state) state = loadState();
      const panel = state.panels[name];
      if (!panel.enabled) return;
      if (event.target.closest?.(".cc-blob-resize-grip,.cc-blob-resize-edge,.cc-blob-close")) return;
      const rect = el.getBoundingClientRect();
      const headerHit = event.clientY - rect.top <= 48;
      if (!headerHit) return;
      remember();
      drag = {
        id: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        panelX: panel.x,
        panelY: panel.y,
        edge: panel.edge || ""
      };
      el.classList.add("cc-blob-dragging");
      el.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    });

    el.addEventListener("pointermove", (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      const panel = state.panels[name];
      panel.x = drag.panelX + event.clientX - drag.startX;
      panel.y = drag.panelY + event.clientY - drag.startY;
      const priorEdge = panel.edge || "";
      panel.edge = wallContact(panel);
      if (panel.edge && panel.edge !== priorEdge) markImpact(el);
      resolveBlobCollisions(name);
      applyPanel(name);
      noteProbe(name, "drag", { moving: true, pressure: 0.38 });
      queueSave();
    });

    function endDrag(event) {
      if (!drag || drag.id !== event.pointerId) return;
      drag = null;
      el.classList.remove("cc-blob-dragging");
      const panel = state.panels[name];
      if (panel) panel.edge = wallContact(panel);
      resolveBlobCollisions(name);
      applyPanel(name);
      noteProbe(name, "drag-end");
      queueSave();
    }
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);

    el.addEventListener("click", (event) => {
      if (!state) state = loadState();
      const panel = state.panels[name];
      if (panel?.enabled) raisePanel(name);
      if (!panel.enabled || !panel.collapsed) return;
      const rect = el.getBoundingClientRect();
      if (event.clientY - rect.top > 58) return;
      remember();
      panel.collapsed = false;
      apply();
      noteProbe(name, "expand");
    });

    el.addEventListener("dblclick", (event) => {
      if (!state) state = loadState();
      const panel = state.panels[name];
      if (!panel.enabled) return;
      const rect = el.getBoundingClientRect();
      if (event.clientY - rect.top > 58) return;
      remember();
      panel.collapsed = !panel.collapsed;
      apply();
      noteProbe(name, panel.collapsed ? "collapse-one" : "expand-one");
    });
  }

  function bindResizeObserver() {
    if (!("ResizeObserver" in window) || window.__continuumBlobResizeObserver) return;
    const observer = new ResizeObserver((entries) => {
      if (!state) return;
      for (const entry of entries) {
        const el = entry.target;
        const name = Object.keys(PANEL_META).find((key) => panelElement(key) === el);
        if (!name) continue;
        const panel = state.panels[name];
        if (!panel.enabled || panel.collapsed || el.classList.contains("cc-blob-dragging")) continue;
        const rect = el.getBoundingClientRect();
        panel.w = Math.round(rect.width);
        panel.h = Math.round(rect.height);
        queueSave();
      }
    });
    window.__continuumBlobResizeObserver = observer;
  }

  function observePanel(name) {
    const observer = window.__continuumBlobResizeObserver;
    const el = panelElement(name);
    if (observer && el && el.dataset.blobObserved !== "1") {
      el.dataset.blobObserved = "1";
      observer.observe(el);
    }
  }

  function init() {
    if (!state) state = loadState();
    bindResizeObserver();
    bindPanel("chat");
    bindPanel("support");
    observePanel("chat");
    observePanel("support");
    apply();
    if (!didProbeInit) {
      didProbeInit = true;
      noteProbe("", "init");
    }
  }

  function isTypingTarget(target) {
    const tag = String(target?.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || Boolean(target?.isContentEditable);
  }

  window.continuumToggleBlobChat = () => togglePanel("chat");
  window.continuumToggleBlobSurface = () => togglePanel("support");
  window.continuumToggleBlobMeld = toggleMeld;
  window.continuumCollapseBlobPanels = collapseEnabled;
  window.continuumResetBlobPanels = resetPanels;
  window.continuumReduceBlobState = reduceBlobState;
  window.continuumUndoBlob = undoBlob;
  window.continuumBlobProbeState = cloneProbe;
  window.continuumBlobProbeNote = noteProbe;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
  if (!window.__continuumBlobKeysBound) {
    window.__continuumBlobKeysBound = true;
    document.addEventListener("keydown", (event) => {
      if (!state) state = loadState();
      if (event.key === "Escape" && anyEnabled()) {
        reduceBlobState();
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === "z" && anyEnabled() && !isTypingTarget(event.target)) {
        undoBlob();
        event.preventDefault();
        event.stopPropagation();
      }
    }, true);
  }
  new MutationObserver(init).observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("resize", () => { if (state) apply(); });
  if (!window.__continuumBlobProbeTicker) {
    window.__continuumBlobProbeTicker = window.setInterval(() => {
      if (!state || !anyEnabled()) return;
      for (const name of Object.keys(PANEL_META)) {
        if (state.panels[name]?.enabled) applyPanel(name);
      }
      queueProbeSave();
    }, 900);
  }
})();
</script>
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
    background_media = _background_media_html()
    if background_media:
        gr.HTML(background_media)
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
                    hf_login = gr.LoginButton(value="Sign in with Hugging Face", size="sm", scale=1)
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

    with gr.Accordion("App Settings", open=False, elem_id="app-settings-panel"):
        app_settings_summary = gr.Markdown(value=runtime_settings_markdown(), elem_id="app-settings-summary")
        with gr.Row():
            app_settings_refresh = gr.Button("Refresh Settings", size="sm", variant="secondary", scale=1)
            hf_pack_launch = gr.Button("Launch HF Provider Pack", size="sm", variant="primary", scale=1)
        hf_pack_include_unverified = gr.Checkbox(
            label="Also attempt unverified catalog chat providers",
            value=False,
            interactive=True,
        )
        hf_pack_status = gr.Markdown(
            "Starts the local free-credit scout pack through `start_all_hf_provider_daemons.bat`/`launch_hf_provider_pack.py`. "
            "Credits are used only when a daemon later answers an assigned turn.",
            elem_id="hf-provider-pack-status",
        )

    with gr.Row(elem_id="main-row"):
        # Left: Chat
        with gr.Column(scale=3, elem_id="chat-col"):
            chatbot = gr.Chatbot(height=650, label=CHATBOT_LABEL, elem_id="continuum-chat")
            intent_mode = gr.Radio(
                choices=INTENT_MODE_CHOICES,
                value="Auto",
                label="Intent Mode",
                elem_id="intent-mode",
                interactive=True,
            )
            with gr.Row(elem_id="blob-controls-row"):
                blob_chat_btn = gr.Button("Blob Chat", size="sm", variant="secondary")
                blob_surface_btn = gr.Button("Blob Surface", size="sm", variant="secondary")
                blob_wallpaper_btn = gr.Button("Wallpaper Blob", size="sm", variant="secondary")
                blob_meld_btn = gr.Button("Meld", size="sm", variant="secondary")
                blob_collapse_btn = gr.Button("Collapse", size="sm", variant="secondary")
                blob_reset_btn = gr.Button("Reset Blobs", size="sm", variant="secondary")
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
                with gr.TabItem("Native Tools"):
                    native_initial_status, native_initial_rows = native_toolkit_surface()
                    native_refresh = gr.Button("Refresh Native Tools", size="sm")
                    native_status = gr.Markdown(native_initial_status)
                    gr.Markdown("**Wallpaper controls**")
                    with gr.Row(elem_id="wallpaper-control-row"):
                        wallpaper_text = gr.Textbox(
                            label="Speech-rain text",
                            placeholder="Text to send into the expressive wallpaper...",
                            lines=2,
                            scale=4,
                        )
                        wallpaper_send = gr.Button("Send Wallpaper Text", scale=1, variant="secondary")
                    with gr.Row(elem_id="wallpaper-settings-row"):
                        wallpaper_settings = gr.Textbox(
                            label="Settings JSON",
                            placeholder='{"fontSize":24,"colorPreset":"aurora","direction":"toward","density":80,"settingsPanel":"minimize"}',
                            lines=2,
                            scale=4,
                        )
                        wallpaper_command = gr.Textbox(
                            label="Command",
                            placeholder="settings_minimize, settings_open, audio_on, audio_off, chaos_once...",
                            lines=2,
                            scale=2,
                        )
                        wallpaper_control_send = gr.Button("Send Wallpaper Control", scale=1, variant="primary")
                    wallpaper_probe = gr.Button("Probe Wallpaper Bridge", size="sm", variant="secondary")
                    wallpaper_status = gr.Markdown("Use this for direct operator text/settings. Tool-less agents can use `[[tool: native.continuum_wallpaper_text | text=...]]`, `[[tool: native.continuum_wallpaper_control | settings_json={...}]]`, or `[[tool: native.continuum_wallpaper_preset | preset=audio]]`; indexed MCP tools are optional.")
                    native_table = gr.Dataframe(
                        headers=["Category", "Tool", "Args", "Description", "Relay Command"],
                        datatype=["str", "str", "str", "str", "str"],
                        column_count=(5, "fixed"),
                        wrap=True,
                        interactive=False,
                        value=native_initial_rows,
                        elem_id="native-tool-table",
                    )
                    with gr.Row(elem_id="native-relay-row"):
                        native_relay_box = gr.Textbox(label="Native relay command", scale=4, interactive=True)
                        native_paste_btn = gr.Button("Paste to chat ▸", scale=1)
                with gr.TabItem("Last Track"):
                    track_initial_summary, _track_initial_audio, track_initial_receipt = latest_music_track()
                    track_refresh = gr.Button("Refresh Last Track", size="sm")
                    track_summary = gr.Markdown(value=track_initial_summary)
                    track_audio = gr.Audio(value=None, label="Latest Music Forge audio", type="filepath")
                    track_receipt = gr.JSON(value=track_initial_receipt, label="Track receipt")
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
    wallpaper_seen = gr.State("")
    wallpaper_event_payload = gr.Textbox(value="", visible=False, elem_id="wallpaper-event-payload")
    wallpaper_timer = gr.Timer(2.0)

    # Wiring
    send_flow = send.click(chat, [box, chatbot, model_dd, intent_mode, mcp_url, session], [chatbot, session, box, timeline_plot, sankey_plot, stage])
    send_flow.then(load_trace_table, [session], [mem_table, mem_status]).then(
        None,
        chatbot,
        None,
        js="(history) => { window.continuumSpeechRainFromHistory?.(history); }",
    )
    submit_flow = box.submit(chat, [box, chatbot, model_dd, intent_mode, mcp_url, session], [chatbot, session, box, timeline_plot, sankey_plot, stage])
    submit_flow.then(load_trace_table, [session], [mem_table, mem_status]).then(
        None,
        chatbot,
        None,
        js="(history) => { window.continuumSpeechRainFromHistory?.(history); }",
    )
    clear.click(reset, None, [chatbot, session, box, timeline_plot, sankey_plot, stage])
    connect_flow = connect_btn.click(
        connect_and_explore,
        [mcp_url, model_dd, chatbot, session],
        [mcp_connect_status, tool_status, tool_table, session, chatbot, box, timeline_plot, sankey_plot],
    )
    connect_flow.then(
        None,
        chatbot,
        None,
        js="(history) => { window.continuumSpeechRainFromHistory?.(history); }",
    )
    tool_search.change(browse_tools, [tool_search, session], [tool_status, tool_table])
    tool_table.select(on_select_tool, [tool_table], [relay_box])
    paste_btn.click(_paste_to_chat, [relay_box, box], [box])
    native_refresh.click(native_toolkit_surface, None, [native_status, native_table])
    native_table.select(on_select_tool, [native_table], [native_relay_box])
    native_paste_btn.click(_paste_to_chat, [native_relay_box, box], [box])
    wallpaper_send.click(
        None,
        [wallpaper_text],
        [wallpaper_status],
        js="""(text) => {
          const clean = String(text || "").trim();
          if (!clean) return "Enter text first.";
          const ok = window.continuumWallpaperCommand?.({ text: clean, source: "operator-control" });
          return ok ? "Queued/sent to expressive wallpaper. Use Probe Wallpaper Bridge for render receipt." : "Wallpaper bridge unavailable on this render.";
        }""",
    )
    wallpaper_control_send.click(
        None,
        [wallpaper_text, wallpaper_settings, wallpaper_command],
        [wallpaper_status],
        js="""(text, settingsText, command) => {
          const clean = String(text || "").trim();
          const rawSettings = String(settingsText || "").trim();
          const cleanCommand = String(command || "").trim();
          let settings = {};
          if (rawSettings) {
            try {
              settings = JSON.parse(rawSettings);
            } catch (err) {
              return "Settings JSON is not valid: " + err.message;
            }
            if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
              return "Settings JSON must be an object.";
            }
          }
          if (!clean && !cleanCommand && !Object.keys(settings).length) return "Enter text, settings JSON, or a command first.";
          const ok = window.continuumWallpaperCommand?.({
            text: clean,
            settings,
            settings_json: rawSettings,
            command: cleanCommand,
            source: "operator-control",
            slot: "wallpaper"
          });
          return ok ? "Queued/sent wallpaper control. Use Probe Wallpaper Bridge for render receipt." : "Wallpaper bridge unavailable on this render.";
        }""",
    )
    wallpaper_probe.click(
        None,
        None,
        [wallpaper_status],
        js="""() => {
          const probe = window.continuumWallpaperBridgeProbe?.();
          if (!probe) return "Wallpaper bridge probe is unavailable on this render.";
          const ok = probe.shell_present && probe.iframe_present && probe.iframe_window;
          const receipt = probe.last_receipt || {};
          return [
            ok ? "**Wallpaper bridge:** iframe reachable" : "**Wallpaper bridge:** iframe not reachable",
            "",
            "- Shell present: `" + probe.shell_present + "`",
            "- Iframe present: `" + probe.iframe_present + "`",
            "- Active underlay class: `" + probe.active_class + "`",
            "- Parent state: `" + (probe.parent_state || "none") + "`",
            "- Pending messages: `" + probe.pending_messages + "`",
            "- Last receipt: `" + (receipt.state || "none") + "`",
            "- Iframe pattern: `" + (probe.iframe?.config?.pattern || "unknown") + "`",
            "- Iframe direction: `" + (probe.iframe?.config?.direction || "unknown") + "`",
            "- Iframe font size: `" + (probe.iframe?.config?.fontSize || "unknown") + "`",
            "- Iframe primary color: `" + (probe.iframe?.config?.primaryColor || "unknown") + "`",
            "",
            "```json",
            JSON.stringify(probe, null, 2).slice(0, 2800),
            "```"
          ].join("\\n");
        }""",
    )
    wallpaper_event_flow = wallpaper_timer.tick(_latest_wallpaper_event, [wallpaper_seen], [wallpaper_event_payload, wallpaper_seen])
    wallpaper_event_flow.then(
        None,
        [wallpaper_event_payload],
        None,
        js="""(payload) => {
          if (!payload) return;
          try {
            const data = JSON.parse(payload);
            if (data) window.continuumWallpaperCommand?.(data);
          } catch (err) {
            console.warn("Continuum wallpaper event dispatch failed", err);
          }
        }""",
    )
    wallpaper_event_payload.change(
        None,
        [wallpaper_event_payload],
        None,
        js="""(payload) => {
          if (!payload) return;
          try {
            const data = JSON.parse(payload);
            if (data) window.continuumWallpaperCommand?.(data);
          } catch {}
        }""",
    )
    peer_save.click(
        save_peer_links,
        [peer_link_1, peer_link_2, peer_link_3, peer_link_4, peer_link_5, session],
        [peer_link_status, session, app_settings_summary, tool_status, tool_table],
    )
    app_settings_refresh.click(runtime_settings_markdown, None, app_settings_summary)
    hf_pack_launch.click(launch_hf_provider_daemon_pack, [hf_pack_include_unverified], [hf_pack_status]).then(
        runtime_settings_markdown,
        None,
        app_settings_summary,
    )
    blob_chat_btn.click(None, None, None, js="() => { window.continuumToggleBlobChat?.(); }")
    blob_surface_btn.click(None, None, None, js="() => { window.continuumToggleBlobSurface?.(); }")
    blob_wallpaper_btn.click(None, None, None, js="() => { window.continuumToggleWallpaperBlob?.(); }")
    blob_meld_btn.click(None, None, None, js="() => { window.continuumToggleBlobMeld?.(); }")
    blob_collapse_btn.click(None, None, None, js="() => { window.continuumCollapseBlobPanels?.(); }")
    blob_reset_btn.click(None, None, None, js="() => { window.continuumResetBlobPanels?.(); }")
    track_refresh.click(latest_music_track, None, [track_summary, track_audio, track_receipt])
    mem_refresh.click(load_trace_table, [session], [mem_table, mem_status])
    mem_table.select(inspect_node, [session], [mem_detail])
    graph_size.change(set_graph_height, [graph_size, session], [timeline_plot, sankey_plot])
    graph_opacity.change(
        None, graph_opacity, None,
        js="(v) => { document.querySelectorAll('#timeline-plot, #sankey-plot').forEach(e => e.style.opacity = v); }",
    )


if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CUSTOM_CSS + (_CLI_SCROLL_CSS if CLI_BRAIN else ""),
        head=CONTINUUM_SETTINGS_HEAD + CONTINUUM_WALLPAPER_HEAD + CONTINUUM_BLOB_OVERLAY_HEAD,
        allowed_paths=_launch_allowed_paths(),
    )
