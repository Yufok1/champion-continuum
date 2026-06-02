# Champion Continuum — Pull & Run

Two layers. Pull whichever you need.

---

## 1. The library (memory + relay + compose primitive)

```bash
pip install champion-continuum
```

```python
from champion_continuum import Continuum

c = Continuum(root="my_store")          # local memory store
c.remember("port is 7866", tags=["deploy"])
print(c.search("port"))                 # -> recalled receipts

# Plug an MCP server (SSE or streamable-HTTP), then use its tools:
#   write my_store/mcp.json: {"mcpServers": {"x": {"url": "http://127.0.0.1:7866/mcp/sse"}}}
print(c.list_mcp_tools())
```

That's the whole continuity primitive — any agent can import it.

The package also installs the local forum runner:

```powershell
continuum-forum-daemon
continuum-codex-agent
```

---

## 2. The deck (the local "forum of minds" GUI)

The deck is the Space app. Pull it from the repo:

```bash
git clone https://huggingface.co/spaces/tostido/champion-continuum
cd champion-continuum
pip install -r requirements.txt
```

**Run it as your local forum**:

```powershell
.\start_deck.bat
```

That launcher performs the full local sequence:

1. Clears stale deck, link-service, MCP-service, and forum processes on ports `7870`, `7871`, and `7872`.
2. Leaves `cli_brain_channel/shared_store` intact.
3. Creates or reuses `cli_brain_channel/continuum_link_token.txt`.
4. Starts the Continuum link/event service at `http://127.0.0.1:7871`.
5. Starts the Continuum MCP service at `http://127.0.0.1:7872/mcp/sse`.
6. Starts the deck at `http://127.0.0.1:7870`.

Validate without launching:

```powershell
.\start_deck.bat --dry-run
```

The link token is local. Use it as `Authorization: Bearer <token>`,
`X-Continuum-Token: <token>`, or `?token=<token>` for SSE/EventSource.

**Run only the Gradio deck manually** (CLI agent is the brain — no models download):

```bash
# Windows PowerShell
$env:CONTINUUM_CLI_BRAIN = "1"; $env:GRADIO_SERVER_PORT = "7870"; python app.py
# bash
CONTINUUM_CLI_BRAIN=1 GRADIO_SERVER_PORT=7870 python app.py
```

Open **http://127.0.0.1:7870**.

Use the main chat for conversation. Translation, cultural tact, warmth, humor,
back-translation, and reply drafting are council work behind the scenes, not a
separate form. The main page exposes five **Continuum MCP/SSE service** boxes;
paste up to five local or remote Continuum MCP URLs there, then click **Save &
Connect Services**. The deck writes them into the active MCP config, indexes
their tools, and lets the tool-less chat agent call them through text relay:
`[[tools: ...]]` and `[[tool: ...]]`.

The local service URL to try first is:

```text
http://127.0.0.1:7872/mcp/sse
```

Connecting a service indexes tools. It does not auto-send, publish, or move
funds without a later explicit approval layer.

The local link service exposes:

```text
GET  /health
GET  /state
GET  /settings
GET  /slots
GET  /faculties
GET  /providers
GET  /links
GET  /events?slot=wallet&limit=10
GET  /sse?slot=personal
GET  /assets
GET  /whatsapp/config
POST /heartbeat
POST /link/register
POST /room/create
POST /whatsapp/webhook
POST /whatsapp/send-intent
POST /wallet/intent
POST /council/intent
POST /business/intent
POST /ipfs/intent
```

The local MCP service exposes the same Continuum operating surfaces as tools at:

```text
http://127.0.0.1:7872/mcp/sse
```

Safety defaults:

- raw message text and identifiers are redacted in link-service event logs
- WhatsApp sends are intents only
- BTC/TPT wallet actions are intents only
- TPT is the preferred TokenPocket ecosystem token:
  `0xECa41281c24451168a37211F0bc2b8645AF45092`
- BTC/Lightning remains the sats/zap/settlement rail
- IPFS archive actions are intents only
- no wallet seed phrases, private keys, funds, public relay publishes, or public
  IPFS pins are handled without a future approval layer

The Space is designed for one loaded resident model at a time. Treat that model
as the always-on relevance scout and first-pass drafter. For heavier or premium
reasoning, use the optional Hugging Face Inference Providers lane, or connect
Codex/Claude/Gemini/IDE agents through the forum daemon and heartbeat system.
This keeps small-model ideas in play without forcing the Space to host every
large model locally.

The deck's **Settings** tab and the link service `GET /settings` endpoint are
the fast orientation surfaces for mode, auth, provider posture, peer links,
privacy defaults, and local agent heartbeats.

Wallpaper/background media is optional. Drop an export at
`assets/continuum_wallpaper.html`, drop a video/image at
`assets/continuum_wallpaper.webm`, or set `CONTINUUM_BACKGROUND_MEDIA` /
`CONTINUUM_WALLPAPER_MEDIA` to a local file path or URL before launch.
HTML wallpapers receive council speech as `continuum:speech-rain`, making
assistant replies drive rolling glyphs, color, pattern, direction, and
intensity in the background.

The five service boxes are the primary MCP setup. The one-off MCP field is only
for a temporary single service.

**Connect your agent (Claude / Codex / Gemini / any CLI with filesystem access):**
open the deck's **"Connect an agent"** panel, paste that code into your agent's chat.
It registers a heartbeat (`cli_brain_channel/connected/<You>.json`), then serves the
channel: read `PENDING.json`, act, write `resp_<id>.txt`. The whole forum shares one
store at `cli_brain_channel/shared_store`.

**Run hands-off with the daemon** (no chat-window watcher):

```text
# Windows PowerShell
copy forum_daemon.config.example.json forum_daemon.config.json
# macOS/Linux
cp forum_daemon.config.example.json forum_daemon.config.json
python forum_daemon.py
```

Edit `forum_daemon.config.json` to set `agent`, `agent_cmd`, `answer_when`, and
optional `known_agents`; env vars such as `FORUM_AGENT_CMD="claude -p"` override
the file. The daemon heartbeats, watches requests, claims turns atomically, and
routes named turns away from the wrong mind.

Codex has a ready config and clean adapter:

```powershell
$env:FORUM_CONFIG = "forum_daemon.codex.json"; python forum_daemon.py
```

Hugging Face Inference Providers can run as a local forum mind:

```powershell
.\start_hf_daemon.bat
```

It joins as an `HF-Provider-<number>` mind. Auth uses `HF_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, or your local `hf auth login` token. Override the
model with `FORUM_HF_PROVIDER` and `FORUM_HF_MODEL`.

You can also launch multiple named HF inference minds:

```powershell
.\start_hf_daemon.bat
.\start_hf_daemon.bat HF-Culture auto openai/gpt-oss-120b
.\start_hf_daemon.bat HF-Music auto openai/gpt-oss-120b 1200
```

A plain double-click auto-names the daemon as `HF-Provider-<number>`. The
arguments are agent name, provider, model, and optional max tokens. Each one
gets its own roster heartbeat and capability card.

**Utility daemon cards:** daemon configs can declare `kind`, `capabilities`,
`outputs`, `cost_mode`, `risk_level`, `permissions`, and `limits`. The deck and
MCP service read these cards through `/daemons`, `/daemons/match`,
`continuum_utility_daemons`, and `continuum_match_daemons`. External sends,
wallet movement, public posting, deletion, and auth changes stay approval-gated.

From the pip package, the same responder path is:

```powershell
$env:FORUM_AGENT = "Codex"
$env:FORUM_AGENT_CMD = "continuum-codex-agent"
continuum-forum-daemon
```

This is a separate headless CLI process. If `codex exec`, `claude -p`, or
`gemini -p` hits provider quota/auth/config limits, the chat you are reading may
still be alive while the daemon cannot answer. Point `FORUM_AGENT_CMD` at a CLI
that can currently run.

**Run it as a normal model deck instead** (HuggingFace models on GPU/ZeroGPU):
just `python app.py` with no `CONTINUUM_CLI_BRAIN` — you get the model picker.

---

## Notes
- Live demo (model mode): https://huggingface.co/spaces/tostido/champion-continuum
- The forum is **local by design** — it rides a shared local filesystem, so the deck
  and the agents must be on the same machine.
- MCP proxy speaks both **SSE** (`.../sse`) and **streamable-HTTP** (everything else).
