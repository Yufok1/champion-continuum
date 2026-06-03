---
title: Champion Continuum
emoji: 🥸
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
hf_oauth: true
hf_oauth_scopes:
  - inference-api
pinned: false
license: mit
short_description: He doesn't always remember. When he does, it's continuity.
---

# Champion Continuum — the Capability Proxy

A ZeroGPU proof of [`champion-continuum`](https://pypi.org/project/champion-continuum/),
the memory primitive for agents too sophisticated for tools.

## Local Launch

For local use, run the one-click launcher:

```powershell
.\start_deck.bat
```

Launch order:

1. Clean stale deck, link-service, MCP-service, and forum processes for ports `7870`, `7871`, and `7872`.
2. Leave `cli_brain_channel/shared_store` intact.
3. Create or reuse `cli_brain_channel/continuum_link_token.txt`.
4. Start the local Continuum link/event service on `http://127.0.0.1:7871`.
5. Start the local Continuum MCP service on `http://127.0.0.1:7872/mcp/sse`.
6. Start the Gradio deck on `http://127.0.0.1:7870`.

To validate the order without starting servers:

```powershell
.\start_deck.bat --dry-run
```

## Continuum Link Service

The link service is local-first and token-gated. Use the token from
`cli_brain_channel/continuum_link_token.txt` as either:

- `Authorization: Bearer <token>`
- `X-Continuum-Token: <token>`
- `?token=<token>` for browser `EventSource` streams

Core endpoints:

- `GET /health`
- `GET /state`
- `GET /settings`
- `GET /slots`
- `GET /faculties`
- `GET /providers`
- `GET /links`
- `GET /events?slot=wallet&limit=10`
- `GET /sse?slot=personal`
- `GET /sse?slot=*`

Adapter/intention endpoints:

- `POST /heartbeat`
- `POST /link/register`
- `POST /room/create`
- `GET /assets`
- `GET /whatsapp/config`
- `GET /whatsapp/webhook` for Meta verification
- `POST /whatsapp/webhook`
- `POST /whatsapp/send-intent`
- `POST /wallet/intent`
- `POST /council/intent`
- `POST /business/intent`
- `POST /ipfs/intent`

By default, raw message text and identifiers are redacted in link-service event
logs. External effects stay draft-only: no WhatsApp send, relay publish, wallet
payment, IPFS pin, or compute execution happens until an explicit approval layer
is added.

## WhatsApp, Wallet, And IPFS Posture

- WhatsApp is the first external conversation channel. It has no native crypto
  token in this system.
- BTC/Lightning is the neutral sats/zap/settlement rail.
- TPT is the preferred TokenPocket ecosystem token if a wallet-native token is
  used: `0xECa41281c24451168a37211F0bc2b8645AF45092`.
- TokenPocket remains an external signing/spending authority. Continuum does
  not store wallet seed phrases, private keys, or custody funds.
- IPFS is optional receipt/archive infrastructure. Local pinning is possible;
  paid pinning is only for persistence and availability.

The new link service uses only the Python standard library, so
`requirements.txt` includes `gradio_client` so the local MCP service can call
public Hugging Face Spaces when the operator chooses that route.

## Main Chat, MCP Services, And Conversation Faculty

The main chat is the only human conversation input. Translation, cultural tact,
reply drafting, voice-message wording, and relationship tone are handled by the
agent/council behind the scenes instead of a separate worksheet tab.

The **Intent Mode** selector beside the composer lets the operator override the
next turn's routing without leaving chat. `Auto` infers the route from the
message. `Translation Bridge`, `Music Forge`, `Resource Audit`, and
`Expressive Wallpaper` make the selected faculty explicit to the council. Tool
backed modes first use indexed MCP tools when available, then Continuum-native
fallbacks where they exist. The selector sharpens routing; it does not pretend a
missing external backend, provider call, or music-generation service is live.

The support rail has two tool views. **Tool Surface** shows whatever MCP/SSE
services are currently indexed. **Native Tools** is the Continuum-native catalog:
music, translation, daemons, peer links, wallpaper, memory, and intent tools with
ready relay commands. Indexed rows call the MCP sidecar. If the sidecar is not
available, rows fall back to direct `native.*` relay commands handled in process,
so tool-less agents still get useful results instead of a zero-tool dead end. On
the local desktop launcher, Native Tools will also try to self-index the local
MCP sidecar at `http://127.0.0.1:7872/mcp/sse` when the active tool surface is
empty; use **Refresh Native Tools** if the sidecar was still starting when the
page loaded.

Native fallback is not just a label. It can read the in-process provider
registry, utility-daemon cards, Music Forge readiness, Music Forge backend
preset payloads, wallpaper text/settings bridge, memory, events, links, and
intent drafts. Real external actions still stay honest: HF Space schema
inspection, provider calls, music generation, external sends, and funds movement
require the corresponding backend/token/operator approval.

The main page exposes five **Continuum MCP/SSE service** boxes. Paste local or
remote Continuum MCP URLs there, for example `http://127.0.0.1:7872/mcp/sse`,
then click **Save & Connect Services**. The deck writes those services into the
active MCP config, indexes their tools, and teaches the tool-less chat agent to
call them through `[[tools: ...]]` and `[[tool: ...]]`.

The same records are also saved into
`cli_brain_channel/continuum_peer_links.json` for visibility and room/link
metadata. Connecting a service indexes tools; it still does not send a WhatsApp
message, move funds, publish a relay event, or pin IPFS without an explicit
approval layer.

The Space should keep one resident model loaded. That resident model is not a
toy; it is the always-on relevance scout for routing, first drafts, cheap
observation, and small-but-useful ideas. Relevance wins over raw model size.

Heavier help comes through separate lanes:

- Hugging Face Inference Providers via the optional `HF Inference Providers -
  auto router` model menu entry. This requires an `HF_TOKEN`/Hub token with
  provider-call permission, or a local Hugging Face CLI login. For hands-off
  forum use, run `start_hf_daemon.bat`; it joins as an `HF-Provider-<number>`
  mind and answers Bear Claw turns through the same shared channel as
  Codex/Gemini.
  Multiple provider daemons can run at once by giving each one a different
  forum name:

  ```powershell
  .\start_hf_daemon.bat
  .\start_hf_daemon.bat HF-Culture auto openai/gpt-oss-120b
  .\start_hf_daemon.bat HF-Music auto openai/gpt-oss-120b 1200
  ```

  A plain double-click auto-names the daemon as `HF-Provider-<number>` so
  repeated launches do not overwrite the same roster heartbeat. Passing the
  first argument gives it a stable role name. Each daemon writes its own
  heartbeat and capability card, so the council can route work by agent name
  without collapsing them into one provider.

  The provider registry includes the current Hugging Face Inference Providers
  catalog as a local read surface. Tool-less agents can ask for it with
  `[[tool: native.continuum_provider_catalog | ]]` or
  `[[tool: continuum_1.continuum_provider_catalog | ]]` when the MCP sidecar is
  indexed. The catalog tracks HF's integrated provider roster, routing policies
  (`auto`, `:fastest`, `:cheapest`, `:preferred`, explicit provider suffixes),
  starter model selectors, and the free-credit boundary. It is intentionally
  honest: HF routed requests may use monthly credits where eligible, but that is
  not unlimited free compute and live pricing/model availability should be
  refreshed before large batch runs.
- Music Forge through the local MCP/SSE endpoint at
  `http://127.0.0.1:7872/mcp/sse`. The tools
  `continuum_music_forge_state`, `continuum_music_compose_packet`,
  `continuum_music_generate_preset`, `continuum_music_hf_space_schema`, and
  `continuum_music_generate_hf_space` let council agents use normal chat to
  compose a song request, call a public music Space, and save returned audio files under
  `cli_brain_channel/music_outputs`.
- CLI/IDE agents through the local forum daemon and link-service heartbeat.
- Local small-model faculties such as Whisper tiny, NLLB distilled, fastText
  language ID, or OPUS-MT as optional future adapters.

### Utility Daemons

Forum daemons are treated as small allocatable workers. Each heartbeat can carry
a capability card with `kind`, `capabilities`, `outputs`, `cost_mode`,
`risk_level`, permissions, and bounded loop limits. The local link service
exposes:

- `GET /daemons`
- `GET /daemons/match?capability=translation&output=text`

The Continuum MCP service exposes the same posture through
`continuum_utility_daemons` and `continuum_match_daemons`. Daemons may draft,
critique, generate files, or verify work, but sends, wallet movement, public
publishing, deletes, and auth changes remain operator-approval gates.

The hidden conversational bridge packet machinery can carry:

- literal back-translation
- contact/conversation profile
- glossary/fixed-term hits
- local faculty readiness metadata
- room-session intent metadata

Nothing about this replaces MCP, the forum, or the resident model. The faculty
registry is a routing map for when to use each lane.

## Rooms And Heartbeats

The link/event service exposes room, peer-link, and heartbeat primitives over
the existing SSE event log:

- `POST /room/create` creates a token-gated room slot and returns join paths
  such as `/sse?slot=room-...`.
- `GET /links` reports up to five registered Continuum service targets.
- `POST /link/register` stores or updates one peer target as metadata. Remote
  tokens are reduced to hashes by default; the service does not auto-dial peers
  yet.
- `POST /heartbeat` records liveness for the deck, Space, CLI/IDE agents,
  adapters, or rooms.

This is one local event server plus one local MCP server. The five visible boxes
can point at up to five Continuum MCP/SSE services so tool-less agents can use
those tools from the main chat. These are local Continuum coordination packets.
They do not publish public invites, send WhatsApp messages, move funds, call
Google APIs, or relay over Nostr.

## Settings And Readiness

The page includes an **App Settings** accordion with readable HF auth, provider,
endpoint, privacy, runtime, daemon, wallpaper, and five-slot readiness status.
The editable MCP/SSE service slots stay visible on the main page so the operator
can connect services without hunting through support tabs. The same Continuum
controls are also injected into the Gradio footer Settings menu.

### Wallpaper Layer

The deck can use an audio-reactive or animated wallpaper asset as a muted
background layer. Drop a web wallpaper at `assets/continuum_wallpaper.html`,
drop an exported video/image at `assets/continuum_wallpaper.webm`,
or set `CONTINUUM_BACKGROUND_MEDIA` / `CONTINUUM_WALLPAPER_MEDIA` to a local
file path or URL before launch. Supported defaults include `.html`, `.webm`,
`.mp4`, `.gif`, `.png`, `.jpg`, and `.jpeg`.

When the active wallpaper is an HTML/web wallpaper, the deck treats it as an
expressive renderer facility. Assistant and council replies are sent to the
embedded wallpaper as `continuum:speech-rain` messages, so the actual words can
roll through the background while color, speed, direction, pattern, density,
and intensity shift from the text itself. Agents can inspect this contract
through `continuum_expressive_wallpaper` and can queue explicit wallpaper text
with `continuum_wallpaper_text`.

The same bridge can also steer the Matrix Rain settings panel opened with `M`.
Use `continuum_wallpaper_control` for settings/audio/modal commands and
`continuum_wallpaper_preset` for named looks. On hosted Spaces or any run
without a sidecar, use the native forms:

```text
[[tool: native.continuum_wallpaper_text | text=HELLO]]
[[tool: native.continuum_wallpaper_control | text=GREETINGS | command=settings_minimize | settings_json={"fontSize":24,"colorPreset":"aurora","direction":"toward","density":80}]]
[[tool: native.continuum_wallpaper_preset | preset=audio | text=AUDIO REACTIVE COUNCIL]]
```

Pipe-separated arguments are the clearest form for tool-less agents. The relay
also tolerates loose `key=value key2=value2` arguments after JSON values, but
agents should prefer pipes when composing commands by text.

Wallpaper tool success means the command was queued for the browser bridge.
Do not treat that as visible truth until the deck applies it. Operators can
click **Probe Wallpaper Bridge** in the Native Tools tab to read the live iframe
receipt, current pattern, direction, font size, color, and pending-message
state. This keeps tool-less council agents honest: they should say "queued"
unless a receipt/probe confirms the render.

Useful `settings_json` keys include `fontSize`, `colorPreset`, `pattern`,
`direction`, `speed`, `intensity`, `density`, `characterSet`,
`customCharacters`, `audioReactive`, `audioReverse`, `audioDiagonals`,
`autoOrchestrator`, `reverseFlow`, `settingsPanel`, and `canvasOpacity`.
Valid Matrix Rain `pattern` values are `classic`, `rainbow`, `pentad`,
`chaos`, `harmonic`, and `particles`; common aliases such as `rain` or
`matrix` are normalized to `classic`.
Useful commands include `settings_open`, `settings_minimize`, `settings_close`,
`audio_on`, `audio_off`, `auto_on`, `auto_off`, `reverse_flow`, and
`chaos_once`. Operators can use the visible **Wallpaper controls** in the
Native Tools tab for direct text, settings JSON, and commands.

The wallpaper is part of the main page. It does not require a separate side
window. Click **Wallpaper Blob** to turn the Matrix Rain background into a
draggable, resizable, collapsible underlay viewport; click the blob `x` to
return it to the full-page underlay.

Facilities can be daemon-shaped without becoming uncontrolled background
processes. Use capability-card style wrappers such as `Wallpaper-Reactor`,
`Matrix-Rain`, or `Music-Forge` for utility daemons, keep their permissions
bounded, and require operator approval for external sends, publishing, funds,
auth changes, or destructive actions.

The link service exposes the same operational posture at `GET /settings` for
agent orientation. The five visible MCP/SSE service boxes are the operator
configuration path.

### MCP Tool Socket
This Space is now a **Universal MCP Proxy**:
1.  **Plug in MCP Services**: Paste up to five MCP/SSE service URLs into the main-page service boxes.
2.  **Dynamic Discovery**: The agent (Qwen2.5-1.5B) instantly learns the new tools.
3.  **Text-Relay Tools**: The agent uses the tools by writing `[[tool: server.name | args]]`.

The one-off MCP field is still available for a single temporary service, but
the five service boxes are the normal setup.

### Try it out
- **Memory**: "Remember my deploy port is 7866" then "What was my port?"
- **Conversation bridge**: Ask in the main chat for a warm message, translation,
  back-translation, or cross-cultural reply. Do not use a separate bridge tab.
- **Continuum services**: Paste up to five Continuum MCP/SSE URLs into the
  main-page service boxes, then ask the agent to search or execute tools.

### Design
The "Moustache" reference skin — dark, premium, and effortlessly confident.
- **Durable Memory** — surviving the context reset.
- **Weighted Search** — recall that knows what matters.
- **MCP Integration** — any tool, any model, purely via text.
