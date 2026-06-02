---
title: Champion Continuum
emoji: 🥸
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
hf_oauth: true
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
`requirements.txt` does not need additional packages for it.

## Main Chat, MCP Services, And Conversation Faculty

The main chat is the only human conversation input. Translation, cultural tact,
reply drafting, voice-message wording, and relationship tone are handled by the
agent/council behind the scenes instead of a separate worksheet tab.

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
  provider-call permission.
- CLI/IDE agents through the local forum daemon and link-service heartbeat.
- Local small-model faculties such as Whisper tiny, NLLB distilled, fastText
  language ID, or OPUS-MT as optional future adapters.

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

The deck includes a **Settings** tab that summarizes the current operating
mode, local link URL, token-file status, CLI-brain roster, HF provider posture,
translation faculty readiness, privacy defaults, and peer-link capacity.

The link service exposes the same operational posture at `GET /settings`. This
is read-only. It is for operator visibility and agent orientation, not hidden
configuration mutation.

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
