# Champion Continuum Universal Conversation Blueprint

## Product Shape

Champion Continuum is its own local conversation faculty. The operator launches
`start_deck.bat`, connects chosen AI agents inside the deck, selects premium
models or agency plans, and uses those minds to make cross-language conversation
feel fluent, warm, and natural.

This is not a translation worksheet. It is a chaired AI-assisted conversation
system.

## Core Promise

The operator can talk with friends, partners, collaborators, or customers who do
not share English as a first language, while both sides retain their own
language, tone, culture, and intent.

The originating use case is personal and concrete: an English speaker wants to
talk naturally with a Chinese speaker without becoming a language student before
every sentence. The system should smooth the edges, preserve affection and
respect, and make the operator feel present instead of buried in translation
work.

Continuum handles:

- source-language preservation
- target-language drafting
- idiom and cultural meaning
- relationship warmth
- back-translation for trust
- reply suggestions
- ambiguity flags
- model/council arbitration
- optional Continuum-to-Continuum sharing

The operator should not need to brainstorm the process before every message.
The default interaction is: paste or speak naturally, pick light preferences
only when useful, receive a message that can actually be sent.

## Lily Mode

The hardware inspiration is a "Lily translation system": the feeling of buying
one device for each person, putting it on, and talking normally. Continuum should
be the software version of that experience.

The social design matters:

- avoid making the other person feel interviewed, managed, or condescended to
- support shared-device, speaker-mode, earbud-mode, and text-only mode
- let both people see both languages when helpful
- keep corrections gentle and invisible by default
- make the AI council do the heavy language work behind the scenes

The target feeling is not "I used a translator." It is "we had a real
conversation."

## Operator Flow

1. Launch `start_deck.bat`.
2. Connect chosen AI agents or models inside the deck.
3. Pick a simple conversation profile:
   - source language
   - target language
   - relationship tone
   - optional expert/model preset
4. Paste or speak the raw message.
5. Continuum asks the agent team to produce:
   - message to send
   - literal back-translation
   - reply suggestion
   - short notes only when they matter
6. Operator cherry-picks, edits, or approves the send path.
7. Optional adapters carry the approved result into WhatsApp, voice, Nostr,
   wallet, or peer-Continuum slots.

## Arbitration Model

Continuum can use Expert-Assisted Parallel-Track Ingestion (EAPTI), a fixed-point
arbitration loop that preserves the raw message while letting selected agents
help certify the translated meaning.

1. Capture `raw_content` immediately for provenance.
2. Ask selected agents to draft or critique the translation from their lens.
3. Produce a `normalized_core` semantic baseline for shared reasoning.
4. Back-translate into the source language.
5. Compare raw intent, emotional force, and factual content.
6. Ask selected agents to dissent only where meaning drifts.
7. Redraft until stable or flag the unstable part.

The point is not to expose the whole process to the operator. The point is to
make the final message trustworthy and easy to use.

The packet shape should stay explicit:

```json
{
  "provenance": {
    "raw_content": "original message",
    "input_lang": "auto",
    "raw_sha256": "content hash"
  },
  "agent_arbitration": {
    "contributions": [],
    "dissent": [],
    "consensus_accuracy_score": 0.0
  },
  "execution_plane": {
    "normalized_core": "",
    "target_language_message": "",
    "literal_back_translation": "",
    "reply_suggestion": "",
    "consensus_lang": "en-US"
  }
}
```

## Agent Council

Agents are not decorations. They are selectable workers.

Useful roles:

- cultural/idiom lens
- warmth/relationship lens
- literal back-translation lens
- business/domain lens
- brevity/delivery lens
- safety/consent lens
- final editor

Example default assignment:

- Gemini-style cultural lens: idiom, tone, warmth, and social grace
- Codex-style accuracy lens: structure, exactness, business/crypto/domain terms
- Claude-style relationship lens: emotional readability and gentleness
- Kilo/other local lens: direct dissent when the translation sounds wrong

The operator can choose elite models, cheap models, local models, or mixed
agency plans. Continuum should make this feel like choosing a team, not wiring a
stack.

Cherry-picking is a feature. The operator may keep one agent's wording, another
agent's back-translation, and a third agent's caution note.

## Model And Provider Routing

The Hugging Face Space has a practical one-resident-model budget. That is a
feature, not a failure. The resident model should be treated as the always-on
relevance scout:

- observe the operator message
- detect likely language/task shape
- draft the first packet
- decide whether escalation is relevant
- preserve small-model ideas when they are sharp

Relevance wins over raw model size. A small model can contribute a decisive
observation; a premier model should be called because the packet needs depth,
not because bigger is automatically better.

Escalation lanes:

- Hugging Face Inference Providers for remote premium/open models through one
  provider router and an HF token.
- CLI/IDE agents such as Codex, Claude, Gemini, or local tools through the
  forum daemon and heartbeat lane.
- Optional local small models for language ID, speech-to-text, and translation
  baselines when runtime budget allows.

These lanes are not competing systems. They are roles in one Continuum packet
flow: resident scout, provider expert, CLI/IDE operator, and optional local
translation utility.

## Council Autonomy Model

The council is allowed to do useful work from chat directives, but every action
must have a typed lane and an explicit external-effect boundary.

Allowed without external approval:

- observe incoming messages and state
- draft translations and replies
- compare agent outputs
- create business or crypto action plans
- create wallet, zap, marketplace, WhatsApp, Nostr, or IPFS intents
- produce Cascade-lattice and Merkle receipt drafts
- flag risks, missing data, and contradictions

Requires explicit operator approval:

- sending a WhatsApp message
- publishing to a public relay
- moving BTC, TPT, sats, zaps, or any funds
- signing with TokenPocket or another wallet
- pinning private content to a public IPFS provider
- executing external compute or marketplace jobs

This makes the council operational rather than passive, while preserving a hard
line between "prepared by AI" and "externally executed by the operator."

## Continuum Backend Boundary

Continuum is the backend for this facility.

It may connect to Champion Council, MCP servers, model APIs, or other
Continuums, but none of those are the authority by default. They are optional
participants.

`start_deck.bat` is the operator-owned launch surface. It should start the local
deck and the local Continuum link service. The operator launches it; agents and
other Continuums connect into it.

## Continuum Link Protocol

The local Continuum should expose a simple SSE-based link service:

- `GET /sse?slot=personal` streams one local Continuum event slot.
- `GET /sse?slot=*` streams all local slots.
- `POST /event` accepts local or peer Continuum events.
- `GET /slots` lists active slots and event counts.
- `GET /state` reports connected agents, channel paths, and recent event ids.
- `GET /health` reports process health.

Events should be Nostr-like but Continuum-native:

- append-only JSON
- deterministic ids
- local-first
- unsigned draft allowed
- signing/relay publishing optional later
- no payment or external publish unless explicitly invoked

This creates the base for peer Continuums to exchange conversation packets,
receipts, and agent offers without requiring Champion Council as a middleman.

Near-term Nostr stance:

- verify existing Nostr/community machinery before copying any pattern
- do not hallucinate a marketplace or reputation layer into this package
- start with local SSE and deterministic event packets
- add Nostr relay publishing only after local packet shape and privacy rules are
  stable

## First External Channel: WhatsApp

WhatsApp is the first practical external messaging adapter.

The intended integration is the official WhatsApp Business Cloud API:

- inbound webhooks become `whatsapp` slot events
- text messages preserve raw content for EAPTI/council drafting
- audio/voice-note messages preserve media metadata and mark media fetch as
  required
- outbound text/audio responses are generated as send intents first
- no WhatsApp message is sent until the operator approves it
- the adapter does not automate consumer WhatsApp Web or a personal account

Local adapter endpoints:

- `GET /whatsapp/config` reports whether required env vars are configured
  without exposing secrets
- `GET /whatsapp/webhook` supports Meta verification with
  `WHATSAPP_VERIFY_TOKEN`
- `POST /whatsapp/webhook` converts inbound webhook payloads into local
  `whatsapp` slot events
- `POST /whatsapp/send-intent` creates a local outbound message intent without
  calling Meta

Expected secret inputs, owned by the operator:

- `WHATSAPP_ACCESS_TOKEN`
- `WHATSAPP_PHONE_NUMBER_ID`
- `WHATSAPP_VERIFY_TOKEN`
- optional `WHATSAPP_APP_SECRET` for webhook signature verification
- optional `WHATSAPP_GRAPH_VERSION`

## Wallet And Bitcoin Posture

Bitcoin belongs in Continuum as receipts, zaps, access signals, and payment
intents, not as custody.

The preferred wallet provider for this operator is TokenPocket. Continuum should
treat TokenPocket as an external signing/spending authority:

- Continuum never stores seed phrases, mnemonics, private keys, or raw wallet
  secrets
- Continuum can create wallet/payment/zap intent packets
- the operator approves and signs in the wallet
- `wallet` slot events must clearly say `funds_moved: false` unless a reviewed
  wallet adapter proves otherwise
- WalletConnect is the practical first route for TokenPocket-style signing
- NIP-57 zap receipts and NIP-47/Nostr Wallet Connect remain the right Nostr
  patterns when a Lightning wallet supports them

TPT posture:

- TPT contract address:
  `0xECa41281c24451168a37211F0bc2b8645AF45092`
- if any TokenPocket-native ecosystem token is used, it should be TPT
- TPT is useful for optional membership, gating, reputation, or loyalty signals
- TPT is not the default zap rail
- the normal value-transfer goal remains: recipients receive sats or clear
  transaction-fee/payment receipts

WhatsApp posture:

- WhatsApp has no native crypto token to build around here
- WhatsApp is the message and voice channel
- payment instructions, sats receipts, or wallet intents can be discussed in the
  conversation, but money movement belongs to wallet/zap adapters

## IPFS And Merkle Receipts

Continuum packets are hashable and can be made IPFS-compatible.

Recommended order:

1. Create local JSON receipt.
2. Hash it and include Merkle/Cascade-lattice metadata.
3. Keep private content local by default.
4. Optionally pin only redacted receipts or public artifacts to IPFS.
5. Use local IPFS pinning first when possible.
6. Use remote pinning providers only when uptime/availability justifies cost.

IPFS itself is not the expensive part. Persistence is. If the operator runs a
local IPFS node, local pinning can be free except for hardware, bandwidth, and
electricity. Paid pinning buys availability from machines that stay online.

## Privacy And Trust

Raw human messages are sacred input.

Default behavior:

- keep raw text local
- store hashes and receipts where possible
- do not publish externally
- do not move money
- do not claim a message was sent
- make ambiguity visible when it matters
- generalize or hash private relationship context in durable receipts unless
  the operator explicitly asks to store the full text

## Near-Term Implementation Slice

1. Maintain the Continuum-native SSE/link server and MCP service in this export.
2. Have `start_deck.bat` launch both services beside the Gradio deck.
3. Make the bridge chat-first:
   - main chat is the only human conversation input
   - translation, tone, humor, back-translation, and tact are inferred by the
     council/faculty from the conversation
   - five visible Continuum MCP/SSE service boxes sit on the main page and feed
     the text-relay tool surface
   - no required MCP or Champion Council guard
4. Keep room sessions and heartbeats on the existing link event log.
5. Expose Continuum services over MCP so tool-less agents can call tools through
   `[[tools: ...]]` and `[[tool: ...]]`.
6. Preserve the operator launch boundary: Codex edits source; the operator runs
   the deck.
7. Add local WhatsApp and wallet intent adapters before any real external send
   or funds movement.
