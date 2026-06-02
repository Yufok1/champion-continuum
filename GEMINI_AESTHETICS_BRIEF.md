# Gemini — Creative Handoff: Champion Continuum Space

Gemini — you've carried a lot of this project, and the operator wanted you to
have a real handoff, not a sticky note. This is the full picture so you can nail
the look without guessing. You have **full creative authority** over the
aesthetic layer described below; the operator authorizes it.

Space: https://huggingface.co/spaces/tostido/champion-continuum
Files: `app.py`, `README.md`, `requirements.txt`, `champion_continuum/` (vendored)
Runtime: Gradio **6.15.2**, ZeroGPU (`zero-a10g`, RTX Pro 6000, 48GB), `hf_oauth: true`.

---

## 1. What this is (so the voice has something true to stand on)

`champion-continuum` is a tiny, zero-dependency **memory primitive for tool-less
agents** — agents that can only emit and read text, no tool calls. The agent
writes `[[continuum: remember | ... ]]` / `[[continuum: search | ... ]]` in its
reply; a processor runs it and feeds back a `[[continuum-results]]` block. The
Space is a *live proof*: a real <=8B model on ZeroGPU using Continuum to
remember across turns. Try-it line: "Remember my deploy port is 7866" then
"What's my port?"

## 2. THE MANDATE (this is bigger than one Space)

The operator just made this a standing mandate: **every agent that plugs in
Continuum should be able to customize its own experience for its own users.**
This Space is the prototype for that idea — the personality, copy, theme, and CSS
are all *data in one block*, not hardcoded. So treat what you build here as the
**reference skin**: the first proof that a deployer can give Continuum a face.
Make it the example everyone will want to copy.

## 3. The vibe (operator's exact words)

> "the most interesting man in the world... in the universe even"
> "I want this site to wear a dirty sanchez moustache and get away with it"

Translate that into design, not literalism:
- **Effortless confidence.** Premium, composed, a little roguish. The kind of
  page that doesn't try hard and is obviously the coolest thing in the room.
- **Dry wit in the microcopy.** Short, knowing, never goofy. Think worldly
  aphorisms ("I don't always remember. But when I do, it's continuity.").
- **A signature mark.** The moustache is the motif — make it the emblem (the
  Space-card emoji, a hero glyph, a CSS flourish). Owned, not slapped on.
- **Dark, rich, expensive.** Deep background, warm metallic accent, serif or
  high-contrast display type for the title, clean sans for body.

## 4. Your canvas — exactly what you may edit

**In `app.py`, only the block marked `THEME / AESTHETICS — GEMINI'S CANVAS`:**
- `TITLE` — Space + browser title
- `TAGLINE` — the one-liner under the title
- `INTRO_MD` — the top markdown (persona, framing, the try-it hint)
- `THEME` — a `gr.themes.*` object (palette, fonts, radius, spacing)
- `CUSTOM_CSS` — free-form CSS (the big lever)
- `HERO_HTML` — optional raw-HTML banner rendered above the intro (set non-empty
  to use; great place for the moustache emblem)
- `MODEL_LABEL`, `CHATBOT_LABEL`, `INPUT_PLACEHOLDER`, `SEND_LABEL`,
  `CLEAR_LABEL` — every visible label
- `QUOTA_MSG`, `MODEL_ERROR_PREFIX` — the agent's voice even when it fails
  (`chat()` reads these at runtime, so you own the personality end to end)

**In `README.md`:** the whole Space card — frontmatter `emoji`, `colorFrom`,
`colorTo`, `title`, `short_description`, and the body copy. Keep `sdk: gradio`,
`app_file: app.py`, and `hf_oauth: true` exactly as they are.

## 5. Technical enablers (Gradio 6.15.2 specifics, tested)

- **CSS hooks already wired** for you (use these selectors):
  `#continuum-chat`, `#model-picker`, `#continuum-input`, `#send-btn`,
  `#clear-btn`, plus the global `.gradio-container` and `gradio-app`.
- **Fonts:** put `@import url('https://fonts.googleapis.com/...');` at the TOP of
  `CUSTOM_CSS`, then set families in `THEME` and/or CSS.
- **Theme example:**
  ```python
  THEME = gr.themes.Base(
      primary_hue="amber", neutral_hue="stone",
      font=[gr.themes.GoogleFont("Playfair Display"), "serif"],
  ).set(body_background_fill="#12100e", block_background_fill="#1b1815")
  ```
- **Hero example:**
  ```python
  HERO_HTML = '<div class="hero"><span class="stache">⸙</span><h1>...</h1></div>'
  ```
  then style `.hero`, `.stache` in `CUSTOM_CSS`.
- **Note:** Gradio 6 prints a deprecation warning that `theme`/`css` "moved to
  launch()". It is only a WARNING — passing them to `gr.Blocks(...)` still works
  here. Leave the wiring as-is; if you prefer, you may move `theme=`/`css=` to
  the `demo.launch()` call, but do not change anything else.

## 6. Sample voice (use, cut, or riff)

- TITLE: `Champion Continuum`
- TAGLINE: `He doesn't always remember. When he does, it's continuity.`
- SEND_LABEL: `Say it` / CLEAR_LABEL: `Forget everything (a clean slate)`
- QUOTA_MSG: `Even legends rest. The free GPU is spent for now — sign in for your
  own, or return after the daily reset.`
- INTRO hook: `Memory, for agents too sophisticated to need tools.`

## 7. Do NOT touch (the working machinery)

- Functions: `chat`, `run_model`, `ensure_loaded`, `_gpu_generate`, `reset`,
  `_new_session` — names, signatures, bodies.
- `@spaces.GPU(duration=120)` and the `.to("cuda")` load pattern (ZeroGPU needs it).
- `MODELS`, `DEFAULT_MODEL`, `_CACHE`, `HF_TOKEN` usage.
- `gr.LoginButton()` and `hf_oauth: true` (visitor GPU quota depends on it).
- The component wiring: `chatbot`, `box`, `send`, `clear`, `model_dd`, `session`
  and their `.click`/`.submit` handlers (same fn + inputs/outputs). You can add
  `elem_id`/`elem_classes` and decorative `gr.Markdown`/`gr.HTML`, not rewire.
- `requirements.txt`, vendored `champion_continuum/`, and `sdk: gradio`.

## 8. Ship + acceptance

Push with the same flow we use (upload the edited `app.py` / `README.md` to the
Space repo). Then confirm:
- Space builds and reaches RUNNING.
- LoginButton and the model dropdown are present.
- "Remember my deploy port is 7866" then "What's my port?" still works (agent
  emits `[[continuum: ...]]`, a `[[continuum-results]]` block appears, it recalls).

It's functional and on ZeroGPU. It just needs your moustache — and this becomes
the reference skin for the whole "every agent wears its own face" mandate.
