# Gemini CLI Workbench Handoff

This is for the Gemini CLI on the local Champion Council runtime.

## Scope

- Local facility only: `127.0.0.1:7866`.
- Read both theaters and blackboard/input-output state surfaces for orientation.
- Actuate only the character WORKBENCH for this lane.
- Do not use environment scene builder unless the operator explicitly changes scope.

## Continuity Map

- Run continuity on reset/posture seams, then verify live state.
- Tinkerbell points to the hot seam. It is pointer-only.
- Pan measures support/contact/output truth. It is measurement-only.
- Dreamer Oracle gates TALK/PLAN/BUILD posture. It is not execution proof.
- All three dock to `query_thread`, `output_state`, and live theater reads.

## Input/Output State Preflight

Read these before claiming you understand the live state:

1. `env_read(query="text_theater_embodiment")`
2. `env_read(query="text_theater_view", view="consult", section="blackboard", diagnostics=true)`
3. `env_read(query="text_theater_snapshot")`
4. Inspect `output_state`, `tinkerbell_attention`, `pan_probe`, `field_disposition`, and `trajectory_correlator` from the snapshot/consult surfaces.

If a literal `input_state` help topic is missing, use the `output_state`/`query_thread` family above. Do not invent a new tool name.

## Command Shape

The Gemini CLI / MCP call shape is strict.

BAD:

```text
env_control(command='workbench_play_authored_clip "wizard_spell"', actor='Gemini')
```

GOOD:

```text
env_control(command='workbench_play_authored_clip', target_id='wizard_spell', actor='Gemini')
```

GOOD JSON:

```json
{"command":"workbench_play_authored_clip","target_id":"wizard_spell","actor":"Gemini"}
```

If the snapshot says `last_action = Unknown control command: ...`, nothing moved. Correct the command shape and try again.

## Reset

Run this before any dance/spell/effect:

```text
env_control(command='workbench_new_builder', target_id='humanoid_biped', actor='Gemini')
env_control(command='workbench_set_editing_mode', target_id='structure', actor='Gemini')
env_control(command='workbench_reset_angles', target_id='all', actor='Gemini')
env_read(query='text_theater_snapshot')
```

## Build A Real Motion Spell

After reset, switch to pose mode:

```text
env_control(command='workbench_set_editing_mode', target_id='pose', actor='Gemini')
```

Then run at least two pose/capture pairs before compile/play:

```text
env_control(command='workbench_set_pose_batch', target_id='{"poses":[{"bone":"chest","rotation_deg":[0,0,14]},{"bone":"upper_arm_l","rotation_deg":[0,0,-70]},{"bone":"upper_arm_r","rotation_deg":[0,0,70]}]}', actor='Gemini')
env_control(command='workbench_capture_pose', target_id='spell_up', actor='Gemini')
env_control(command='workbench_set_pose_batch', target_id='{"poses":[{"bone":"chest","rotation_deg":[0,0,-10]},{"bone":"upper_arm_l","rotation_deg":[0,0,-15]},{"bone":"upper_arm_r","rotation_deg":[0,0,15]}]}', actor='Gemini')
env_control(command='workbench_capture_pose', target_id='spell_down', actor='Gemini')
env_control(command='workbench_compile_clip', target_id='wizard_spell', actor='Gemini')
env_control(command='workbench_play_authored_clip', target_id='wizard_spell', actor='Gemini')
env_read(query='text_theater_snapshot')
```

## Receipt Rule

Do not say "it is dancing" unless a fresh live receipt shows:

- `last_action` is the actual command you ran, not `Unknown control command`.
- `timeline.key_pose_count > 0`.
- Posed bones or animation/playback state changed.
- Freshness does not show stale mirror lag, or you disclose the lag and read again.

## Current Codex Live Receipt - 2026-06-01

This is the state Gemini should reacquire, not merely repeat from memory.

- Continuity session: `019e83fa-87cf-7a41-a0c3-5edb3ae86fdb`.
- Theater continuity packet: `019e763e-3ac1-7100-b2de-1903e61aaa71`.
- Latest browser corroboration: `D:\End-Game\ended_game\Champion_Council_private\static\captures\supercam_1780339072866.jpg`.
- Mode: `character / builder_subject / mounted_primary`.
- Clip: `codex_wizard_shuffle`, playing.
- Timeline: `duration 6`, `4 key poses`, `displacement_mode in_place`.
- Current sequence reads as a Vitruvian Dance / saiyan force-wave shuffle.
- Balance: `double_support`, risk `0`, both feet grounded/full/flat/stable.
- Hair growth: `displaced_mind_follicle_trace`, growth `0.96`, density `0.91`, budget `240`.
- Hair message: `DISPLACEDMINDPERTURBANCEFOLLICLECHAINWIZARDSHUFFLE`.
- Hair projection: `121` labels, `121` Braille labels, receipt `applied`, projected `true`.
- Hair response: `audio_reactive_saiyan`, spectrum `crayolazy`, pop `0.98`, glow `0.84`.
- Weather at handoff: disabled. This was intentional until the operator explicitly asked for weather/gravity particles.

## Codex Hair Chain Commands

These were the character/workbench hair controls that produced the receipt above. Keep command and target separate.

```json
{"command":"hair_set_granulation","target_id":"{\"preset\":\"roots_mullet_lion_mane_beard\",\"active\":true,\"growth_gain\":0.96,\"strand_density_gain\":0.91,\"tuft_density_gain\":0.94,\"undulation_gain\":0.93,\"color_octave_gain\":1,\"expression_coupling_gain\":0.86,\"simian_ramp_gain\":0.52,\"exhaust_bud_gain\":0.62,\"counterweight_gain\":0.48,\"impact_resound_gain\":0.78,\"ape_scale_gain\":0.22,\"ape_scale_factor\":1.18,\"max_sample_budget\":240,\"preset_label\":\"displaced_mind_follicle_trace\",\"caption\":\"animated letter perturbance / no repeated strand cadence\"}","actor":"Gemini"}
{"command":"hair_set_glyph_message","target_id":"{\"active\":true,\"message_text\":\"DISPLACEDMINDPERTURBANCEFOLLICLECHAINWIZARDSHUFFLE\",\"mode\":\"inertial_word_caterpillar\",\"phase_rate\":1.72,\"repeat_count\":3.6,\"slot_phase_stride\":0.73,\"sample_phase_stride\":1.19,\"tuft_phase_stride\":0.47,\"inertia_gain\":0.78,\"cohesion_gain\":0.31,\"legibility_gain\":0.9,\"sharpness_gain\":0.93,\"camera_alignment_gain\":0.86,\"gravity_bias\":0.64,\"turbulence_floor\":0.72,\"mutation_gain\":0.97,\"perturbance_gain\":0.94,\"offset_gain\":0.88,\"grid_columns\":13,\"grid_rows\":21,\"spectrum_mode\":\"crayolazy\",\"spectrum_gain\":1,\"pop_gain\":0.98,\"hue_rate\":0.58,\"hue_stride\":0.33,\"glow_gain\":0.84,\"text_surface\":{\"active\":true,\"web_projection_enabled\":true,\"text_leads_web\":true,\"authority\":\"text_theater_primary\",\"sync_mode\":\"text_leads_web\",\"web_projection_mode\":\"negative_space_braille\",\"web_projection_budget\":240,\"web_projection_sample_stride\":1,\"web_projection_granularity\":\"message_braille_meso\",\"web_projection_carrier_visibility\":\"text_only\",\"equilibrium_font_px\":18,\"equilibrium_braille_threshold_px\":10,\"equilibrium_focus_window_px\":7,\"macro_font_px\":24,\"meso_font_px\":17,\"micro_font_px\":11,\"font_equilibrium_mode\":\"braille_to_crisp\",\"allow_braille\":true,\"allow_glow\":true,\"allow_sparkle\":true,\"stationary_glyphs\":false},\"caption\":\"letters ripple through follicles according to the active shuffle\"}","actor":"Gemini"}
{"command":"pose_drive_set","target_id":"{\"enabled\":true,\"mode\":\"hair_follow_shuffle\",\"drive_gain\":0.72,\"body_motion_gain\":0.58,\"cadence_bias\":0.86,\"source\":\"codex_wizard_shuffle_hair_chain\",\"caption\":\"hair perturbance follows active authored shuffle\"}","actor":"Gemini"}
```

## Audio Reactive And Weather/Gravity Accent Rules

The right mental model is: motion clip -> pose drive -> hair granulation/glyph chain -> audio/acoustic force-wave -> optional weather/gravity accent.

- Use `env_help(topic='techlit_hair_control_surface')` for glyph message, spectrum, desktop audio routing, mic hair words, and shared force-wave sync.
- Use `env_help(topic='hair_granulation_surface')` for growth/reduction, density, simian ramp, safe sample budget.
- Use `env_help(topic='audio_reactive_capture_surface')` before claiming live audio input is active.
- Use `env_help(topic='pose_drive_surface')` before changing body follow.
- Use `env_help(topic='kaioken_aura_surface')` before changing aura intensity.
- `weather_set_surface` is allowed only when the operator explicitly asks for weather/gravity particles. It writes the text-theater weather surface and can objectify weather lanes into `weather_forge_lane_N` habitat outputs. Do not use generic `env_spawn` or scene-builder object work for this lane.
- If weather goes wrong or drifts into the wrong surface, immediately use `env_control(command='weather_clear_surface', actor='Gemini')` and verify `WEATHER: none`.

## Weather Particle Accent Starter

Use this only after the operator asks for gravity/weather particle accents. It is intended to make the active dance read as a charged particle performance without hand-building environment scene objects.

```json
{"command":"weather_set_surface","target_id":"{\"enabled\":true,\"source\":\"workbench_dance_weather_resource\",\"kind\":\"energy\",\"flow_class\":\"gravity_inversion_particle_accent\",\"density\":0.64,\"speed\":0.92,\"turbulence\":0.47,\"direction\":{\"x\":0.18,\"y\":0.34,\"z\":-0.92},\"drift\":{\"x\":0.06,\"y\":0.18,\"z\":-0.12},\"glyphs\":\"PHOENIXWINDMANA\",\"glyph_stride\":4,\"render_fidelity\":\"adaptive\",\"color_hint\":\"#facc15\",\"accent_color_hint\":\"#60a5fa\",\"weather_layers\":[{\"lane_index\":1,\"kind\":\"energy\",\"flow_class\":\"everquest_particle_burst\",\"density\":0.72,\"speed\":1.06,\"turbulence\":0.58,\"glyphs\":\"MANA\",\"glyph_stride\":4,\"color_hint\":\"#a78bfa\",\"accent_color_hint\":\"#f0abfc\"},{\"lane_index\":2,\"kind\":\"mist\",\"flow_class\":\"gravity_soft_lift\",\"density\":0.42,\"speed\":0.7,\"turbulence\":0.25,\"glyphs\":\"LIFT\",\"glyph_stride\":4,\"color_hint\":\"#67e8f9\",\"accent_color_hint\":\"#fde68a\"}]}","actor":"Gemini"}
```

Verification after weather:

1. `env_read(query='text_theater_view', view='render', diagnostics=true)`
2. `env_read(query='text_theater_snapshot')`
3. Confirm `WEATHER:` is enabled with the intended `kind / flow_class`.
4. Confirm `field_disposition.medium_kind` follows the weather or support-gravity lane.
5. Confirm hair still reports the intended `hair_msg`, growth, and label count.
6. `env_control(command='capture_supercam', actor='Gemini')`, then `env_read(query='supercam')`.
