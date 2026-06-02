#!/usr/bin/env python
"""Small guarded workbench tool for Gemini CLI.

This intentionally exposes a narrow lane:
- read current theater receipts
- reset the character workbench
- build/play one known-good authored clip
- apply readable hair glyph resources

It uses the local Champion Council HTTP proxy instead of asking Gemini to juggle
raw MCP calls and receipt gates by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:7866"
DEFAULT_ACTOR = "GeminiWorkbenchTool"
DEFAULT_CLIP = "gemini_wizard_guarded_shuffle"
DEFAULT_MESSAGE = "VITRUVIANBASSDROPAUDIOSYNCACTIVE"


class ToolError(RuntimeError):
    pass


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise ToolError(f"HTTP {exc.code} from {url}: {raw[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"Could not reach {url}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise ToolError(f"Non-JSON response from {url}: {raw[:1200]}") from exc
    if not isinstance(parsed, dict):
        raise ToolError(f"Unexpected response from {url}: {parsed!r}")
    return parsed


def _unwrap_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    """Unwrap /api/tool responses into the actual tool payload."""
    if "result" not in response:
        return response
    result = response.get("result")
    if not isinstance(result, dict):
        return response
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    first = content[0]
    if not isinstance(first, dict):
        return result
    text = first.get("text")
    if not isinstance(text, str):
        return result
    try:
        parsed = json.loads(text)
    except Exception:
        return {"text": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


class WorkbenchClient:
    def __init__(self, base_url: str, actor: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.actor = actor

    def tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        response = _post_json(f"{self.base_url}/api/tool/{name}", args or {})
        payload = _unwrap_tool_response(response)
        if isinstance(payload, dict) and payload.get("error"):
            raise ToolError(f"{name} failed: {payload.get('error')} :: {payload}")
        return payload

    def control(self, command: str, target_id: Any = "", *, actor: str | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {
            "command": command,
            "actor": actor or self.actor,
        }
        if target_id is not None and target_id != "":
            args["target_id"] = _json_dumps(target_id) if isinstance(target_id, (dict, list)) else str(target_id)
        return self.tool("env_control", args)

    def read(self, query: str, **kwargs: Any) -> dict[str, Any]:
        args = {"query": query}
        args.update(kwargs)
        return self.tool("env_read", args)

    def embodiment_text(self) -> str:
        payload = self.read("text_theater_embodiment")
        text = payload.get("text_theater_embodiment", "")
        return text if isinstance(text, str) else ""

    def snapshot(self) -> dict[str, Any]:
        payload = self.read("text_theater_snapshot", diagnostics=True)
        snap = payload.get("text_theater_snapshot")
        return snap if isinstance(snap, dict) else {}


def _match_int(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except Exception:
        return default


def _match_float(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return float(match.group(1))
    except Exception:
        return default


def facts(client: WorkbenchClient) -> dict[str, Any]:
    text = client.embodiment_text()
    snap = client.snapshot()
    render = snap.get("render") if isinstance(snap.get("render"), dict) else {}
    hair_projection = render.get("hair_text_projection") if isinstance(render.get("hair_text_projection"), dict) else {}
    hair_eq = snap.get("hair_text_equilibrium") if isinstance(snap.get("hair_text_equilibrium"), dict) else {}
    pan = snap.get("pan_probe") if isinstance(snap.get("pan_probe"), dict) else {}
    pan_timeline = pan.get("timeline") if isinstance(pan.get("timeline"), dict) else {}

    key_pose_count = _match_int(r"TIMELINE:.*?/\s*(\d+)\s+key poses", text)
    if not key_pose_count and isinstance(pan_timeline.get("key_pose_count"), int):
        key_pose_count = int(pan_timeline.get("key_pose_count") or 0)
    cursor = _match_float(r"TIMELINE:\s*cursor\s*([0-9.]+)", text)
    if not cursor and isinstance(pan_timeline.get("cursor"), (int, float)):
        cursor = float(pan_timeline.get("cursor") or 0)
    clip_match = re.search(r"\bclip\s+([A-Za-z0-9_.:-]+)", text)
    return {
        "mode": ((snap.get("theater") or {}).get("mode") if isinstance(snap.get("theater"), dict) else ""),
        "visual_mode": ((snap.get("theater") or {}).get("visual_mode") if isinstance(snap.get("theater"), dict) else ""),
        "last_action": snap.get("last_action", ""),
        "last_sync_reason": snap.get("last_sync_reason", ""),
        "mirror_lag": bool((snap.get("stale_flags") or {}).get("mirror_lag")) if isinstance(snap.get("stale_flags"), dict) else None,
        "canvas_visible": render.get("canvas_visible"),
        "render_revision": render.get("render_revision"),
        "posed_bones": _match_int(r"BONES\s*\(\d+\):\s*\d+\s+selected,\s*(\d+)\s+posed", text),
        "key_pose_count": key_pose_count,
        "timeline_cursor": cursor,
        "clip": clip_match.group(1) if clip_match else "",
        "balance_risk": _match_float(r"BALANCE:.*?risk\s*([0-9.]+)", text),
        "weather": ((snap.get("weather") or {}).get("summary") if isinstance(snap.get("weather"), dict) else ""),
        "hair_message": hair_projection.get("message_text") or hair_eq.get("message_text") or "",
        "hair_glyph_state": hair_projection.get("glyph_state") or hair_eq.get("glyph_state") or "",
        "hair_label_count": hair_projection.get("label_count"),
        "hair_visible_label_count": hair_projection.get("visible_label_count"),
        "hair_apparent_font_px": hair_eq.get("apparent_font_px"),
        "hair_crisp_gain": hair_eq.get("crisp_gain"),
        "hair_allow_readable_text": hair_eq.get("allow_readable_text"),
    }


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def reset_workbench(client: WorkbenchClient) -> dict[str, Any]:
    steps = []
    for command, target in (
        ("workbench_new_builder", "humanoid_biped"),
        ("workbench_set_editing_mode", "structure"),
        ("workbench_reset_angles", "all"),
        ("workbench_set_editing_mode", "pose"),
    ):
        payload = client.control(command, target)
        steps.append({"command": command, "target_id": target, "status": payload.get("status"), "summary": payload.get("summary")})
        time.sleep(0.2)
    current = facts(client)
    return {"ok": current["mode"] == "character" and current["visual_mode"] == "builder_subject", "steps": steps, "facts": current}


def _pose_batch(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"poses": entries}


def _capture_pose(client: WorkbenchClient, label: str, timestamp: float, poses: list[dict[str, Any]], expected_min: int) -> dict[str, Any]:
    client.control("workbench_set_pose_batch", _pose_batch(poses))
    time.sleep(0.25)
    before_capture = facts(client)
    if before_capture["posed_bones"] <= 0:
        raise ToolError(f"Pose batch did not create posed bones for {label}: {before_capture}")
    client.control("workbench_capture_pose", {"label": label, "timestamp": timestamp})
    time.sleep(0.25)
    after_capture = facts(client)
    if after_capture["key_pose_count"] < expected_min:
        raise ToolError(f"Capture {label} did not advance key poses to {expected_min}: {after_capture}")
    return {"label": label, "timestamp": timestamp, "facts": after_capture}


def build_and_play(client: WorkbenchClient, clip_name: str = DEFAULT_CLIP, duration: float = 8.0) -> dict[str, Any]:
    client.control("workbench_set_editing_mode", "pose")
    time.sleep(0.2)
    captured = []
    sequence = [
        (
            "guarded_ready",
            0.0,
            [
                {"bone": "spine", "rotation_deg": [0, 0, -4]},
                {"bone": "chest", "rotation_deg": [0, 0, 8]},
                {"bone": "head", "rotation_deg": [0, 0, -5]},
                {"bone": "upper_arm_l", "rotation_deg": [0, 0, -45]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -24]},
                {"bone": "upper_arm_r", "rotation_deg": [0, 0, 45]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 24]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -10]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 10]},
            ],
        ),
        (
            "skycast",
            1.6,
            [
                {"bone": "spine", "rotation_deg": [-6, 0, 5]},
                {"bone": "chest", "rotation_deg": [-14, 0, 12]},
                {"bone": "head", "rotation_deg": [-8, 0, -4]},
                {"bone": "upper_arm_l", "rotation_deg": [0, 0, -128]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -34]},
                {"bone": "upper_arm_r", "rotation_deg": [0, 0, 128]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 34]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -18]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 18]},
            ],
        ),
        (
            "bass_drop",
            3.2,
            [
                {"bone": "hips", "offset": {"x": 0.08, "y": -0.1, "z": 0}},
                {"bone": "spine", "rotation_deg": [10, 0, -8]},
                {"bone": "chest", "rotation_deg": [16, 0, -14]},
                {"bone": "head", "rotation_deg": [10, 0, 8]},
                {"bone": "upper_arm_l", "rotation_deg": [35, 20, -48]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -74]},
                {"bone": "upper_arm_r", "rotation_deg": [35, -20, 48]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 74]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -24]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 24]},
            ],
        ),
        (
            "split_wave",
            4.8,
            [
                {"bone": "hips", "offset": {"x": -0.08, "y": -0.05, "z": 0}},
                {"bone": "spine", "rotation_deg": [0, 0, 16]},
                {"bone": "chest", "rotation_deg": [-4, 0, 24]},
                {"bone": "head", "rotation_deg": [0, 0, 12]},
                {"bone": "upper_arm_l", "rotation_deg": [0, 0, -84]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -48]},
                {"bone": "upper_arm_r", "rotation_deg": [0, 0, 36]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 62]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -34]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 34]},
            ],
        ),
        (
            "recover_flash",
            6.4,
            [
                {"bone": "hips", "offset": {"x": 0, "y": 0, "z": 0}},
                {"bone": "spine", "rotation_deg": [-5, 0, -8]},
                {"bone": "chest", "rotation_deg": [-10, 0, -16]},
                {"bone": "head", "rotation_deg": [-5, 0, -8]},
                {"bone": "upper_arm_l", "rotation_deg": [0, 0, -118]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -16]},
                {"bone": "upper_arm_r", "rotation_deg": [0, 0, 118]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 16]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -14]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 14]},
            ],
        ),
        (
            "return_ready",
            duration,
            [
                {"bone": "hips", "offset": {"x": 0, "y": 0, "z": 0}},
                {"bone": "spine", "rotation_deg": [0, 0, -4]},
                {"bone": "chest", "rotation_deg": [0, 0, 8]},
                {"bone": "head", "rotation_deg": [0, 0, -5]},
                {"bone": "upper_arm_l", "rotation_deg": [0, 0, -45]},
                {"bone": "lower_arm_l", "rotation_deg": [0, 0, -24]},
                {"bone": "upper_arm_r", "rotation_deg": [0, 0, 45]},
                {"bone": "lower_arm_r", "rotation_deg": [0, 0, 24]},
                {"bone": "upper_leg_l", "rotation_deg": [0, 0, -10]},
                {"bone": "upper_leg_r", "rotation_deg": [0, 0, 10]},
            ],
        ),
    ]
    for index, (label, timestamp, poses) in enumerate(sequence, start=1):
        captured.append(_capture_pose(client, label, timestamp, poses, index))

    client.control("workbench_compile_clip", {"clip_name": clip_name, "duration": duration})
    time.sleep(0.4)
    compiled = facts(client)
    if compiled["key_pose_count"] < len(sequence):
        raise ToolError(f"Compile lost key poses: {compiled}")

    client.control("workbench_play_authored_clip", {"clip_name": clip_name, "duration": duration, "loop": "repeat", "speed": 1})
    time.sleep(0.8)
    first = facts(client)
    time.sleep(1.0)
    second = facts(client)
    if second["key_pose_count"] < len(sequence):
        raise ToolError(f"Play verification has no key poses: first={first} second={second}")
    if not second["clip"] and clip_name not in json.dumps(second):
        raise ToolError(f"Play verification did not expose a clip name: {second}")
    return {
        "ok": True,
        "clip_name": clip_name,
        "captured": captured,
        "after_play_first": first,
        "after_play_second": second,
        "cursor_moved": first["timeline_cursor"] != second["timeline_cursor"],
    }


def apply_hair(client: WorkbenchClient, message: str = DEFAULT_MESSAGE) -> dict[str, Any]:
    granulation = {
        "preset": "roots_mullet_lion_mane_beard",
        "active": True,
        "growth_gain": 0.96,
        "strand_density_gain": 0.91,
        "tuft_density_gain": 0.94,
        "undulation_gain": 0.93,
        "color_octave_gain": 1,
        "expression_coupling_gain": 0.86,
        "simian_ramp_gain": 0.52,
        "exhaust_bud_gain": 0.62,
        "counterweight_gain": 0.48,
        "impact_resound_gain": 0.78,
        "ape_scale_gain": 0.22,
        "ape_scale_factor": 1.18,
        "max_sample_budget": 240,
        "preset_label": "displaced_mind_follicle_trace",
    }
    glyph = {
        "active": True,
        "message_text": message,
        "mode": "inertial_word_caterpillar",
        "phase_rate": 1.72,
        "repeat_count": 3.6,
        "slot_phase_stride": 0.73,
        "sample_phase_stride": 1.19,
        "tuft_phase_stride": 0.47,
        "inertia_gain": 0.78,
        "cohesion_gain": 0.31,
        "legibility_gain": 1,
        "sharpness_gain": 1,
        "camera_alignment_gain": 1,
        "gravity_bias": 0.64,
        "turbulence_floor": 0.72,
        "mutation_gain": 0.97,
        "perturbance_gain": 0.94,
        "offset_gain": 0.88,
        "grid_columns": 13,
        "grid_rows": 21,
        "spectrum_mode": "crayolazy",
        "spectrum_gain": 1,
        "pop_gain": 0.98,
        "hue_rate": 0.58,
        "hue_stride": 0.33,
        "glow_gain": 0.84,
        "text_surface": {
            "active": True,
            "web_projection_enabled": True,
            "text_leads_web": True,
            "authority": "text_theater_primary",
            "sync_mode": "text_leads_web",
            "web_projection_mode": "negative_space_braille",
            "web_projection_budget": 240,
            "web_projection_sample_stride": 1,
            "web_projection_granularity": "message_braille_meso",
            "web_projection_carrier_visibility": "text_only",
            "equilibrium_font_px": 18,
            "equilibrium_camera_distance": 24,
            "equilibrium_braille_threshold_px": 10,
            "equilibrium_focus_window_px": 7,
            "macro_font_px": 24,
            "meso_font_px": 17,
            "micro_font_px": 11,
            "font_equilibrium_mode": "braille_to_crisp",
            "force_readable_text": True,
            "readability_required": True,
            "allow_braille": True,
            "allow_glow": True,
            "allow_sparkle": True,
            "stationary_glyphs": False,
        },
    }
    client.control("hair_set_granulation", granulation)
    time.sleep(0.25)
    client.control("hair_set_glyph_message", glyph)
    time.sleep(0.25)
    client.control("pose_drive_set", {
        "enabled": True,
        "mode": "hair_follow_shuffle",
        "drive_gain": 0.72,
        "body_motion_gain": 0.58,
        "hair_motion_gain": 0.86,
        "source": "gemini_workbench_tool",
    })
    time.sleep(0.25)
    client.control("workbench_set_turntable", "off")
    client.control("workbench_frame_part", {"bone": "head", "view": "front"})
    time.sleep(0.4)
    current = facts(client)
    if current["hair_message"] != message:
        raise ToolError(f"Hair message did not apply: {current}")
    return {"ok": True, "facts": current}


def capture_receipts(client: WorkbenchClient) -> dict[str, Any]:
    client.control("capture_supercam")
    time.sleep(0.5)
    supercam = client.read("supercam")
    return {
        "facts": facts(client),
        "supercam": supercam.get("supercam", {}),
    }


def doctor(client: WorkbenchClient) -> dict[str, Any]:
    current = facts(client)
    diagnosis = []
    if not current.get("canvas_visible"):
        diagnosis.append("render canvas is not visible")
    if int(current.get("key_pose_count") or 0) <= 0:
        diagnosis.append("timeline has zero key poses; rebuild before claiming dance/playback")
    if float(current.get("hair_apparent_font_px") or 0) < 12 and not current.get("hair_allow_readable_text"):
        diagnosis.append("hair text is too small and not forced readable; it will look like tiny braille/no visible lettering")
    if current.get("mirror_lag"):
        diagnosis.append("mirror lag is present; read render/supercam before reporting")
    if not diagnosis:
        diagnosis.append("no immediate receipt failure detected")
    return {"ok": True, "facts": current, "diagnosis": diagnosis}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Guarded local workbench tool for Gemini CLI.")
    parser.add_argument("action", choices=["doctor", "reset", "build-play", "hair", "wizard", "capture"], help="Action to run.")
    parser.add_argument("--base-url", default=os.environ.get("CHAMPION_LOCAL_URL", DEFAULT_BASE_URL))
    parser.add_argument("--actor", default=os.environ.get("CHAMPION_ACTOR", DEFAULT_ACTOR))
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--no-reset", action="store_true", help="For wizard/build-play, do not reset first.")
    args = parser.parse_args(argv)

    client = WorkbenchClient(args.base_url, args.actor)
    try:
        if args.action == "doctor":
            result = doctor(client)
        elif args.action == "reset":
            result = reset_workbench(client)
        elif args.action == "build-play":
            if not args.no_reset:
                reset_workbench(client)
            result = build_and_play(client, args.clip)
        elif args.action == "hair":
            result = apply_hair(client, args.message)
        elif args.action == "wizard":
            if not args.no_reset:
                reset_workbench(client)
            result = {
                "build_play": build_and_play(client, args.clip),
                "hair": apply_hair(client, args.message),
                "receipts": capture_receipts(client),
            }
            result["ok"] = True
        elif args.action == "capture":
            result = capture_receipts(client)
        else:
            raise ToolError(f"Unknown action: {args.action}")
    except ToolError as exc:
        print_json({"ok": False, "error": str(exc)})
        return 2
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
