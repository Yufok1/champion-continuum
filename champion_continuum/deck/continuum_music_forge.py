from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


ROOT = Path(__file__).resolve().parent
CHANNEL = Path(os.environ.get("CONTINUUM_BRAIN_CHANNEL", ROOT / "cli_brain_channel"))
OUTPUT_ROOT = Path(os.environ.get("CONTINUUM_MUSIC_OUTPUTS", CHANNEL / "music_outputs"))
MAX_PREVIEW_CHARS = 2400
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"}

PUBLIC_MUSIC_BACKENDS = [
    {
        "label": "ACE-Step v1.5",
        "space_id": "ACE-Step/Ace-Step-v1.5",
        "kind": "hf_space",
        "notes": "Best first public Suno-ish lane; Space API can change, inspect before calling.",
    },
    {
        "label": "ACE-Step",
        "space_id": "ACE-Step/ACE-Step",
        "kind": "hf_space",
        "notes": "Earlier ACE-Step public Space.",
    },
    {
        "label": "ACE Jam",
        "space_id": "victor/ace-step-jam",
        "kind": "hf_space",
        "notes": "Public ACE-Step workflow Space.",
    },
    {
        "label": "MusicGen",
        "space_id": "facebook/MusicGen",
        "kind": "hf_space",
        "notes": "Reliable text-to-instrumental/music baseline.",
    },
]


def _optional_gradio_client():
    try:
        from gradio_client import Client  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"
    return Client, ""


def _slug(value: str, fallback: str = "song") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip()).strip("-._")
    return (value or fallback)[:80]


def _run_dir(title: str, backend: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return OUTPUT_ROOT / f"{stamp}-{_slug(backend, 'backend')}-{_slug(title, 'song')}"


def _json_preview(value: Any) -> str:
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        header = value.split(",", 1)[0]
        return f"{header},...[base64 omitted; audio saved separately when detected]"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except TypeError:
        text = repr(value)
    if len(text) > MAX_PREVIEW_CHARS:
        return text[:MAX_PREVIEW_CHARS] + "\n...[truncated]"
    return text


def _hash_saved_files(saved_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in saved_files:
        row = dict(item)
        path = row.get("path")
        if path and "sha256" not in row:
            try:
                row["sha256"] = sha256(Path(str(path)).read_bytes()).hexdigest()
            except OSError:
                row["sha256"] = ""
        enriched.append(row)
    return enriched


def _parse_payload_json(payload_json: str) -> dict[str, Any]:
    text = (payload_json or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("payload_json must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("payload_json must decode to a JSON object")
    return value


def _copy_local_file(path: Path, run_dir: Path, name_hint: str = "") -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".audio"
    name = _slug(name_hint or path.stem or "audio")
    target = run_dir / f"{name}{suffix}"
    counter = 2
    while target.exists():
        target = run_dir / f"{name}-{counter}{suffix}"
        counter += 1
    shutil.copy2(path, target)
    return {"kind": "local_file", "path": str(target), "bytes": target.stat().st_size}


def _download_audio(url: str, run_dir: Path, name_hint: str = "") -> dict[str, Any] | None:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix and suffix not in AUDIO_EXTENSIONS:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    name = _slug(name_hint or Path(parsed.path).stem or "audio")
    target = run_dir / f"{name}{suffix or '.mp3'}"
    counter = 2
    while target.exists():
        target = run_dir / f"{name}-{counter}{suffix or '.mp3'}"
        counter += 1
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if suffix == "" and not content_type.startswith(("audio/", "video/")):
            return None
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    return {"kind": "download", "url": url, "path": str(target), "bytes": target.stat().st_size}


def _save_data_audio_uri(value: str, run_dir: Path, name_hint: str = "") -> dict[str, Any] | None:
    match = re.match(r"^data:(audio/[^;,]+|video/[^;,]+);base64,(.+)$", value or "", re.DOTALL)
    if not match:
        return None
    media_type = match.group(1).lower()
    payload = match.group(2)
    ext_by_type = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "video/webm": ".webm",
    }
    suffix = ext_by_type.get(media_type, ".audio")
    run_dir.mkdir(parents=True, exist_ok=True)
    name = _slug(name_hint or "audio")
    target = run_dir / f"{name}{suffix}"
    counter = 2
    while target.exists():
        target = run_dir / f"{name}-{counter}{suffix}"
        counter += 1
    target.write_bytes(base64.b64decode(payload))
    return {"kind": "data_uri", "media_type": media_type, "path": str(target), "bytes": target.stat().st_size}


def _collect_audio_outputs(value: Any, run_dir: Path, name_hint: str = "audio") -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    if value is None:
        return saved
    if isinstance(value, (str, os.PathLike)):
        text = str(value)
        data_file = _save_data_audio_uri(text, run_dir, name_hint=name_hint)
        if data_file:
            saved.append(data_file)
            return saved
        local = Path(text)
        if local.exists() and local.is_file():
            suffix = local.suffix.lower()
            if suffix in AUDIO_EXTENSIONS or suffix == "":
                saved.append(_copy_local_file(local, run_dir, name_hint=name_hint))
            return saved
        if text.startswith(("http://", "https://")):
            downloaded = _download_audio(text, run_dir, name_hint=name_hint)
            if downloaded:
                saved.append(downloaded)
            return saved
        return saved
    if isinstance(value, dict):
        for key in ("path", "url", "name", "orig_name"):
            if key in value:
                saved.extend(_collect_audio_outputs(value.get(key), run_dir, name_hint=str(value.get("orig_name") or name_hint)))
        for key, item in value.items():
            if key in {"path", "url", "name", "orig_name"}:
                continue
            saved.extend(_collect_audio_outputs(item, run_dir, name_hint=f"{name_hint}-{key}"))
        return saved
    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value, start=1):
            saved.extend(_collect_audio_outputs(item, run_dir, name_hint=f"{name_hint}-{idx}"))
    return saved


def music_forge_state() -> dict[str, Any]:
    Client, import_error = _optional_gradio_client()
    return {
        "status": "ok",
        "service": "continuum-music-forge",
        "output_dir": str(OUTPUT_ROOT),
        "gradio_client_available": Client is not None,
        "gradio_client_error": import_error,
        "public_backends": PUBLIC_MUSIC_BACKENDS,
        "call_order": [
            "continuum_music_compose_packet",
            "continuum_music_backend_preset",
            "continuum_music_hf_space_schema",
            "continuum_music_generate_preset",
            "continuum_music_generate_hf_space",
        ],
        "notes": [
            "Use the existing Continuum MCP/SSE endpoint; no separate login scraping is needed.",
            "HF public Spaces can change their API names and inputs; inspect schema before generation.",
            "Generated files are saved locally and returned as file paths plus a manifest.",
        ],
    }


def compose_song_packet(
    idea: str,
    style: str = "",
    lyrics: str = "",
    language: str = "en-US",
    duration: str = "30 seconds",
    avoid: str = "Do not mimic living artists or request a copyrighted song clone.",
) -> dict[str, Any]:
    idea = (idea or "").strip()
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    title_seed = idea.splitlines()[0] if idea else "Continuum song"
    title = re.sub(r"\s+", " ", title_seed)[:70]
    caption_parts = [
        f"Original {duration} song",
        f"in {language}",
        style or "warm, memorable, human, emotionally clear",
        idea or "built from the user's conversation intent",
    ]
    caption = ". ".join(part for part in caption_parts if part) + "."
    if avoid:
        caption += f" Avoid: {avoid}"
    return {
        "status": "ok",
        "schema": "champion-continuum/music-packet/v1",
        "title": title,
        "language": language,
        "duration": duration,
        "idea": idea,
        "style": style,
        "lyrics": lyrics,
        "caption": caption,
        "suggested_hf_payload": {
            "args": [caption],
            "api_name": "/predict",
            "note": "Inspect the chosen Space schema first; replace args/api_name to match that Space.",
        },
    }


def hf_space_schema(space_id: str) -> dict[str, Any]:
    Client, import_error = _optional_gradio_client()
    if Client is None:
        return {
            "status": "error",
            "error": "gradio_client is not installed or could not import",
            "detail": import_error,
            "install": "pip install gradio_client",
        }
    client = Client(space_id)
    try:
        api = client.view_api(return_format="dict")
    except TypeError:
        api = client.view_api()
    return {"status": "ok", "space_id": space_id, "api": api}


def _ace_step_v15_generation_kwargs(prompt: str, lyrics: str, duration: float, seed: int) -> dict[str, Any]:
    random_seed = int(seed) < 0
    vocal_language = "en" if lyrics else "unknown"
    duration_value = float(duration) if float(duration) > 0 else -1.0
    return {
        "selected_model": "acestep-v15-xl-turbo",
        "generation_mode": "custom",
        "simple_query_input": prompt,
        "simple_vocal_language": vocal_language,
        "param_4": prompt,
        "param_5": lyrics,
        "param_6": 0,
        "param_7": "",
        "param_8": "",
        "param_9": vocal_language,
        "param_10": 8,
        "param_11": 7.0,
        "param_12": random_seed,
        "param_13": "-1" if random_seed else str(int(seed)),
        "param_14": None,
        "param_15": duration_value,
        "param_16": 1,
        "param_17": None,
        "param_18": "",
        "param_19": 0.0,
        "param_20": -1.0,
        "param_21": "Fill the audio semantic mask based on the given conditions:",
        "param_22": 1.0,
        "param_23": "text2music",
        "param_24": False,
        "param_25": 0.0,
        "param_26": 1.0,
        "param_27": 3.0,
        "param_28": "ode",
        "param_29": "",
        "param_30": "mp3",
        "param_31": 0.85,
        "param_32": True,
        "param_33": 2.0,
        "param_34": 0,
        "param_35": 0.9,
        "param_36": "NO USER INPUT",
        "param_37": True,
        "param_38": True,
        "param_39": True,
        "param_41": False,
        "param_42": True,
        "param_43": False,
        "param_44": False,
        "param_45": 0.5,
        "param_46": 8,
        "param_47": "vocals",
        "param_48": [],
        "param_49": False,
    }


def music_backend_preset_payload(
    backend: str,
    prompt: str,
    lyrics: str = "",
    duration: float = 30.0,
    seed: int = -1,
) -> dict[str, Any]:
    backend_key = _slug(backend or "ace_jam").lower().replace("-", "_")
    prompt = (prompt or "").strip()
    lyrics = (lyrics or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if backend_key in {"ace_step_v15", "acestep_v15", "ace_v15", "ace_step"}:
        return {
            "backend": "ace_step_v15",
            "space_id": "ACE-Step/Ace-Step-v1.5",
            "api_name": "/generation_wrapper",
            "kwargs": _ace_step_v15_generation_kwargs(prompt, lyrics, duration, seed),
            "notes": "ACE-Step v1.5 currently exposes /generation_wrapper. Inspect schema before generation if this Space changes again.",
        }
    if backend_key in {"ace_jam", "victor_ace_step_jam", "jam"}:
        return {
            "backend": "ace_jam",
            "space_id": "victor/ace-step-jam",
            "api_name": "/generate",
            "args": [
                prompt,
                lyrics,
                float(duration),
                8,
                7.0,
                int(seed),
                "",
                0.8,
            ],
            "notes": "ACE Jam has the smallest public API surface; good first smoke test.",
        }
    if backend_key in {"musicgen", "facebook_musicgen"}:
        return {
            "backend": "musicgen",
            "space_id": "facebook/MusicGen",
            "api_name": "/predict_batched",
            "args": [prompt, None],
            "notes": "MusicGen is mostly instrumental; melody input is optional in the UI but may vary by Space API.",
        }
    raise ValueError("unknown backend preset; use ace_jam, ace_step_v15, or musicgen")


def generate_music_preset(
    backend: str,
    prompt: str,
    lyrics: str = "",
    duration: float = 30.0,
    seed: int = -1,
    title: str = "",
) -> dict[str, Any]:
    preset = music_backend_preset_payload(
        backend=backend,
        prompt=prompt,
        lyrics=lyrics,
        duration=duration,
        seed=seed,
    )
    payload: dict[str, Any] = {"api_name": preset["api_name"], "title": title or prompt[:80]}
    if "kwargs" in preset:
        payload["kwargs"] = preset["kwargs"]
    else:
        payload["args"] = preset["args"]
    return generate_hf_space_song(
        space_id=preset["space_id"],
        prompt=prompt,
        payload_json=json.dumps(payload, ensure_ascii=False),
        api_name=preset["api_name"],
        title=title or prompt[:80],
    ) | {"preset": preset}


def generate_hf_space_song(
    space_id: str,
    prompt: str = "",
    payload_json: str = "",
    api_name: str = "/predict",
    title: str = "",
) -> dict[str, Any]:
    Client, import_error = _optional_gradio_client()
    if Client is None:
        return {
            "status": "error",
            "error": "gradio_client is not installed or could not import",
            "detail": import_error,
            "install": "pip install gradio_client",
        }
    payload = _parse_payload_json(payload_json)
    space_id = (space_id or "").strip()
    if not space_id:
        raise ValueError("space_id is required")
    api_name = str(payload.get("api_name") or api_name or "/predict")
    run_title = str(title or payload.get("title") or prompt or space_id)
    run_dir = _run_dir(run_title, space_id.replace("/", "-"))
    run_dir.mkdir(parents=True, exist_ok=True)

    client = Client(space_id)
    started = time.time()
    if "args" in payload:
        args = payload.get("args")
        if not isinstance(args, list):
            raise ValueError("payload_json args must be a list")
        result = client.predict(*args, api_name=api_name)
    elif "kwargs" in payload:
        kwargs = payload.get("kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError("payload_json kwargs must be an object")
        result = client.predict(**kwargs, api_name=api_name)
    else:
        result = client.predict(prompt, api_name=api_name)

    saved_files = _hash_saved_files(_collect_audio_outputs(result, run_dir, name_hint=_slug(run_title, "song")))
    receipt_payload = {
        "space_id": space_id,
        "api_name": api_name,
        "title": run_title,
        "prompt_sha256": sha256((prompt or "").encode("utf-8")).hexdigest(),
        "artifact_sha256": [item.get("sha256") for item in saved_files if item.get("sha256")],
    }
    manifest = {
        "schema": "champion-continuum/music-generation/v1",
        "receipt_id": "music_receipt_" + sha256(
            json.dumps(receipt_payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20],
        "action_class": "generate",
        "approval_state": "operator_or_council_requested",
        "created_ms": int(time.time() * 1000),
        "space_id": space_id,
        "api_name": api_name,
        "title": run_title,
        "prompt": prompt,
        "payload_json": payload,
        "duration_sec": round(time.time() - started, 3),
        "saved_files": saved_files,
        "result_preview": _json_preview(result),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok" if saved_files else "no_audio_files_detected",
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "saved_files": saved_files,
        "result_preview": manifest["result_preview"],
    }
