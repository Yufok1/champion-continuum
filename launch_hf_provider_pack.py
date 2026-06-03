"""Launch a local pack of Hugging Face Inference Provider forum daemons.

This starts local forum_daemon.py processes. It does not call any provider by
itself; credits are only used later when a daemon is assigned work and answers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from continuum_provider_registry import (
    HF_DEFAULT_MODEL,
    HF_INFERENCE_PROVIDER_CATALOG,
    HF_PROVIDER_CLIENT_UNSUPPORTED_IDS,
    HF_PROVIDER_DAEMON_ROUTES,
    HF_PROVIDER_MANUAL_OR_CREDIT_GATED_IDS,
)


ROOT = Path(__file__).resolve().parent
CHANNEL = Path(os.environ.get("CONTINUUM_BRAIN_CHANNEL") or os.environ.get("FORUM_CHANNEL") or ROOT / "cli_brain_channel")
CONNECTED = CHANNEL / "connected"
LOG_DIR = CHANNEL / "daemon_logs"


def _token_present() -> bool:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return clean.strip("-") or "HF-Provider"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _recent_heartbeat(agent: str, stale_after: float = 24.0) -> bool:
    heartbeat = _read_json(CONNECTED / f"{agent}.json")
    ts = float(heartbeat.get("ts") or 0)
    return bool(ts and (time.time() - ts) <= stale_after)


def _all_catalog_chat_routes() -> list[dict[str, Any]]:
    existing = {str(route.get("provider")) for route in HF_PROVIDER_DAEMON_ROUTES}
    routes = [dict(route) for route in HF_PROVIDER_DAEMON_ROUTES]
    for item in HF_INFERENCE_PROVIDER_CATALOG:
        provider = str(item.get("id") or "").strip()
        strengths = set(item.get("strengths") or [])
        if (
            not provider
            or provider in existing
            or provider in HF_PROVIDER_CLIENT_UNSUPPORTED_IDS
            or "chat_completion_llm" not in strengths
        ):
            continue
        routes.append(
            {
                "agent": f"HF-{_safe_name(provider)}",
                "provider": provider,
                "model": HF_DEFAULT_MODEL,
                "max_tokens": 900,
                "lane": "unverified_catalog_chat_attempt",
                "verified_route": False,
                "manual_or_credit_gated": provider in HF_PROVIDER_MANUAL_OR_CREDIT_GATED_IDS,
            }
        )
    return routes


def launch_pack(include_unverified: bool = False, dry_run: bool = False, restart_live: bool = False) -> dict[str, Any]:
    routes = _all_catalog_chat_routes() if include_unverified else [dict(route) for route in HF_PROVIDER_DAEMON_ROUTES]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONNECTED.mkdir(parents=True, exist_ok=True)
    token_present = _token_present()
    if not token_present and not dry_run:
        return {
            "status": "blocked",
            "reason": "No Hugging Face token was found. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or run `hf auth login` first.",
            "launched": [],
            "skipped": [],
            "route_count": len(routes),
        }

    launched: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for route in routes:
        agent = _safe_name(str(route.get("agent") or f"HF-{route.get('provider') or 'Provider'}"))
        provider = str(route.get("provider") or "auto")
        model = str(route.get("model") or HF_DEFAULT_MODEL)
        max_tokens = str(route.get("max_tokens") or 900)
        if not restart_live and _recent_heartbeat(agent):
            skipped.append({"agent": agent, "reason": "recent heartbeat already present"})
            continue
        env = os.environ.copy()
        env.update(
            {
                "FORUM_CONFIG": "forum_daemon.hf.json",
                "FORUM_AGENT": agent,
                "FORUM_HF_PROVIDER": provider,
                "FORUM_HF_MODEL": model,
                "FORUM_HF_MAX_TOKENS": max_tokens,
                "FORUM_CHANNEL": str(CHANNEL),
                "CONTINUUM_BRAIN_CHANNEL": str(CHANNEL),
            }
        )
        cmd = [sys.executable, str(ROOT / "forum_daemon.py")]
        stdout_path = LOG_DIR / f"{agent}.out.log"
        stderr_path = LOG_DIR / f"{agent}.err.log"
        if dry_run:
            launched.append({"agent": agent, "provider": provider, "model": model, "dry_run": True})
            continue
        flags = 0
        if os.name == "nt":
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, creationflags=flags)
        except Exception as exc:
            stdout.close()
            stderr.close()
            skipped.append({"agent": agent, "provider": provider, "model": model, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        launched.append(
            {
                "agent": agent,
                "provider": provider,
                "model": model,
                "pid": proc.pid,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "verified_route": bool(route.get("verified_route")),
                "manual_or_credit_gated": bool(route.get("manual_or_credit_gated")),
            }
        )
    return {
        "status": "ok",
        "schema": "champion-continuum/hf-provider-daemon-pack/v1",
        "token_present": token_present,
        "include_unverified": include_unverified,
        "route_count": len(routes),
        "launched": launched,
        "skipped": skipped,
        "truth_boundary": "Processes are started locally. Provider credits are only used when a daemon later answers an assigned turn.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Champion Continuum HF provider daemon pack.")
    parser.add_argument("--include-unverified", action="store_true", help="Also attempt chat-capable catalog providers without a verified model/provider route.")
    parser.add_argument("--restart-live", action="store_true", help="Launch even when a recent heartbeat exists for an agent name.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned launches without starting processes.")
    args = parser.parse_args()
    result = launch_pack(
        include_unverified=args.include_unverified,
        dry_run=args.dry_run,
        restart_live=args.restart_live,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
