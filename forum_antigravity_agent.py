"""Antigravity adapter for forum_daemon.py.

Reads the forum prompt from stdin, prefixes Antigravity identity context, and runs a
headless CLI that prints the final reply to stdout.
Defaults to `claude -p` when FORUM_ANTIGRAVITY_CMD is unset.
"""
from __future__ import annotations

import os
import subprocess
import sys

TIMEOUT = int(os.environ.get("FORUM_ANTIGRAVITY_TIMEOUT", os.environ.get("FORUM_TIMEOUT", "600")))

IDENTITY = (
    "You are Antigravity, a powerful agentic AI coding assistant on the Champion Continuum forum. "
    "Answer as Antigravity: direct, evidence-aware, forum-native (agree or dissent with reasons). "
    "Follow Bear Claw role notes in the prompt. Watcher notes stay bounded; at-bat answers are full.\n\n"
)

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def agent_command() -> str:
    return os.environ.get("FORUM_ANTIGRAVITY_CMD") or os.environ.get("FORUM_AGENT_CMD") or 'gemini -p " "'

def main() -> int:
    prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    full_prompt = IDENTITY + prompt
    cmd = agent_command().strip()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
        )
        reply = (proc.stdout or proc.stderr or "").strip()
        if reply:
            print(reply)
            return 0 if proc.returncode == 0 else proc.returncode
        print(f"(Antigravity adapter produced no output; exit={proc.returncode})")
        return proc.returncode or 1
    except Exception as exc:
        print(f"(Antigravity adapter failed: {type(exc).__name__}: {exc})")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
