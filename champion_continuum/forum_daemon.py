"""Hands-off Champion Continuum forum daemon.

Run in a local deck/channel directory after `pip install champion-continuum`:

    $env:FORUM_AGENT = "Codex"
    $env:FORUM_AGENT_CMD = "continuum-codex-agent"
    continuum-forum-daemon

or point at a config file:

    $env:FORUM_CONFIG = "forum_daemon.codex.json"
    continuum-forum-daemon
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _config_path() -> Path:
    raw = os.environ.get("FORUM_CONFIG")
    path = Path(raw) if raw else Path.cwd() / "forum_daemon.config.json"
    return path.resolve()


_CFG_PATH = _config_path()
BASE_DIR = _CFG_PATH.parent
try:
    _CFG = json.loads(_CFG_PATH.read_text(encoding="utf-8")) if _CFG_PATH.exists() else {}
except (ValueError, OSError):
    _CFG = {}


def _conf(key: str, env: str, default):
    value = os.environ.get(env)
    if value is not None and value != "":
        return value
    return _CFG.get(key, default)


def _path_conf(key: str, env: str, default: str) -> Path:
    path = Path(str(_conf(key, env, default)))
    return path if path.is_absolute() else BASE_DIR / path


CHANNEL = _path_conf("channel", "FORUM_CHANNEL", "cli_brain_channel")
AGENT = str(_conf("agent", "FORUM_AGENT", "Claude"))
AGENT_CMD = str(_conf("agent_cmd", "FORUM_AGENT_CMD", "")).strip()
ANSWER_WHEN = str(_conf("answer_when", "FORUM_ANSWER_WHEN", "addressed")).lower()
HEARTBEAT_EVERY = float(_conf("heartbeat_every", "FORUM_HEARTBEAT", 8.0))
POLL_EVERY = float(_conf("poll_every", "FORUM_POLL", 1.0))
ANSWER_TIMEOUT = int(_conf("answer_timeout", "FORUM_TIMEOUT", 600))


def _agent_cmd_for_stdin_prompt() -> str:
    """Return a command that can accept the real prompt on stdin.

    Gemini CLI requires an argument after -p/--prompt even though it then appends
    stdin. Older paste-code used "gemini -p"; make that survivable.
    """
    cmd = AGENT_CMD.strip()
    if re.search(r"(^|\s)gemini(?:\.cmd|\.ps1)?\s+(?:-p|--prompt)\s*$", cmd, re.IGNORECASE):
        return cmd + ' " "'
    return cmd


def heartbeat() -> None:
    directory = CHANNEL / "connected"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{AGENT}.json").write_text(
        json.dumps({
            "agent": AGENT,
            "ts": time.time(),
            "pid": os.getpid(),
            "channel": str(CHANNEL),
            "root": str(BASE_DIR),
            "status": "ready" if AGENT_CMD else "presence-only",
            "busy": False,
            "can_speak": bool(AGENT_CMD),
            "can_watch": bool(AGENT_CMD),
        }),
        encoding="utf-8",
    )


def build_prompt(messages: list[dict], role_note: str = "") -> str:
    parts = []
    if role_note:
        parts.append(f"[SYSTEM]\n{role_note}")
    for message in messages:
        parts.append(f"[{str(message.get('role', '')).upper()}]\n{message.get('content', '')}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def _configured_agents() -> set[str]:
    raw = _conf("known_agents", "FORUM_KNOWN_AGENTS", [])
    names = raw.split(",") if isinstance(raw, str) else list(raw or [])
    return {str(name).strip().lower() for name in names if str(name).strip()}


def known_agent_names() -> set[str]:
    names = {"claude", "gemini", "codex", AGENT.lower()}
    names.update(_configured_agents())
    try:
        for presence in (CHANNEL / "connected").glob("*.json"):
            names.add(presence.stem.lower())
    except OSError:
        pass
    return {name for name in names if name}


def _mentioned(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def addressed(messages: list[dict]) -> bool:
    users = [message for message in messages if message.get("role") == "user"]
    if not users:
        return True
    text = str(users[-1].get("content", "")).lower()
    named = [name for name in known_agent_names() if _mentioned(name, text)]
    if AGENT.lower() in named:
        return True
    return not named


def claim(rid: str) -> bool:
    try:
        fd = os.open(str(CHANNEL / f"claim_{rid}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, AGENT.encode())
        os.close(fd)
        (CHANNEL / f"claim_{rid}.{AGENT}").write_text(AGENT, encoding="utf-8")
        return True
    except (FileExistsError, OSError):
        return False


def _name_matches(name: str, candidates: list[str]) -> bool:
    return any(str(candidate).lower() == name.lower() for candidate in candidates)


def bear_claw_role(data: dict) -> str:
    forum = data.get("forum") or {}
    if (data.get("mode") or forum.get("mode")) != "bear_claw":
        return ""
    agent = AGENT.lower()
    at_bat = str(forum.get("at_bat") or "")
    watchers = [str(value) for value in (forum.get("watchers") or [])]
    hearers = [str(value) for value in (forum.get("hearers") or [])]
    if at_bat.lower() == agent:
        return "at_bat"
    if _name_matches(AGENT, watchers):
        return "watcher"
    if _name_matches(AGENT, hearers):
        return "hearer"
    return ""


def bear_claw_prompt_note(data: dict, role: str) -> str:
    forum = data.get("forum") or {}
    at_bat = forum.get("at_bat") or "?"
    watchers = ", ".join(forum.get("watchers") or []) or "(none)"
    hearers = ", ".join(forum.get("hearers") or []) or "(none)"
    if role == "at_bat":
        return (
            "Bear Claw forum run. You are AT BAT for this operator turn. "
            "Answer fully and concretely. Other minds are watching and may add "
            "corroboration or objections. Keep your own judgment. "
            f"Hearers: {hearers}. Watchers: {watchers}."
        )
    if role == "watcher":
        return (
            "Bear Claw forum run. You are a WATCHER, not the main speaker. "
            f"{at_bat} is at bat. Read the same turn and write a bounded note: "
            "WATCH_OBJECTION if you see a concrete bug/risk/missing evidence; "
            "WATCH_CORROBORATION if you can support the at-bat path with useful "
            "evidence; WATCH_CLEAR if there is no material objection. Max 120 words. "
            "Do not use compose directives."
        )
    return (
        "Bear Claw forum run. You are a HEARER for continuity. Only write a short "
        "watch note if you have a concrete objection; otherwise write WATCH_CLEAR."
    )


def response_status(reply: str) -> tuple[bool, str]:
    text = (reply or "").strip()
    if not text:
        return False, "empty_output"
    low = text.lower()
    if re.match(r"^(you('ve| have) hit your (session|usage) limit|session limit reached|usage limit reached)\b", low):
        return False, "session_limit"
    failures = [
        ("prompt_encoding", ("invalid byte", "not valid utf-8", "failed to read prompt")),
        ("agent_command_failed", ("agent command failed",)),
        ("timeout", ("timed out", "timeout")),
        ("cli_usage", ("not enough arguments following", "usage: gemini", "show help")),
        ("rate_limit", ("rate limit", "quota")),
        ("auth", ("api key", "authentication", "unauthorized", "permission denied")),
    ]
    for label, patterns in failures:
        if any(pattern in low for pattern in patterns):
            return False, label
    return True, ""


def write_bear_claw_response(data: dict, role: str, reply: str) -> None:
    rid = str(data.get("id") or "")
    if not rid:
        return
    response_dir = CHANNEL / "runs" / rid / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", AGENT.strip()) or "agent"
    path = response_dir / f"{safe}.json"
    if path.exists():
        return
    ok, error_class = response_status(reply)
    payload = {
        "id": rid,
        "agent": AGENT,
        "role": role,
        "ok": ok,
        "ts": time.time(),
        "text": reply,
    }
    if error_class:
        payload["error_class"] = error_class
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def answer(prompt: str) -> str:
    if not AGENT_CMD:
        return ""
    try:
        agent_cmd = _agent_cmd_for_stdin_prompt()
        if "{prompt_file}" in agent_cmd:
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as temp:
                temp.write(prompt)
                temp_path = temp.name
            try:
                cmd = agent_cmd.replace("{prompt_file}", temp_path)
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=ANSWER_TIMEOUT,
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        else:
            proc = subprocess.run(
                agent_cmd,
                shell=True,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=ANSWER_TIMEOUT,
            )
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return f"(agent command failed: {type(exc).__name__}: {exc})"


def main() -> int:
    CHANNEL.mkdir(parents=True, exist_ok=True)
    print(
        f"forum daemon up - agent={AGENT} - channel={CHANNEL} - "
        f"cmd={AGENT_CMD or '(presence-only)'} - answer_when={ANSWER_WHEN}",
        flush=True,
    )
    last_hb = 0.0
    while True:
        now = time.time()
        if now - last_hb >= HEARTBEAT_EVERY:
            heartbeat()
            last_hb = now
        for req in sorted(CHANNEL.glob("req_*.json")):
            try:
                data = json.loads(req.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            rid = str(data.get("id") or req.stem.replace("req_", ""))
            if (CHANNEL / f"resp_{rid}.txt").exists():
                continue
            messages = data.get("messages", [])
            if not AGENT_CMD:
                continue
            bc_role = bear_claw_role(data)
            if bc_role:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", AGENT.strip()) or "agent"
                run_response = CHANNEL / "runs" / rid / "responses" / f"{safe}.json"
                if run_response.exists():
                    continue
                reply = answer(build_prompt(messages, bear_claw_prompt_note(data, bc_role))) or f"({AGENT} produced no output)"
                write_bear_claw_response(data, bc_role, reply)
                print(f"bear_claw {bc_role} answered {rid} ({len(reply)} chars)", flush=True)
                continue
            if data.get("mode") == "bear_claw":
                continue
            if ANSWER_WHEN == "addressed" and not addressed(messages):
                continue
            if not claim(rid):
                continue
            reply = answer(build_prompt(messages)) or f"({AGENT} produced no output)"
            (CHANNEL / f"resp_{rid}.txt").write_text(reply, encoding="utf-8")
            print(f"answered {rid} ({len(reply)} chars)", flush=True)
        time.sleep(POLL_EVERY)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("forum daemon stopped", flush=True)
        raise SystemExit(0)
