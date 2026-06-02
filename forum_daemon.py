"""Forum router daemon — the hands-off way to keep a mind on the channel.

Run this once and it stays on: it heartbeats your presence, watches the channel,
and answers each assigned turn by invoking your agent CLI headlessly —
independent of any chat window. This is what lets the operator stop pinging
anyone.

    copy forum_daemon.config.example.json forum_daemon.config.json
    python forum_daemon.py

or with environment variables:

    FORUM_AGENT=Claude  FORUM_AGENT_CMD="claude -p"       python forum_daemon.py
    FORUM_AGENT=Gemini  FORUM_AGENT_CMD="gemini -p \" \""  python forum_daemon.py

Knobs (env vars):
  FORUM_CHANNEL     path to cli_brain_channel  (default: ./cli_brain_channel)
  FORUM_AGENT       your name on the roster      (default: Claude)
  FORUM_AGENT_CMD   shell command that reads the prompt on STDIN and prints the
                    reply to STDOUT. Empty = presence-only (won't answer).
  FORUM_ANSWER_WHEN  Legacy single-answer routing only. "addressed" (default)
                     answers named/open turns; "always" answers every unclaimed
                     turn. Bear Claw ignores this and uses explicit assignments.
  FORUM_KNOWN_AGENTS comma-separated routing aliases; connected/*.json is also read.

Two request modes:
  - legacy: one global resp_<id>.txt, protected by claim_<id>.lock.
  - bear_claw: every targeted daemon writes runs/<id>/responses/<Agent>.json.
    One mind is at bat; the rest are watchers with bounded notes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Configuration: a JSON settings file (forum_daemon.config.json), each key
# overridable by the matching env var. See forum_daemon.config.example.json.
BASE_DIR = Path(__file__).resolve().parent
_CFG_PATH = Path(os.environ.get("FORUM_CONFIG") or (BASE_DIR / "forum_daemon.config.json"))
if not _CFG_PATH.is_absolute():
    _CFG_PATH = BASE_DIR / _CFG_PATH
try:
    _CFG = json.loads(_CFG_PATH.read_text(encoding="utf-8")) if _CFG_PATH.exists() else {}
except (ValueError, OSError):
    _CFG = {}


def _conf(key: str, env: str, default):
    v = os.environ.get(env)
    if v is not None and v != "":
        return v
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


def _list_conf(key: str, env: str, default: list[str]) -> list[str]:
    value = _conf(key, env, default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def _bool_conf(key: str, env: str, default: bool = False) -> bool:
    value = _conf(key, env, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_conf(key: str, env: str, default: float = 0.0) -> float:
    value = _conf(key, env, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def capability_card() -> dict:
    permissions = dict(_CFG.get("permissions") or {})
    permissions.update({
        "can_delete": _bool_conf("can_delete", "FORUM_CAN_DELETE", bool(permissions.get("can_delete", False))),
        "can_publish": _bool_conf("can_publish", "FORUM_CAN_PUBLISH", bool(permissions.get("can_publish", False))),
        "can_send_messages": _bool_conf("can_send_messages", "FORUM_CAN_SEND_MESSAGES", bool(permissions.get("can_send_messages", False))),
        "can_move_funds": _bool_conf("can_move_funds", "FORUM_CAN_MOVE_FUNDS", bool(permissions.get("can_move_funds", False))),
        "can_change_auth": _bool_conf("can_change_auth", "FORUM_CAN_CHANGE_AUTH", bool(permissions.get("can_change_auth", False))),
        "requires_operator_approval_for_external_effects": _bool_conf(
            "requires_operator_approval_for_external_effects",
            "FORUM_REQUIRES_OPERATOR_APPROVAL",
            bool(permissions.get("requires_operator_approval_for_external_effects", True)),
        ),
    })
    limits = dict(_CFG.get("limits") or {})
    limits.update({
        "max_job_seconds": int(_float_conf("max_job_seconds", "FORUM_MAX_JOB_SECONDS", float(limits.get("max_job_seconds", ANSWER_TIMEOUT)))),
        "max_retries": int(_float_conf("max_retries", "FORUM_MAX_RETRIES", float(limits.get("max_retries", 1)))),
        "max_kleene_iterations": int(_float_conf("max_kleene_iterations", "FORUM_MAX_KLEENE_ITERATIONS", float(limits.get("max_kleene_iterations", 3)))),
        "max_spend_usd": _float_conf("max_spend_usd", "FORUM_MAX_SPEND_USD", float(limits.get("max_spend_usd", 0.0))),
    })
    return {
        "schema": "champion-continuum/utility-daemon-card/v1",
        "agent": AGENT,
        "kind": str(_conf("kind", "FORUM_KIND", "utility_daemon")),
        "capabilities": _list_conf("capabilities", "FORUM_CAPABILITIES", ["chat"]),
        "outputs": _list_conf("outputs", "FORUM_OUTPUTS", ["text"]),
        "cost_mode": str(_conf("cost_mode", "FORUM_COST_MODE", "unknown")),
        "risk_level": str(_conf("risk_level", "FORUM_RISK_LEVEL", "unknown")),
        "permissions": permissions,
        "limits": limits,
    }


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
    d = CHANNEL / "connected"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{AGENT}.json").write_text(
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
            "capability_card": capability_card(),
        }),
        encoding="utf-8",
    )


def build_prompt(messages: list[dict], role_note: str = "") -> str:
    parts = []
    if role_note:
        parts.append(f"[SYSTEM]\n{role_note}")
    for m in messages:
        parts.append(f"[{str(m.get('role', '')).upper()}]\n{m.get('content', '')}")
    parts.append("[ASSISTANT]\n")  # the agent completes this turn
    return "\n\n".join(parts)


def _configured_agents() -> set[str]:
    raw = _conf("known_agents", "FORUM_KNOWN_AGENTS", [])
    if isinstance(raw, str):
        names = raw.split(",")
    else:
        names = list(raw or [])
    return {str(n).strip().lower() for n in names if str(n).strip()}


def known_agent_names() -> set[str]:
    names = {"claude", "gemini", "codex", AGENT.lower()}
    names.update(_configured_agents())
    try:
        for presence in (CHANNEL / "connected").glob("*.json"):
            names.add(presence.stem.lower())
    except OSError:
        pass
    return {n for n in names if n}


def _mentioned(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def addressed(messages: list[dict]) -> bool:
    """Is the latest user message aimed at this agent (names it, or names no one)?"""
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        return True
    text = str(users[-1].get("content", "")).lower()
    me = AGENT.lower()
    # named explicitly -> yes; nobody named -> open to anyone; another named -> no
    named = [n for n in known_agent_names() if _mentioned(n, text)]
    if me in named:
        return True
    if not named:
        return True
    return False


def claim(rid: str) -> bool:
    """Atomically claim a turn so two daemons don't both answer it."""
    cf = CHANNEL / f"claim_{rid}.{AGENT}"
    try:
        fd = os.open(str(CHANNEL / f"claim_{rid}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, AGENT.encode())
        os.close(fd)
        cf.write_text(AGENT, encoding="utf-8")
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def _name_matches(name: str, candidates: list[str]) -> bool:
    return any(str(c).lower() == name.lower() for c in candidates)


def bear_claw_role(data: dict) -> str:
    forum = data.get("forum") or {}
    if (data.get("mode") or forum.get("mode")) != "bear_claw":
        return ""
    agent = AGENT.lower()
    at_bat = str(forum.get("at_bat") or "")
    watchers = [str(x) for x in (forum.get("watchers") or [])]
    hearers = [str(x) for x in (forum.get("hearers") or [])]
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
    run_dir = CHANNEL / "runs" / rid
    resp_dir = run_dir / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", AGENT.strip()) or "agent"
    path = resp_dir / f"{safe}.json"
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
    """Invoke the configured agent CLI. Two modes, both injection-safe:
      - default: pipe the prompt on STDIN  (e.g. AGENT_CMD="claude -p")
      - {prompt_file}: AGENT_CMD contains the token, replaced by a temp file path
        holding the prompt (e.g. "mycli --file {prompt_file}")."""
    if not AGENT_CMD:
        return ""
    try:
        agent_cmd = _agent_cmd_for_stdin_prompt()
        if "{prompt_file}" in agent_cmd:
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(prompt)
                tf_path = tf.name
            try:
                cmd = agent_cmd.replace("{prompt_file}", tf_path)
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
                    os.unlink(tf_path)
                except OSError:
                    pass
        else:
            proc = subprocess.run(agent_cmd, shell=True, input=prompt,
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=ANSWER_TIMEOUT)
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:  # noqa
        return f"(agent command failed: {type(exc).__name__}: {exc})"


def main() -> None:
    CHANNEL.mkdir(parents=True, exist_ok=True)
    print(f"forum daemon up · agent={AGENT} · channel={CHANNEL} · "
          f"cmd={AGENT_CMD or '(presence-only)'} · answer_when={ANSWER_WHEN}", flush=True)
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
            msgs = data.get("messages", [])
            if not AGENT_CMD:
                continue  # presence-only: leave the answering to a mind that can
            bc_role = bear_claw_role(data)
            if bc_role:
                run_resp = CHANNEL / "runs" / rid / "responses" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', AGENT.strip()) or 'agent'}.json"
                if run_resp.exists():
                    continue
                reply = answer(build_prompt(msgs, bear_claw_prompt_note(data, bc_role))) or f"({AGENT} produced no output)"
                write_bear_claw_response(data, bc_role, reply)
                print(f"bear_claw {bc_role} answered {rid} ({len(reply)} chars)", flush=True)
                continue
            if data.get("mode") == "bear_claw":
                continue
            if ANSWER_WHEN == "addressed" and not addressed(msgs):
                continue  # this turn is for another mind
            if not claim(rid):
                continue  # another mind got it first
            reply = answer(build_prompt(msgs)) or f"({AGENT} produced no output)"
            (CHANNEL / f"resp_{rid}.txt").write_text(reply, encoding="utf-8")
            print(f"answered {rid} ({len(reply)} chars)", flush=True)
        time.sleep(POLL_EVERY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("forum daemon stopped", flush=True)
        sys.exit(0)
