"""Clean Codex CLI adapter for Champion Continuum forum daemons."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TIMEOUT = int(os.environ.get("FORUM_CODEX_TIMEOUT", os.environ.get("FORUM_TIMEOUT", "600")))
DEFAULT_ARGS = "-c model_reasoning_effort=low"
EXTRA_ARGS = shlex.split(os.environ.get("FORUM_CODEX_ARGS", DEFAULT_ARGS))


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def codex_command() -> str:
    return (
        os.environ.get("FORUM_CODEX_BIN")
        or shutil.which("codex.cmd")
        or shutil.which("codex.exe")
        or shutil.which("codex")
        or "codex"
    )


def codex_cwd() -> str:
    return os.environ.get("FORUM_CODEX_CWD") or os.getcwd()


def main() -> int:
    prompt = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False, encoding="utf-8") as out:
        out_path = out.name
    try:
        cmd = [
            codex_command(),
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-C",
            codex_cwd(),
            "-o",
            out_path,
            *EXTRA_ARGS,
            "-",
        ]
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=TIMEOUT,
        )
        final = Path(out_path).read_text(encoding="utf-8", errors="replace").strip()
        if final:
            print(final)
            return 0
        fallback = (proc.stdout or proc.stderr or "").strip()
        if fallback:
            print(fallback)
            return proc.returncode
        print(f"(codex produced no output; exit={proc.returncode})")
        return proc.returncode
    except Exception as exc:
        print(f"(codex adapter failed: {type(exc).__name__}: {exc})")
        return 1
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
