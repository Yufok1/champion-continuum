from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .core import Continuum
from .engine import engine_self_test, launch_engine, reconstitute_engine
from .system_prompts import get_system_prompt, list_profiles


def _print(payload: Any, as_json: bool = False) -> None:
    if as_json or isinstance(payload, (dict, list)):
        _emit(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _emit(str(payload))


def _emit(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        data = (text + "\n").encode(encoding, errors="backslashreplace")
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        else:
            print(data.decode(encoding, errors="replace"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="continuum", description="Portable continuity memory primitive.")
    default_root = os.environ.get("CONTINUUM_ROOT", ".continuum")
    parser.add_argument("--root", default=default_root, help="Continuum data root (or set CONTINUUM_ROOT).")
    parser.add_argument("--version", action="version", version=f"champion-continuum {__version__}")

    # Shared parent so --root may also follow the subcommand. SUPPRESS keeps the
    # attribute unset unless --root is passed here, so a value given before the
    # subcommand is never clobbered by the subparser default.
    root_after = argparse.ArgumentParser(add_help=False)
    root_after.add_argument(
        "--root", default=argparse.SUPPRESS, help="Continuum data root (may also precede the subcommand)."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def _cmd(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[root_after], **kwargs)

    _cmd("init", help="Initialize a Continuum store.")
    _cmd("status", help="Show local Continuum status.")

    doctor = _cmd("doctor", help="Inspect root attachment, contracts, and optional companion packages.")
    doctor.add_argument("--cwd", default="")

    remember = _cmd("remember", help="Store a continuity record.")
    remember.add_argument("text")
    remember.add_argument("--kind", default="note")
    remember.add_argument("--tag", action="append", default=[])
    remember.add_argument("--json", action="store_true")

    ingest = _cmd("ingest-file", help="Store a text file as a continuity record.")
    ingest.add_argument("path")
    ingest.add_argument("--kind", default="document")
    ingest.add_argument("--tag", action="append", default=[])
    ingest.add_argument("--json", action="store_true")

    search = _cmd("search", help="Search local continuity records.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--json", action="store_true")

    restore = _cmd("restore", help="Restore continuity from Codex session archives if present.")
    restore.add_argument("--summary", default="")
    restore.add_argument("--cwd", default="")
    restore.add_argument("--codex-home", default=None)
    restore.add_argument("--limit", type=int, default=3)
    restore.add_argument("--scan-limit", type=int, default=80)
    restore.add_argument("--json", action="store_true")

    packet = _cmd("packet", help="Emit a full bootstrap packet.")
    packet.add_argument("--summary", default="")
    packet.add_argument("--cwd", default="")
    packet.add_argument("--codex-home", default=None)
    packet.add_argument("--scan-limit", type=int, default=48)
    packet.add_argument("--json", action="store_true")

    receipt = _cmd("receipt", help="Write a structured receipt for a meaningful action.")
    receipt.add_argument("action")
    receipt.add_argument("--summary", default="")
    receipt.add_argument("--status", default="ok")
    receipt.add_argument("--tag", action="append", default=[])
    receipt.add_argument("--json", action="store_true")

    handoff = _cmd("handoff", help="Emit browser/no-tools handoff text plus the backing packet.")
    handoff.add_argument("--summary", default="")
    handoff.add_argument("--cwd", default="")
    handoff.add_argument("--audience", default="browser", choices=["browser", "tiny", "codex", "portable"])
    handoff.add_argument("--codex-home", default=None)
    handoff.add_argument("--scan-limit", type=int, default=24)
    handoff.add_argument("--json", action="store_true")

    export = _cmd("export", help="Export local records to JSON.")
    export.add_argument("--output", default="")

    import_cmd = _cmd("import", help="Import records from a Continuum JSON export.")
    import_cmd.add_argument("source")
    import_cmd.add_argument("--json", action="store_true")

    system = _cmd("system-prompt", help="Print a model-facing Continuum system prompt.")
    system.add_argument("--profile", default="portable", choices=list_profiles())
    system.add_argument("--json", action="store_true")

    process = _cmd("process", help="Execute continuum commands a tool-less agent emitted in its text.")
    process.add_argument("--input", default="", help="File holding the agent's text (default: read stdin).")
    process.add_argument("--json", action="store_true")

    engine = _cmd("engine", help="Reconstitute or launch a bundled Champion Council backend payload.")
    engine.add_argument("--payload", required=True)
    engine.add_argument("--target", default="engine/champion-council")
    engine.add_argument("--overwrite", action="store_true")
    engine.add_argument("--self-test", action="store_true")
    engine.add_argument("--launch", action="store_true")
    engine.add_argument("--web-port", type=int, default=7866)
    engine.add_argument("--mcp-port", type=int, default=8766)
    engine.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_path = Path(args.root).resolve()
    continuum = Continuum(root_path)

    try:
        if args.command == "init":
            _print(continuum.init(), True)
        elif args.command == "status":
            _print(continuum.status(), True)
        elif args.command == "doctor":
            _print(continuum.doctor(cwd=args.cwd), True)
        elif args.command == "remember":
            _print(continuum.remember(args.text, kind=args.kind, tags=args.tag), args.json)
        elif args.command == "ingest-file":
            _print(continuum.ingest_file(args.path, kind=args.kind, tags=args.tag), args.json)
        elif args.command == "search":
            _print(continuum.search(args.query, limit=args.limit), True)
        elif args.command == "restore":
            _print(
                continuum.restore(
                    summary=args.summary,
                    cwd=args.cwd,
                    codex_home=args.codex_home,
                    limit=args.limit,
                    scan_limit=args.scan_limit,
                ),
                True,
            )
        elif args.command == "packet":
            _print(
                continuum.packet(
                    summary=args.summary,
                    cwd=args.cwd,
                    codex_home=args.codex_home,
                    scan_limit=args.scan_limit,
                ),
                True,
            )
        elif args.command == "receipt":
            _print(
                continuum.receipt(
                    action=args.action,
                    summary=args.summary,
                    status=args.status,
                    tags=args.tag,
                ),
                True,
            )
        elif args.command == "handoff":
            handoff = continuum.handoff(
                summary=args.summary,
                cwd=args.cwd,
                audience=args.audience,
                codex_home=args.codex_home,
                scan_limit=args.scan_limit,
            )
            _print(handoff, True) if args.json else _emit(handoff["markdown"])
        elif args.command == "export":
            _print({"output": str(continuum.store.export_json(args.output or None))}, True)
        elif args.command == "import":
            _print(continuum.store.import_json(args.source), True)
        elif args.command == "system-prompt":
            prompt = get_system_prompt(args.profile)
            _print({"profile": args.profile, "prompt": prompt}, True) if args.json else _emit(prompt)
        elif args.command == "process":
            from .processor import process_text

            text = Path(args.input).read_text(encoding="utf-8", errors="replace") if args.input else sys.stdin.read()
            result = process_text(text, root=root_path)
            _print(result, True) if args.json else _emit(result["rendered"])
        elif args.command == "engine":
            if args.self_test:
                result = engine_self_test(args.payload, args.target)
            else:
                result = reconstitute_engine(args.payload, args.target, overwrite=args.overwrite)
            if args.launch:
                proc = launch_engine(
                    Path(result["runtime_shell"]),
                    web_port=args.web_port,
                    mcp_port=args.mcp_port,
                )
                result["launch"] = {"pid": proc.pid, "web_port": args.web_port, "mcp_port": args.mcp_port}
            _print(result, True)
        return 0
    except Exception as exc:
        _print(
            {
                "error": "runtime_exception",
                "command": getattr(args, "command", ""),
                "message": str(exc),
                "failure_classes": ["truth", "contract", "transport", "rendering", "gating", "stale_runtime_state"],
            },
            True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
