import io
import json
from pathlib import Path

from champion_continuum import Continuum
from champion_continuum.system_prompts import get_system_prompt, list_profiles


def test_store_search_and_packet(tmp_path: Path) -> None:
    continuum = Continuum(tmp_path / ".continuum")
    status = continuum.init()
    assert status["initialized"] is True

    continuum.remember("Continuity restore is archive-side reacclimation only.", tags=["doctrine"])
    hits = continuum.search("archive continuity reacclimation")
    assert hits
    assert hits[0]["tags"] == ["doctrine"]

    packet = continuum.packet(summary="resume continuity package", cwd=str(tmp_path), scan_limit=4)
    assert packet["packet_kind"] == "continuum_bootstrap_packet"
    assert packet["archive_restore"]["status"] in {"ok", "no_archive_match"}


def test_export_import(tmp_path: Path) -> None:
    src = Continuum(tmp_path / "src")
    src.init()
    src.remember("Portable memory survives context compression.", tags=["portable"])
    export_path = src.store.export_json(tmp_path / "continuum_export.json")

    dst = Continuum(tmp_path / "dst")
    result = dst.store.import_json(export_path)
    assert result["imported"] == 1
    assert dst.search("context compression")


def test_short_tags_are_searchable(tmp_path: Path) -> None:
    continuum = Continuum(tmp_path / ".continuum")
    continuum.init()
    continuum.remember("Short tag test", tags=["v2"])
    assert continuum.search("v2")


def test_system_prompts() -> None:
    assert "tiny" in list_profiles()
    assert "Answer memory claims from packets or current evidence" in get_system_prompt("tiny")


def test_root_flag_accepts_either_position(monkeypatch) -> None:
    monkeypatch.delenv("CONTINUUM_ROOT", raising=False)
    from champion_continuum.cli import build_parser

    assert build_parser().parse_args(["--root", "A", "init"]).root == "A"
    assert build_parser().parse_args(["init", "--root", "B"]).root == "B"
    assert build_parser().parse_args(["remember", "text", "--root", "C"]).root == "C"
    assert build_parser().parse_args(["remember", "--root", "D", "text"]).root == "D"
    assert build_parser().parse_args(["--root", "A", "remember", "--root", "B", "text"]).root == "B"
    assert build_parser().parse_args(["init"]).root == ".continuum"


def test_root_env_default(monkeypatch) -> None:
    from champion_continuum.cli import build_parser

    monkeypatch.setenv("CONTINUUM_ROOT", "/tmp/pinned")
    assert build_parser().parse_args(["status"]).root == "/tmp/pinned"
    # An explicit --root still wins over the env default, in either position.
    assert build_parser().parse_args(["--root", "X", "status"]).root == "X"
    assert build_parser().parse_args(["status", "--root", "Y"]).root == "Y"


def test_packet_reports_its_root(tmp_path: Path) -> None:
    continuum = Continuum(tmp_path / ".continuum")
    continuum.init()
    packet = continuum.packet(summary="root visibility", cwd=str(tmp_path), scan_limit=4)
    assert packet["continuum_root"] == str((tmp_path / ".continuum").resolve())


def test_cli_main_resolves_relative_root_to_absolute(tmp_path: Path, monkeypatch, capsys) -> None:
    from champion_continuum.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["--root", "relative-root", "packet", "--summary", "absolute root"]) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["continuum_root"] == str((tmp_path / "relative-root").resolve())


def test_doctor_receipt_and_handoff(tmp_path: Path) -> None:
    continuum = Continuum(tmp_path / ".continuum")
    doctor_before = continuum.doctor(cwd=str(tmp_path))
    assert doctor_before["status"] == "needs_init_or_repair"

    continuum.init()
    doctor_after = continuum.doctor(cwd=str(tmp_path))
    # Status remains 'needs_init_or_repair' because CHECKSUMS.txt is missing in tmp_path
    assert doctor_after["status"] == "needs_init_or_repair"
    assert doctor_after["integrity"]["status"] == "missing"
    assert "quinesmith" in doctor_after["companion_packages"]

    receipt = continuum.receipt("test_action", summary="test_summary")
    assert receipt["packet_kind"] == "continuum_receipt"
    assert receipt["action"] == "test_action"

    handoff = continuum.handoff(summary="test_handoff", audience="tiny")
    assert handoff["packet_kind"] == "continuum_handoff"
    assert "test_handoff" in handoff["markdown"]


def test_cli_new_commands_parse() -> None:
    from champion_continuum.cli import build_parser

    assert build_parser().parse_args(["doctor"]).command == "doctor"
    assert build_parser().parse_args(["receipt", "act"]).command == "receipt"
    assert build_parser().parse_args(["handoff"]).command == "handoff"
    assert build_parser().parse_args(["handoff", "--audience", "tiny"]).audience == "tiny"


def test_processor_remember_and_search(tmp_path: Path) -> None:
    from champion_continuum import process_text

    root = tmp_path / ".continuum"
    text = "thinking... [[continuum: remember | deploy web port is 7866 | tags=deploy,ports]] ok"
    result = process_text(text, root=root)
    assert result["command_count"] == 1
    assert result["results"][0]["ok"] is True

    recall = process_text("[[continuum: search | port]]", root=root)
    hits = recall["results"][0]["hits"]
    assert any("7866" in h["text"] for h in hits)
    assert "[[continuum-results]]" in recall["rendered"]
    assert "7866" in recall["rendered"]


def test_processor_ignores_non_commands(tmp_path: Path) -> None:
    from champion_continuum import process_text

    result = process_text("no commands here, just prose.", root=tmp_path / ".continuum")
    assert result["command_count"] == 0
    assert "no continuum commands" in result["rendered"]


def test_relay_profile_and_cli_process(monkeypatch, capsys, tmp_path: Path) -> None:
    from champion_continuum.system_prompts import get_system_prompt, list_profiles
    from champion_continuum.cli import build_parser, main

    assert "relay" in list_profiles()
    assert "[[continuum: remember" in get_system_prompt("relay")
    assert build_parser().parse_args(["process"]).command == "process"

    monkeypatch.setattr("sys.stdin", io.StringIO("[[continuum: remember | hello relay | tags=t]]"))
    assert main(["--root", str(tmp_path / ".continuum"), "process", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["command_count"] == 1


def _seed_tool(continuum, server, name, desc, args):
    continuum.store.remember(
        text=f"{server}.{name} {desc} args: {', '.join(args)}",
        kind="mcp_tool",
        tags=[server],
        metadata={"server": server, "name": name, "description": desc, "args": args, "input_schema": {}},
    )


def test_tool_surface_summary_and_search(tmp_path: Path) -> None:
    from champion_continuum import Continuum

    c = Continuum(tmp_path / ".continuum")
    _seed_tool(c, "council", "web_search", "Search the web for a query", ["query"])
    _seed_tool(c, "council", "deliberate", "Run a council debate", ["topic"])

    summary = c.indexed_tool_summary()
    assert summary["count"] == 2
    assert summary["servers"] == ["council"]

    hits = c.search_tools("search the web")
    assert any(h["name"] == "web_search" for h in hits)


def test_index_mcp_tools_reports_total_for_ui_trace(tmp_path: Path) -> None:
    from champion_continuum import Continuum

    c = Continuum(tmp_path / ".continuum")
    c.list_mcp_tools = lambda: [
        {
            "server": "council",
            "name": "web_search",
            "description": "Search the web for a query",
            "input_schema": {"properties": {"query": {"type": "string"}}},
        }
    ]

    indexed = c.index_mcp_tools()
    assert indexed["discovered"] == 1
    assert indexed["total"] == 1
    assert indexed["indexed_new"] == 1


def test_processor_tools_search(tmp_path: Path) -> None:
    from champion_continuum import Continuum, process_text

    c = Continuum(tmp_path / ".continuum")
    _seed_tool(c, "council", "web_search", "Search the web", ["query"])

    out = process_text("let me look [[tools: search | web]]", root=c.store.root)
    assert out["command_count"] == 1
    assert "web_search" in out["rendered"]
    assert "[[tool: council.web_search" in out["rendered"]


def test_strip_results_removes_fabrications() -> None:
    from champion_continuum import strip_results

    text = (
        "Sure. [[tool: x.y | a=]]\n"
        "[[continuum-results]]\n- fake brain_weights.txt result\n[[/continuum-results]]\n"
        "done [[/tool: x.y]]"
    )
    out = strip_results(text)
    assert "continuum-results" not in out
    assert "brain_weights" not in out
    assert "[[/tool" not in out
    assert "[[tool: x.y | a=]]" in out  # the genuine request is preserved
