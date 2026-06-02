from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def _copytree_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def reconstitute_engine(
    payload: str | Path,
    target: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    payload_path = Path(payload)
    target_path = Path(target)
    if not payload_path.exists():
        raise FileNotFoundError(f"payload not found: {payload_path}")
    if target_path.exists() and overwrite:
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(payload_path, "r") as archive:
        archive.extractall(target_path)

    product_root = _find_product_root(target_path)
    runtime_shell = product_root / "runtime_shell"
    seed_state = product_root / "seed_state"
    if not runtime_shell.exists():
        raise FileNotFoundError(f"runtime_shell not found inside payload: {product_root}")
    if seed_state.exists():
        data_root = runtime_shell / "data" / "champion-council-state"
        _copytree_contents(seed_state, data_root)

    return {
        "status": "ok",
        "target": str(target_path.resolve()),
        "product_root": str(product_root.resolve()),
        "runtime_shell": str(runtime_shell.resolve()),
        "seed_state_copied": seed_state.exists(),
        "run_local": str((runtime_shell / "run_local.ps1").resolve()),
    }


def engine_self_test(payload: str | Path, target: str | Path) -> dict[str, Any]:
    result = reconstitute_engine(payload=payload, target=target, overwrite=False)
    runtime_shell = Path(result["runtime_shell"])
    checks = {
        "server_py": (runtime_shell / "server.py").exists(),
        "requirements_txt": (runtime_shell / "requirements.txt").exists(),
        "capsule_gz": (runtime_shell / "capsule" / "capsule.gz").exists(),
        "seed_bag": (runtime_shell / "data" / "champion-council-state" / "bag" / "bag_export.json").exists(),
    }
    result["checks"] = checks
    result["self_test_ok"] = all(checks.values())
    return result


def launch_engine(
    runtime_shell: str | Path,
    host: str = "127.0.0.1",
    web_port: int = 7866,
    mcp_port: int = 8766,
) -> subprocess.Popen:
    shell = Path(runtime_shell)
    env = os.environ.copy()
    env.setdefault("WEB_HOST", host)
    env.setdefault("WEB_PORT", str(web_port))
    env.setdefault("MCP_PORT", str(mcp_port))
    env.setdefault("APP_MODE", "development")
    env.setdefault("MCP_EXTERNAL_POLICY", "full")
    env.setdefault("PERSISTENCE_MODE", "local")
    env.setdefault("PERSISTENCE_DATA_DIR", str(shell / "data" / "champion-council-state"))
    if os.name == "nt":
        command = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(shell / "run_local.ps1"),
        ]
    else:
        command = [sys.executable, "server.py"]
    return subprocess.Popen(command, cwd=str(shell), env=env)


def _find_product_root(target: Path) -> Path:
    if (target / "product_manifest.json").exists():
        return target
    matches = list(target.glob("*/product_manifest.json"))
    if matches:
        return matches[0].parent
    return target

