from __future__ import annotations

import json
import time
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


DRIFT_CLASSES = [
    "confirmed",
    "partly_confirmed",
    "mismatch",
    "stale_state",
    "gated",
    "no_archive_match",
]

FAILURE_CLASSES = [
    "truth",
    "contract",
    "transport",
    "rendering",
    "gating",
    "stale_runtime_state",
]

OPERATIONAL_LOOP = [
    "observe",
    "normalize",
    "derive_output_state",
    "name_drift",
    "patch_smallest_honest_slice",
    "publish_receipt",
    "re_read",
]

RECOVERY_QUESTIONS = [
    "What is the subject?",
    "What is the objective?",
    "What is the seam?",
    "What evidence is current?",
    "What drift is present?",
    "What is the next smallest honest read?",
]

AUTHORITY_ORDER = [
    "fresh user request",
    "live/local runtime reads",
    "project source and test output",
    "continuum memory",
    "archive continuity hints",
    "docs and cultural overlays",
]

COMPANION_PACKAGES = {
    "cascade-lattice": {
        "role": "optional provenance and diagnostic lattice",
        "canon_surface": "receipts",
        "import_names": ["cascade", "cascade_lattice"],
    },
    "quinesmith": {
        "role": "optional quine/capsule edit safety protocol",
        "canon_surface": "self-pack verification",
        "import_names": ["quinesmith"],
    },
    "brotology-field-guide": {
        "role": "optional operator-language field guide",
        "canon_surface": "human/model conveyance",
        "import_names": ["brotology_field_guide"],
    },
}


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_digest(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()[:length]


def companion_package_status() -> dict[str, dict[str, Any]]:
    """Inspect optional companion packages without importing their runtime modules."""

    out: dict[str, dict[str, Any]] = {}
    for package_name, info in COMPANION_PACKAGES.items():
        try:
            dist = importlib_metadata.distribution(package_name)
            meta = dist.metadata
            out[package_name] = {
                "installed": True,
                "version": dist.version,
                "role": info["role"],
                "canon_surface": info["canon_surface"],
                "summary": meta.get("Summary", ""),
                "home_page": meta.get("Home-page", ""),
                "import_names": list(info["import_names"]),
            }
        except importlib_metadata.PackageNotFoundError:
            out[package_name] = {
                "installed": False,
                "version": "",
                "role": info["role"],
                "canon_surface": info["canon_surface"],
                "summary": "",
                "home_page": "",
                "import_names": list(info["import_names"]),
            }
    return out


def build_query_thread(summary: str, cwd: str, root_status: dict[str, Any]) -> dict[str, Any]:
    objective = str(summary or "continuum reacclimation").strip() or "continuum reacclimation"
    subject_id = str(cwd or root_status.get("root") or "").strip()
    subject_kind = "cwd" if cwd else "continuum_root"
    subject_key = f"{subject_kind}:{subject_id}" if subject_id else "continuum_root:unknown"
    sequence_seed = f"{objective}|{subject_key}|{root_status.get('latest_record_ms', 0)}"
    sequence_id = f"continuum/query/{stable_digest(sequence_seed, 12)}"
    return {
        "sequence_id": sequence_id,
        "segment_id": f"status:{root_status.get('latest_record_ms', 0)}",
        "session_id": stable_digest(root_status.get("root", ""), 12),
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "subject_key": subject_key,
        "status": "active",
        "current_pivot_id": "portable_memory_alignment",
        "provenance": {
            "raw_input": "",
            "input_lang": "",
            "lattice_signature": ""
        },
        "agent_arbitration": {
            "gemini_tonal_filter": "",
            "codex_domain_filter": "",
            "consensus_accuracy_score": 0.0
        },
        "execution_plane": {
            "normalized_core": "",
            "consensus_lang": "en-US"
        },
        "priority_pivots": [
            {
                "pivot_id": "portable_memory_alignment",
                "label": "Portable Memory Alignment",
                "status": "active",
                "reason": "Attach the agent to one local memory store before widening context.",
            },
            {
                "pivot_id": "archive_live_drift_check",
                "label": "Archive / Live Drift Check",
                "status": "active",
                "reason": "Separate archive hints from current runtime truth.",
            },
        ],
        "objective_id": "continuum_reacclimation",
        "objective_label": "Continuum Reacclimation",
        "objective_seed": objective,
        "visible_read": _visible_read(root_status),
        "anchor_row_ids": [
            "continuum.initialized",
            "continuum.record_count",
            "continuum.packet_count",
            "archive_restore.status",
            "companion_packages.installed",
        ],
        "help_lane": [
            {"command": "continuum system-prompt --profile portable", "reason": "Load the portable operating contract."},
            {"command": "continuum doctor", "reason": "Inspect root attachment and optional companion packages."},
        ],
        "next_reads": [
            {"command": "continuum status", "reason": "Check local store attachment before trusting memory."},
            {"command": f"continuum search {json.dumps(objective)}", "reason": "Recover only objective-relevant memory."},
            {"command": "inspect local source/tests", "reason": "Current files outrank stored memory."},
        ],
        "raw_state_guardrail": "Do not treat raw archives, old transcripts, or docs as current truth until live/local evidence agrees.",
    }


def _visible_read(root_status: dict[str, Any]) -> str:
    initialized = bool(root_status.get("initialized"))
    records = int(root_status.get("record_count") or 0)
    packets = int(root_status.get("packet_count") or 0)
    if initialized:
        return f"Continuum root is initialized with {records} records and {packets} saved packets."
    return "Continuum root is not initialized yet; run init before expecting durable memory."


def classify_drift(
    root_status: dict[str, Any],
    archive_restore: dict[str, Any] | None,
    memory_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    archive_status = str((archive_restore or {}).get("status") or "not_run")
    if not root_status.get("initialized"):
        status = "gated"
        reason = "memory root is not initialized"
    elif archive_status == "no_archive_match":
        status = "no_archive_match"
        reason = "local memory is usable but no matching archive session was found"
    elif archive_status in {"ok", "not_run"} and memory_hits:
        status = "partly_confirmed"
        reason = "memory evidence exists; current source/runtime still decides truth"
    elif archive_status == "ok":
        status = "partly_confirmed"
        reason = "archive context exists but needs current corroboration"
    else:
        status = "confirmed"
        reason = "store is usable and no contradiction is visible"
    return {
        "status": status,
        "reason": reason,
        "archive_status": archive_status,
        "memory_hit_count": len(memory_hits),
        "classes": DRIFT_CLASSES,
    }


def build_output_state(
    summary: str,
    cwd: str,
    root_status: dict[str, Any],
    archive_restore: dict[str, Any] | None,
    memory_hits: list[dict[str, Any]],
    companions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query_thread = build_query_thread(summary, cwd, root_status)
    companion_report = companions or companion_package_status()
    installed = [name for name, item in companion_report.items() if item.get("installed")]
    drift = classify_drift(root_status, archive_restore, memory_hits)
    return {
        "packet_kind": "continuum_output_state",
        "summary": f"{query_thread['objective_label']} / {query_thread['current_pivot_id']} / {drift['status']}",
        "placement": {
            "subject_key": query_thread["subject_key"],
            "objective_id": query_thread["objective_id"],
            "objective_label": query_thread["objective_label"],
            "current_pivot_id": query_thread["current_pivot_id"],
            "evidence": query_thread["anchor_row_ids"],
        },
        "query_thread": query_thread,
        "equilibrium": {
            "band": "usable" if root_status.get("initialized") else "needs_init",
            "summary": _visible_read(root_status),
        },
        "drift": drift,
        "watch_board": {
            "band": "track",
            "signals": [
                f"root initialized={bool(root_status.get('initialized'))}",
                f"records={int(root_status.get('record_count') or 0)}",
                f"companions={','.join(installed) if installed else 'none'}",
            ],
            "alerts": [] if root_status.get("initialized") else ["memory root not initialized"],
        },
        "next_reads": query_thread["next_reads"],
        "receipts": {
            "required_after_change": True,
            "command": "continuum receipt <action> --summary <result>",
            "rule": "Every meaningful mutation should leave one structured receipt.",
        },
        "freshness": {
            "generated_ms": now_ms(),
            "latest_record_ms": int(root_status.get("latest_record_ms") or 0),
        },
        "confidence": {
            "band": "bounded",
            "score": 1 if root_status.get("initialized") else 0.5,
            "missing_sources": [] if root_status.get("initialized") else ["initialized memory root"],
        },
        "sources": {
            "continuum_store": {
                "ready": bool(root_status.get("initialized")),
                "root": str(root_status.get("root") or ""),
            },
            "archive_restore": {
                "ready": bool((archive_restore or {}).get("status") == "ok"),
                "status": str((archive_restore or {}).get("status") or "not_run"),
            },
            "companion_packages": companion_report,
        },
    }


def build_receipt(
    action: str,
    summary: str = "",
    status: str = "ok",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    root_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_ms = now_ms()
    payload = {
        "packet_kind": "continuum_receipt",
        "action": str(action or "unspecified"),
        "summary": str(summary or ""),
        "status": str(status or "ok"),
        "tags": [str(tag) for tag in (tags or []) if str(tag).strip()],
        "metadata": dict(metadata or {}),
        "root_status": dict(root_status or {}),
        "created_ms": created_ms,
    }
    payload["receipt_id"] = "receipt_" + stable_digest(payload, 20)
    payload["authority_order"] = AUTHORITY_ORDER
    return payload


def verify_integrity(cwd: str = "") -> dict[str, Any]:
    """Verify local files against CHECKSUMS.txt if present."""
    root = Path(cwd or ".").resolve()
    checksums_path = root / "CHECKSUMS.txt"
    if not checksums_path.exists():
        # Try one level up (common in drop-in bundles)
        checksums_path = root.parent / "CHECKSUMS.txt"
    
    if not checksums_path.exists():
        return {"status": "missing", "message": "CHECKSUMS.txt not found; integrity cannot be verified."}

    verified: list[str] = []
    mismatch: list[str] = []
    missing: list[str] = []
    
    try:
        with checksums_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or " " not in line:
                    continue
                expected_sha, rel_path = line.split(None, 1)
                file_path = checksums_path.parent / rel_path
                if not file_path.exists():
                    missing.append(rel_path)
                    continue
                
                actual_sha = sha256(file_path.read_bytes()).hexdigest()
                if actual_sha == expected_sha:
                    verified.append(rel_path)
                else:
                    mismatch.append(rel_path)
    except Exception as exc:
        return {"status": "error", "message": f"Integrity check failed: {exc}"}

    status = "ok" if not mismatch and not missing else "mismatch"
    return {
        "status": status,
        "verified_count": len(verified),
        "mismatch": mismatch,
        "missing": missing,
    }


def build_doctor_report(root_status: dict[str, Any], cwd: str = "") -> dict[str, Any]:
    companions = companion_package_status()
    integrity = verify_integrity(cwd)
    initialized = bool(root_status.get("initialized"))
    warnings: list[str] = []
    if not initialized:
        warnings.append("Continuum root is not initialized; memory commands will create it lazily, but explicit init is clearer.")
    
    if integrity["status"] == "mismatch":
        warnings.append(f"Integrity mismatch detected in {len(integrity['mismatch'])} file(s).")
    elif integrity["status"] == "missing":
        warnings.append(integrity["message"])

    missing_companions = [name for name, item in companions.items() if not item.get("installed")]
    if missing_companions:
        warnings.append("Optional companion package(s) not installed: " + ", ".join(missing_companions))
    
    return {
        "packet_kind": "continuum_doctor_report",
        "status": "ok" if initialized and integrity["status"] == "ok" else "needs_init_or_repair",
        "cwd": str(cwd or ""),
        "root_status": root_status,
        "companion_packages": companions,
        "integrity": integrity,
        "authority_order": AUTHORITY_ORDER,
        "operational_loop": OPERATIONAL_LOOP,
        "failure_classes": FAILURE_CLASSES,
        "recovery_questions": RECOVERY_QUESTIONS,
        "warnings": warnings,
        "next_actions": [
            "continuum init" if not initialized else "continuum packet --summary <objective>",
            "continuum handoff --summary <objective> for browser/no-tools agents",
            "continuum receipt <action> --summary <result> after meaningful changes",
        ],
    }


def build_browser_handoff(
    summary: str,
    packet: dict[str, Any],
    audience: str = "browser",
) -> str:
    root = str(packet.get("continuum_root") or "")
    drift = ((packet.get("output_state") or {}).get("drift") or {}).get("status", "unknown")
    companions = packet.get("companion_packages") or {}
    installed = [name for name, data in companions.items() if isinstance(data, dict) and data.get("installed")]
    command_root = root or ".continuum"
    lines = [
        "# Continuum Handoff",
        "",
        f"Audience: {audience}",
        f"Objective: {summary or packet.get('summary') or 'reacclimate with Continuum'}",
        f"Continuum root: {command_root}",
        f"Current drift class: {drift}",
        "",
        "## Known",
        "- Continuum is a local continuity memory primitive.",
        "- A memory packet is recovery context, not authority.",
        f"- Optional companions detected: {', '.join(installed) if installed else 'none detected'}.",
        "",
        "## Inferred",
        "- If this agent has a shell, it should initialize/search/packet before acting.",
        "- If this agent has no shell, it should treat visible files and this handoff as static evidence only.",
        "",
        "## Unknown",
        "- Current runtime truth is unknown until this environment runs local commands or exposes files.",
        "- Archive continuity is unknown until `continuum packet` or `continuum restore` runs in the target environment.",
        "",
        "## Safest Next Step",
        "If shell access exists:",
        f"1. `continuum --root {json.dumps(command_root)} init`",
        f"2. `continuum --root {json.dumps(command_root)} packet --summary {json.dumps(summary or 'reacclimate')}`",
        "3. Inspect current files/tests before editing.",
        "4. Save one receipt after any meaningful change.",
        "",
        "If no shell access exists:",
        "1. Read AGENT_READ_THIS_FIRST.md.",
        "2. Report known / inferred / unknown / safest next step.",
        "3. Do not claim install, launch, tests, or memory updates occurred.",
        "",
        "## Recovery Questions",
    ]
    lines.extend(f"- {question}" for question in RECOVERY_QUESTIONS)
    lines.extend(
        [
            "",
            "## Operating Loop",
            "Observe -> Normalize -> Derive output_state -> Name drift -> Patch smallest honest slice -> Publish receipt -> Re-read.",
        ]
    )
    return "\n".join(lines)
