from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9_./:-]{2,}", re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "again", "because", "before", "being", "could", "every",
    "from", "have", "here", "into", "just", "like", "make", "more", "need",
    "only", "other", "over", "please", "same", "should", "some", "state",
    "still", "that", "their", "them", "then", "there", "these", "this",
    "those", "through", "what", "when", "where", "which", "while", "with",
    "would", "your",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(value: Any) -> list[str]:
    text = str(value or "").lower()
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.strip("._/-:")
        if len(token) >= 2 and token not in _STOPWORDS:
            out.append(token)
    return out


@dataclass
class MemoryRecord:
    id: str
    kind: str
    text: str
    tags: list[str]
    metadata: dict[str, Any]
    created_ms: int


class ContinuumStore:
    """Small local JSONL store for continuity records and packets."""

    def __init__(self, root: str | Path = ".continuum") -> None:
        self.root = Path(root)
        self.records_path = self.root / "records.jsonl"
        self.packet_dir = self.root / "packets"
        self.exports_dir = self.root / "exports"
        self.meta_path = self.root / "continuum_meta.json"

    def initialize(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.packet_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        if not self.records_path.exists():
            self.records_path.write_text("", encoding="utf-8")
        meta = {
            "schema": "champion-continuum/v1",
            "created_ms": _now_ms(),
            "records_path": str(self.records_path),
            "packet_dir": str(self.packet_dir),
        }
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return self.status()

    def status(self) -> dict[str, Any]:
        records = list(self.iter_records()) if self.records_path.exists() else []
        initialized = self.records_path.exists()
        status = {
            "root": str(self.root.resolve()),
            "initialized": initialized,
            "record_count": len(records),
            "packet_count": len(list(self.packet_dir.glob("*.json"))) if self.packet_dir.exists() else 0,
            "latest_record_ms": max((r.created_ms for r in records), default=0),
        }
        if not initialized:
            status["next_step"] = "run continuum init for this root"
        return status

    def remember(
        self,
        text: str,
        kind: str = "note",
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self.initialize()
        created_ms = _now_ms()
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        record_id = sha256(
            json.dumps(
                {
                    "kind": kind,
                    "text": text,
                    "tags": clean_tags,
                    "created_ms": created_ms,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        record = MemoryRecord(
            id=record_id,
            kind=str(kind or "note"),
            text=str(text or ""),
            tags=clean_tags,
            metadata=dict(metadata or {}),
            created_ms=created_ms,
        )
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def iter_records(self) -> Iterable[MemoryRecord]:
        if not self.records_path.exists():
            return
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield MemoryRecord(
                    id=str(data.get("id") or ""),
                    kind=str(data.get("kind") or "note"),
                    text=str(data.get("text") or ""),
                    tags=list(data.get("tags") or []),
                    metadata=dict(data.get("metadata") or {}),
                    created_ms=int(data.get("created_ms") or 0),
                )

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.iter_records():
            # Weighted token overlap
            score = 0.0
            
            # 1. Tags (Highest weight)
            tag_tokens = set(tokenize(" ".join(record.tags)))
            score += 2.0 * len(query_tokens & tag_tokens)
            
            # 2. Kind
            kind_tokens = set(tokenize(record.kind))
            score += 1.0 * len(query_tokens & kind_tokens)
            
            # 3. Metadata (Medium weight)
            meta_tokens = set(tokenize(json.dumps(record.metadata)))
            score += 1.5 * len(query_tokens & meta_tokens)
            
            # 4. Text (Base weight)
            text_tokens = set(tokenize(record.text))
            score += 1.0 * len(query_tokens & text_tokens)
            
            if score <= 0:
                # Fuzzy fallback: whole-string substring match
                q_lower = query.lower()
                if q_lower in record.text.lower() or q_lower in record.kind.lower():
                    score = 0.5
            
            if score > 0:
                # Recency boost (up to 0.5 for records in the last 24h)
                age_ms = max(0, _now_ms() - record.created_ms)
                day_ms = 24 * 60 * 60 * 1000
                recency_boost = 0.5 * max(0.0, (day_ms - age_ms) / day_ms)
                scored.append((score + recency_boost, record))
        
        scored.sort(key=lambda item: (item[0], item[1].created_ms), reverse=True)
        return [
            {
                "score": round(score, 4),
                **asdict(record),
            }
            for score, record in scored[: max(1, int(limit or 8))]
        ]

    def save_packet(self, packet: dict[str, Any], name: str = "continuity_packet") -> Path:
        self.initialize()
        stamp = _now_ms()
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "continuity_packet"
        path = self.packet_dir / f"{stamp}_{safe}.json"
        path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def export_json(self, output: str | Path | None = None) -> Path:
        self.initialize()
        path = Path(output) if output else self.exports_dir / f"continuum_export_{_now_ms()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "champion-continuum/export/v1",
            "exported_ms": _now_ms(),
            "status": self.status(),
            "records": [asdict(record) for record in self.iter_records()],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def import_json(self, source: str | Path) -> dict[str, Any]:
        self.initialize()
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, list):
            raise ValueError("Continuum import file has no records list")
        imported = 0
        with self.records_path.open("a", encoding="utf-8") as handle:
            for item in records:
                if not isinstance(item, dict):
                    continue
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                imported += 1
        return {"imported": imported, "status": self.status()}
