from __future__ import annotations

import re
from hashlib import sha256
from typing import Any


SCHEMA = "champion-continuum/translation-faculty/v1"

FACULTY_MODELS: dict[str, list[dict[str, Any]]] = {
    "language_id": [
        {
            "id": "facebook/fasttext-language-identification",
            "role": "fast language detection",
            "load": "optional",
            "notes": "Useful before routing to ASR, MT, or council agents.",
        }
    ],
    "speech_to_text": [
        {
            "id": "openai/whisper-tiny",
            "role": "small local voice-note transcription",
            "load": "optional",
            "notes": "Good first Space/local candidate for WhatsApp audio drafts.",
        }
    ],
    "text_translation": [
        {
            "id": "facebook/nllb-200-distilled-600M",
            "role": "local multilingual translation baseline",
            "load": "optional",
            "notes": "Broad coverage; use as a draft baseline, then let the council polish.",
        },
        {
            "id": "Helsinki-NLP/opus-mt-en-zh",
            "role": "focused English-Chinese baseline",
            "load": "optional",
            "notes": "Useful for smoke tests and contact-specific phrase memory.",
        },
    ],
    "council_llm": [
        {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "role": "tiny scout and JSON packet drafter",
            "load": "already supported by app model menu",
            "notes": "Fastest cheap assistant for structure and guardrails.",
        },
        {
            "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "role": "lightweight wording critic",
            "load": "already supported by app model menu",
            "notes": "Small enough for Space experiments without changing architecture.",
        },
    ],
}


def _hash_text(value: str) -> str:
    return sha256((value or "").encode("utf-8")).hexdigest()


def parse_glossary(text: str) -> list[dict[str, str]]:
    """Parse simple operator-authored term locks without requiring a file format."""
    out: list[dict[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            source, target = line.split("=>", 1)
        elif "=" in line:
            source, target = line.split("=", 1)
        elif ":" in line:
            source, target = line.split(":", 1)
        else:
            source, target = line, ""
        source = source.strip()
        target = target.strip()
        if source:
            out.append({"source": source, "target": target})
    return out[:64]


def glossary_hits(raw_content: str, glossary_terms: list[dict[str, str]]) -> list[dict[str, str]]:
    raw_lower = (raw_content or "").lower()
    hits: list[dict[str, str]] = []
    for term in glossary_terms:
        source = term.get("source", "")
        if source and re.search(rf"\b{re.escape(source.lower())}\b", raw_lower):
            hits.append(term)
    return hits


def translation_faculty_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "local_first_with_provider_slots",
        "external_effects_performed": False,
        "space_target": "https://huggingface.co/spaces/tostido/champion-continuum",
        "faculties": FACULTY_MODELS,
        "quality_loop": [
            "preserve_raw",
            "resident_small_model_relevance_scan",
            "detect_language",
            "apply_contact_profile",
            "apply_glossary",
            "draft_target_message",
            "literal_back_translate",
            "council_dissent_if_drift",
            "escalate_to_premier_provider_or_cli_agent_only_when_relevant",
            "operator_approval",
        ],
        "space_preload_posture": {
            "enabled_by_default": False,
            "reason": "Keep the public Space deployable on CPU. Preload models only after a runtime budget decision.",
            "recommended_first": [
                "Qwen/Qwen2.5-0.5B-Instruct",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "openai/whisper-tiny",
                "facebook/nllb-200-distilled-600M",
            ],
        },
    }


def build_translation_faculty_packet(
    raw_content: str,
    source_lang: str,
    target_lang: str,
    conversation_profile: str,
    glossary_text: str,
    provider_plan: str,
) -> dict[str, Any]:
    glossary_terms = parse_glossary(glossary_text)
    return {
        "schema": SCHEMA,
        "raw_sha256": _hash_text(raw_content),
        "source_lang": source_lang or "auto",
        "target_lang": target_lang or "en-US",
        "provider_plan": provider_plan or "council-first",
        "conversation_profile": {
            "present": bool((conversation_profile or "").strip()),
            "sha256": _hash_text(conversation_profile or ""),
            "text": conversation_profile or "",
        },
        "glossary": {
            "term_count": len(glossary_terms),
            "terms": glossary_terms,
            "hits": glossary_hits(raw_content, glossary_terms),
        },
        "faculty_state": translation_faculty_state(),
        "external_effects_performed": False,
    }
