"""Deterministic, source-grounded enrichment; no LLM claims or hidden fallback."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib


@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict


def enrich_chunks(chunks) -> list[EnrichedChunk]:
    """Prefix each unchanged passage with its available provenance/context.

    Count/order and original text are preserved. Version/status come only from
    source metadata; older policies are labelled, not removed or rewritten.
    """
    result = []
    for ordinal, chunk in enumerate(chunks):
        text = chunk["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {ordinal} must contain non-empty source text")
        metadata = deepcopy(dict(chunk.get("metadata", {})))
        headers = []
        for key, label in (
            ("source", "Nguồn"), ("title", "Tài liệu"), ("version", "Phiên bản"),
            ("effective_date", "Ngày hiệu lực"), ("department", "Phòng ban"),
            ("status", "Trạng thái"),
        ):
            value = metadata.get(key)
            if value and value != "unspecified":
                headers.append(f"{label}: {value}")
        sections = list(dict.fromkeys(metadata.get("section_path", []) + metadata.get("headings", [])))
        if sections:
            headers.append("Mục: " + " > ".join(sections))
        if headers:
            enriched_text = "[Thông tin nguồn]\n" + "\n".join(headers) + "\n[/Thông tin nguồn]\n\n" + text
        else:
            enriched_text = text
        result.append(EnrichedChunk(enriched_text, {
            **metadata,
            "raw_text": text,
            "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "enrichment_method": "source_metadata_v1",
            "enrichment_ordinal": ordinal,
        }))
    return result
