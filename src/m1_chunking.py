"""Load every supported corpus file and split it without discarding source text.

Chunk budgets are whitespace-delimited words, not model tokens. Splits prefer
Markdown headings, paragraphs, then sentences; exact character offsets allow
the source to be reconstructed. PDF extraction needs a text layer, not OCR.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE


@dataclass
class Chunk:
    text: str
    metadata: dict
    parent_id: str = ""
    chunk_id: str = ""


_HEADER = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_FIELDS = {
    "version": "Phiên bản",
    "effective_date": "Ngày hiệu lực",
    "department": "Phòng ban",
    "status": "Trạng thái",
}


def _document_metadata(text: str, source: str, *, title: str = "") -> dict:
    heading = _HEADER.search(text)
    metadata = {
        "source": source,
        "title": heading.group(2).strip() if heading else (title or Path(source).stem),
        "version": "unspecified",
        "file_type": Path(source).suffix.lower().lstrip("."),
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    for key, label in _FIELDS.items():
        match = re.search(rf"{label}\s*:\s*([^|\r\n]+)", text, re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()
    return metadata


def _read_ocr_sidecar(path: Path, page_count: int) -> dict | None:
    """Accept only an explicit, complete OCR result bound to these PDF bytes."""
    sidecar = path.with_name(path.name + ".ocr.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("root must be an object")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if data.get("source_sha256") != actual_hash:
            raise ValueError("source_sha256 does not match the PDF (stale OCR)")
        if not isinstance(data.get("engine"), str) or not data["engine"].strip():
            raise ValueError("engine is required")
        languages = data.get("recognition_languages")
        if not isinstance(languages, list) or not languages or not all(isinstance(item, str) and item.strip() for item in languages):
            raise ValueError("recognition_languages must be a nonempty string list")
        pages = data.get("pages")
        if not isinstance(pages, list) or len(pages) != page_count:
            raise ValueError("pages must contain every PDF page exactly once in order")
        for number, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or type(page.get("page")) is not int or page["page"] != number:
                raise ValueError("pages must contain every PDF page exactly once in order")
            if not isinstance(page.get("text"), str) or not page["text"].strip():
                raise ValueError(f"OCR page {number} has no text; inspect/redo OCR")
            confidence = page.get("mean_confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"OCR page {number} needs a finite mean_confidence in [0,1]")
        return data
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid OCR sidecar {sidecar.name}: {exc}") from exc


def load_documents(data_dir=None, *, pdf_reader_factory=None) -> list[dict]:
    """Read all .md/.pdf recursively; a failed/empty document is a visible error.

    ``pdf_reader_factory`` is an injection point for offline tests. Empty PDF
    pages are reported in metadata and a warning, never silently dropped. A
    wholly image-only PDF raises an OCR-required error instead of disappearing.
    A validated ``<file>.pdf.ocr.json`` supplies real OCR for empty pages only;
    its PDF hash, engine, languages, complete page order/text/confidence are
    checked before any OCR text is used. Original PDFs are never modified.
    """
    root = Path(DATA_DIR if data_dir is None else data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".pdf"})
    if not paths:
        raise ValueError(f"No Markdown or PDF documents found in {root}")
    documents = []
    for path in paths:
        source = path.relative_to(root).as_posix()
        extra = {}
        title = ""
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
        else:
            if pdf_reader_factory is None:
                try:
                    from pypdf import PdfReader
                except ImportError as exc:
                    raise ImportError("pypdf is required to load every corpus PDF; install requirements/requirements.txt") from exc
                pdf_reader_factory = PdfReader
            try:
                reader = pdf_reader_factory(str(path))
                if getattr(reader, "is_encrypted", False) and not reader.decrypt(""):
                    raise ValueError("password-protected PDF")
                pages = [page.extract_text() or "" for page in reader.pages]
                pdf_metadata = getattr(reader, "metadata", None)
                if pdf_metadata and isinstance(getattr(pdf_metadata, "title", None), str):
                    title = pdf_metadata.title.strip()
                empty_pages = [i + 1 for i, page in enumerate(pages) if not page.strip()]
                extra = {"page_count": len(pages), "empty_text_pages": empty_pages, "extraction_method": "pypdf"}
                if empty_pages:
                    ocr = _read_ocr_sidecar(path, len(pages))
                    if ocr is not None:
                        for page_number in empty_pages:
                            pages[page_number - 1] = ocr["pages"][page_number - 1]["text"]
                        extra.update({
                            "extraction_method": "pypdf+ocr",
                            "ocr_source": source + ".ocr.json", "ocr_engine": ocr["engine"],
                            "ocr_pages": empty_pages,
                            "ocr_recognition_languages": ocr["recognition_languages"],
                            "ocr_mean_confidence": {
                                str(i): ocr["pages"][i - 1]["mean_confidence"] for i in empty_pages
                            },
                        })
                        empty_pages = []
                text = "\n\n".join(pages)
            except Exception as exc:
                raise ValueError(f"Could not read PDF {source}: {exc}") from exc
            if not text.strip():
                raise ValueError(f"PDF {source} has no extractable text; OCR is required before indexing")
            if empty_pages:
                warnings.warn(
                    f"PDF {source}: no extracted text on pages {empty_pages}; inspect whether OCR is needed",
                    UserWarning,
                    stacklevel=2,
                )
        if not text.strip():
            raise ValueError(f"Corpus document is empty: {source}")
        metadata = {**_document_metadata(text, source, title=title), **extra}
        documents.append({"text": text, "metadata": metadata})
    return documents


def _split_spans(text: str, budget: int) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    spans, first_word, start = [], 0, 0
    while first_word < len(words):
        stop_word = min(first_word + budget, len(words))
        if stop_word == len(words):
            end = len(text)
        else:
            # Keep at least half a chunk, then choose the latest strong boundary.
            lower = first_word + max(1, budget // 2)
            preferred = None
            for candidate in range(stop_word, lower - 1, -1):
                gap = text[words[candidate - 1].end():words[candidate].start()]
                is_heading = words[candidate].group().startswith("#") and "\n" in gap
                if gap.count("\n") >= 2 or is_heading:
                    preferred = candidate
                    break
            if preferred is None:
                preferred = next((
                    candidate for candidate in range(stop_word, lower - 1, -1)
                    if re.search(r"[.!?…][\"')\]]?$", words[candidate - 1].group())
                ), stop_word)
            stop_word = preferred
            end = words[stop_word].start()
        spans.append((start, end))
        start, first_word = end, stop_word
    return spans


def _section_metadata(text: str, start: int, end: int) -> dict:
    stack = []
    contained = []
    for heading in _HEADER.finditer(text):
        if heading.start() >= end:
            break
        label = heading.group(2).strip()
        if heading.start() <= start:
            level = len(heading.group(1))
            stack = [(depth, item) for depth, item in stack if depth < level]
            stack.append((level, label))
        if start <= heading.start() < end:
            contained.append(label)
    return {"section_path": [item for _, item in stack], "headings": contained}


def chunk_hierarchical(text: str, metadata=None, *, parent_size=None, child_size=None):
    """Return stable parent/child chunks, retaining each original byte of text.

    ``char_start``/``char_end`` are offsets into the input string. Parent and
    child IDs include document content/source and offsets, so repeated runs are
    stable and changed documents cannot reuse stale chunk IDs.
    """
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")
    parent_size = HIERARCHICAL_PARENT_SIZE if parent_size is None else parent_size
    child_size = HIERARCHICAL_CHILD_SIZE if child_size is None else child_size
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in (parent_size, child_size)):
        raise ValueError("Parent/child word budgets must be positive integers")
    if child_size > parent_size:
        raise ValueError("Child word budget cannot exceed parent word budget")
    metadata = deepcopy(dict(metadata or {}))
    document_id = hashlib.sha256((str(metadata.get("source", "document")) + "\0" + text).encode("utf-8")).hexdigest()
    parents, children = [], []
    for parent_number, (start, end) in enumerate(_split_spans(text, parent_size)):
        parent_id = hashlib.sha256(f"{document_id}:parent:{start}:{end}".encode()).hexdigest()
        parent_metadata = {
            **deepcopy(metadata), **_section_metadata(text, start, end),
            "document_id": document_id, "chunk_id": parent_id,
            "char_start": start, "char_end": end, "parent_number": parent_number,
        }
        parents.append(Chunk(text[start:end], parent_metadata, parent_id, parent_id))
        for child_number, (child_start, child_end) in enumerate(_split_spans(text[start:end], child_size)):
            absolute_start, absolute_end = start + child_start, start + child_end
            chunk_id = hashlib.sha256(f"{document_id}:child:{absolute_start}:{absolute_end}".encode()).hexdigest()
            child_metadata = {
                **deepcopy(metadata), **_section_metadata(text, absolute_start, absolute_end),
                "document_id": document_id, "chunk_id": chunk_id, "parent_id": parent_id,
                "char_start": absolute_start, "char_end": absolute_end,
                "parent_number": parent_number, "child_number": child_number,
            }
            children.append(Chunk(text[absolute_start:absolute_end], child_metadata, parent_id, chunk_id))
    return parents, children
