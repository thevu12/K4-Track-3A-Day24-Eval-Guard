"""Offline contracts for the actual RAG modules, not the setup adapter mocks."""
from copy import deepcopy
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch, _tokenize
from src.m3_rerank import CrossEncoderReranker
from src.m5_enrichment import enrich_chunks


def _pdf_reader(pages):
    return SimpleNamespace(
        is_encrypted=False,
        metadata=SimpleNamespace(title="PDF policy"),
        pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages],
    )


def test_loader_reads_markdown_and_pdf_recursively_with_metadata(tmp_path):
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "leave.md").write_text(
        "# Nghỉ phép\n> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự\n\n15 ngày phép.",
        encoding="utf-8",
    )
    (tmp_path / "safety.pdf").touch()
    docs = load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader(["An toàn.", "Bảo mật."]))
    assert len(docs) == 2
    assert docs[0]["metadata"]["source"] == "policies/leave.md"
    assert docs[0]["metadata"]["version"] == "2.0"
    assert docs[0]["metadata"]["title"] == "Nghỉ phép"
    assert docs[0]["metadata"]["effective_date"] == "01/01/2024"
    assert docs[1]["text"] == "An toàn.\n\nBảo mật."
    assert docs[1]["metadata"]["title"] == "PDF policy"
    assert docs[1]["metadata"]["version"] == "unspecified"
    assert docs[1]["metadata"]["page_count"] == 2


def test_loader_does_not_silently_skip_scanned_or_broken_pdf(tmp_path):
    (tmp_path / "scan.pdf").touch()
    with pytest.raises(ValueError, match="OCR is required"):
        load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader([None, ""]))
    def broken(_):
        raise OSError("invalid PDF")
    with pytest.raises(ValueError, match="scan.pdf.*invalid PDF"):
        load_documents(tmp_path, pdf_reader_factory=broken)


def test_loader_reports_partial_pdf_text_loss(tmp_path):
    (tmp_path / "mixed.pdf").touch()
    with pytest.warns(UserWarning, match="pages \\[2\\]"):
        docs = load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader(["Readable", ""]))
    assert docs[0]["metadata"]["empty_text_pages"] == [2]


def _ocr_sidecar(pdf, pages):
    record = {
        "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "engine": "Apple Vision VNRecognizeTextRequest",
        "recognition_languages": ["vi-VT", "en-US"],
        "pages": [{"page": i, "text": text, "mean_confidence": 0.99} for i, text in enumerate(pages, 1)],
    }
    sidecar = pdf.with_name(pdf.name + ".ocr.json")
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    return sidecar, record


def test_loader_uses_validated_ocr_only_on_blank_pdf_pages(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"pdf bytes")
    _ocr_sidecar(pdf, ["OCR first page", "OCR tiếng Việt trang hai"])
    docs = load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader(["Keep original page", ""]))
    assert docs[0]["text"] == "Keep original page\n\nOCR tiếng Việt trang hai"
    assert docs[0]["metadata"]["extraction_method"] == "pypdf+ocr"
    assert docs[0]["metadata"]["ocr_source"] == "scan.pdf.ocr.json"
    assert docs[0]["metadata"]["ocr_pages"] == [2]
    assert docs[0]["metadata"]["ocr_engine"] == "Apple Vision VNRecognizeTextRequest"
    assert docs[0]["metadata"]["ocr_mean_confidence"] == {"2": 0.99}
    assert pdf.read_bytes() == b"pdf bytes"


def test_loader_rejects_stale_ocr_hash(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"original bytes")
    _ocr_sidecar(pdf, ["recognized text"])
    pdf.write_bytes(b"changed PDF")
    with pytest.raises(ValueError, match="stale OCR"):
        load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader([""]))


@pytest.mark.parametrize("change", ["missing", "reordered", "empty", "confidence"])
def test_loader_rejects_incomplete_or_invalid_ocr_pages(tmp_path, change):
    pdf = tmp_path / "scan.pdf"
    pdf.touch()
    sidecar, record = _ocr_sidecar(pdf, ["first", "second"])
    if change == "missing":
        record["pages"].pop()
    elif change == "reordered":
        record["pages"].reverse()
    elif change == "empty":
        record["pages"][0]["text"] = " "
    else:
        record["pages"][0]["mean_confidence"] = float("nan")
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid OCR sidecar"):
        load_documents(tmp_path, pdf_reader_factory=lambda _: _pdf_reader(["", ""]))


def test_loader_fails_for_missing_pdf_dependency(monkeypatch, tmp_path):
    (tmp_path / "file.pdf").touch()
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(ImportError, match="every corpus PDF"):
        load_documents(tmp_path)


def test_loader_rejects_empty_or_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
    with pytest.raises(ValueError, match="No Markdown or PDF"):
        load_documents(tmp_path)
    (tmp_path / "empty.md").write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_documents(tmp_path)


def test_hierarchical_chunks_preserve_exact_text_stable_ids_and_metadata():
    text = "# Chính sách\n\n## Nghỉ phép\nQuy định có mười lăm ngày.\n\n## Hồ sơ\nNộp đơn đúng hạn.\n"
    metadata = {"source": "policy.md", "version": "2.0", "nested": {"department": "HR"}}
    parents, children = chunk_hierarchical(text, metadata, parent_size=12, child_size=6)
    again = chunk_hierarchical(text, metadata, parent_size=12, child_size=6)
    assert "".join(p.text for p in parents) == text
    assert "".join(c.text for c in children) == text
    assert [c.chunk_id for c in children] == [c.chunk_id for c in again[1]]
    assert len({c.chunk_id for c in children}) == len(children)
    assert all(len(p.text.split()) <= 12 for p in parents)
    assert all(len(c.text.split()) <= 6 for c in children)
    assert all(c.parent_id in {p.chunk_id for p in parents} for c in children)
    assert all(c.text == text[c.metadata["char_start"]:c.metadata["char_end"]] for c in children)
    assert all(c.metadata["version"] == "2.0" for c in children)
    assert any("Nghỉ phép" in c.metadata["section_path"] + c.metadata["headings"] for c in children)
    children[0].metadata["nested"]["department"] = "changed"
    assert metadata["nested"]["department"] == "HR"
    assert children[-1].metadata["nested"]["department"] == "HR"
    changed = chunk_hierarchical(text + "Thay đổi.", metadata, parent_size=12, child_size=6)
    assert children[0].chunk_id != changed[1][0].chunk_id


def test_hierarchical_chunks_prefer_paragraph_boundary():
    text = "one two three four five.\n\nsix seven eight nine ten eleven twelve."
    _, children = chunk_hierarchical(text, parent_size=20, child_size=8)
    assert children[0].text == "one two three four five.\n\n"
    assert chunk_hierarchical("   ") == ([], [])
    with pytest.raises(ValueError, match="positive"):
        chunk_hierarchical("text", child_size=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        chunk_hierarchical("text", parent_size=2, child_size=3)


def test_enrichment_preserves_order_raw_text_versions_and_identity():
    chunks = [
        {"text": "Nội dung cũ.\n", "metadata": {
            "source": "v1.md", "title": "Nghỉ phép", "version": "1.0", "chunk_id": "first",
            "status": "ĐÃ THAY THẾ bởi v2.0", "section_path": ["Nghỉ phép"], "headings": ["Hồ sơ"],
        }},
        {"text": "Nội dung mới.", "metadata": {"source": "v2.md", "version": "2.0", "chunk_id": "second"}},
    ]
    before = deepcopy(chunks)
    enriched = enrich_chunks(chunks)
    assert len(enriched) == 2
    assert [e.auto_metadata["chunk_id"] for e in enriched] == ["first", "second"]
    for row, original in zip(enriched, chunks):
        assert row.enriched_text.endswith(original["text"])
        assert row.auto_metadata["raw_text"] == original["text"]
        assert row.auto_metadata["enrichment_method"] == "source_metadata_v1"
    assert "ĐÃ THAY THẾ bởi v2.0" in enriched[0].enriched_text
    assert "Mục: Nghỉ phép > Hồ sơ" in enriched[0].enriched_text
    assert chunks == before
    assert enrich_chunks([]) == []


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.queries = []

    def get_scores(self, query):
        self.queries.append(query)
        return [sum(doc.count(term) for term in query) for doc in self.corpus]


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[2.0, 1.0, 0.0] for _ in texts]


FAKE_MODELS = SimpleNamespace(
    Distance=SimpleNamespace(COSINE="Cosine"),
    VectorParams=lambda **kwargs: SimpleNamespace(**kwargs),
    PointStruct=lambda **kwargs: SimpleNamespace(**kwargs),
)


class FakeQdrant:
    def __init__(self):
        self.collections = {}
        self.creates, self.upserts, self.searches = [], [], []
        self.search_order = None

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.creates.append(collection_name)
        self.collections[collection_name] = {"config": vectors_config, "points": {}}

    def get_collection(self, collection_name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.collections[collection_name]["config"])))

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def retrieve(self, collection_name, ids, **kwargs):
        points = self.collections[collection_name]["points"]
        return [points[i] for i in ids if i in points]

    def upsert(self, collection_name, points, wait):
        assert wait is True
        self.upserts.append((collection_name, points))
        self.collections[collection_name]["points"].update({point.id: point for point in points})

    def search(self, **kwargs):
        self.searches.append(kwargs)
        ids = list(self.collections[kwargs["collection_name"]]["points"])
        order = self.search_order if self.search_order is not None else list(range(len(ids)))
        return [SimpleNamespace(id=ids[i], score=1 - rank / 10) for rank, i in enumerate(order[:kwargs["limit"]])]


def _search(**kwargs):
    return HybridSearch(
        encoder=kwargs.pop("encoder", FakeEncoder()), client=kwargs.pop("client", FakeQdrant()),
        bm25_factory=FakeBM25, qdrant_models=FAKE_MODELS,
        model_name="test/embedding", embedding_dim=3, collection_name="test_rag", **kwargs,
    )


CHUNKS = [
    {"text": "nghỉ nghỉ phép", "metadata": {"source": "leave.md"}},
    {"text": "xin nghỉ", "metadata": {"source": "request.md"}},
    {"text": "bảo mật", "metadata": {"source": "security.md"}},
]


def test_hybrid_indexes_normalized_embeddings_and_fuses_real_rankings():
    search = _search(bm25_top_k=2, dense_top_k=2, hybrid_top_k=3)
    search.index(CHUNKS)
    search.client.search_order = [2, 1]
    rows = search.search("nghỉ")
    assert [row.text for row in rows] == ["xin nghỉ", "nghỉ nghỉ phép", "bảo mật"]
    assert rows[0].score == pytest.approx(2 / 62)
    assert rows[1].score == pytest.approx(1 / 61)
    assert rows[0].metadata["retrieval"]["bm25_rank"] == 2
    assert rows[0].metadata["retrieval"]["dense_rank"] == 2
    assert search.encoder.calls[0][0] == [row["text"] for row in CHUNKS]
    assert search.encoder.calls[-1][0] == ["nghỉ"]
    assert search.encoder.calls[0][1]["normalize_embeddings"] is True
    points = search.client.collections[search.collection_name]["points"].values()
    assert all(math.sqrt(sum(v * v for v in p.vector)) == pytest.approx(1) for p in points)
    assert all(p.payload["embedding_model"] == "test/embedding" for p in points)
    assert len(search.client.creates) == 1


def test_hybrid_collection_reuse_is_safe_and_does_not_reembed():
    client = FakeQdrant()
    first = _search(client=client)
    first.index(CHUNKS)
    second = _search(client=client)
    second.index(CHUNKS)
    assert first.collection_name == second.collection_name
    assert second.encoder.calls == []
    assert len(client.creates) == 1 and len(client.upserts) == 1
    changed = _search(client=client)
    changed.index([*CHUNKS, {"text": "new corpus", "metadata": {}}])
    assert changed.collection_name != first.collection_name
    assert len(client.collections) == 2


def test_hybrid_detects_foreign_payload_before_any_overwrite():
    client = FakeQdrant()
    first = _search(client=client)
    first.index(CHUNKS)
    existing = next(iter(client.collections[first.collection_name]["points"].values()))
    existing.payload["text"] = "user changed this"
    upserts_before = len(client.upserts)
    with pytest.raises(ValueError, match="nothing will be overwritten"):
        _search(client=client).index(CHUNKS)
    assert len(client.upserts) == upserts_before
    assert existing.payload["text"] == "user changed this"


def test_hybrid_resumes_incomplete_index_and_rejects_foreign_ids():
    client = FakeQdrant()
    first = _search(client=client)
    first.index(CHUNKS)
    points = client.collections[first.collection_name]["points"]
    missing_id = list(points)[-1]
    del points[missing_id]
    resumed = _search(client=client)
    resumed.index(CHUNKS)
    assert resumed.encoder.calls[0][0] == [CHUNKS[-1]["text"]]
    assert len(points) == len(CHUNKS)
    points["foreign"] = points.pop(missing_id)
    with pytest.raises(ValueError, match="unexpected IDs"):
        _search(client=client).index(CHUNKS)


def test_hybrid_rejects_wrong_vector_shape_and_invalid_state():
    search = _search(encoder=SimpleNamespace(encode=lambda *args, **kwargs: [[1.0]] * len(args[0])))
    with pytest.raises(RuntimeError, match="index"):
        search.search("question")
    with pytest.raises(ValueError, match="empty corpus"):
        search.index([])
    with pytest.raises(ValueError, match="3-dimensional"):
        search.index(CHUNKS)
    with pytest.raises(RuntimeError, match="index"):
        search.search("question")
    assert _tokenize("NGHỈ nghỉ phép") == ["nghỉ", "nghỉ", "phép"]


def test_hybrid_defaults_import_actual_model_and_bm25_lazily(monkeypatch):
    constructor_calls = []
    encoder = FakeEncoder()
    def load_model(name, **kwargs):
        constructor_calls.append((name, kwargs))
        return encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load_model))
    monkeypatch.setitem(sys.modules, "rank_bm25", SimpleNamespace(BM25Okapi=FakeBM25))
    search = HybridSearch(client=FakeQdrant(), qdrant_models=FAKE_MODELS, embedding_dim=3)
    assert constructor_calls == []
    search.index(CHUNKS)
    assert constructor_calls[0][0] == "BAAI/bge-m3"
    assert encoder.max_seq_length == search.max_length


def test_hybrid_with_installed_qdrant_19_and_bm25_offline():
    """Exercise pinned client models/storage without downloading neural weights."""
    qdrant = pytest.importorskip("qdrant_client")
    pytest.importorskip("rank_bm25")
    client = qdrant.QdrantClient(":memory:")
    try:
        search = HybridSearch(
            client=client, encoder=FakeEncoder(), model_name="offline-test", embedding_dim=3,
            collection_name="compatibility_test",
        )
        search.index(CHUNKS)
        results = search.search("bảo mật")
        assert results[0].metadata["source"] == "security.md"
        assert results[0].metadata["retrieval"]["bm25_score"] > 0
        assert client.count(search.collection_name, exact=True).count == 3
        reused = HybridSearch(
            client=client, encoder=FakeEncoder(), model_name="offline-test", embedding_dim=3,
            collection_name="compatibility_test",
        )
        reused.index(CHUNKS)
        assert reused.encoder.calls == []
    finally:
        client.close()


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_reranker_uses_pair_scores_not_lexical_or_input_scores():
    model = FakeCrossEncoder([-4.0, 8.0, 8.0])
    reranker = CrossEncoderReranker(model=model)
    docs = [dict(row, score=1000 if i == 0 else -1000) for i, row in enumerate(CHUNKS)]
    before = deepcopy(docs)
    result = reranker.rerank("nghỉ", docs, top_k=2)
    assert [r.text for r in result] == ["xin nghỉ", "bảo mật"]
    assert result[0].score == 8.0
    assert model.calls[0][0] == [("nghỉ", doc["text"]) for doc in docs]
    assert result[0].metadata["reranking"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert docs == before


@pytest.mark.parametrize("scores, match", [([1.0], "number"), ([float("nan")] * 3, "non-finite")])
def test_reranker_rejects_invalid_model_output(scores, match):
    with pytest.raises(ValueError, match=match):
        CrossEncoderReranker(model=FakeCrossEncoder(scores)).rerank("query", CHUNKS)


def test_reranker_loads_multilingual_crossencoder_lazily(monkeypatch):
    calls = []
    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        return FakeCrossEncoder([[0.2]])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=load_model))
    reranker = CrossEncoderReranker()
    assert calls == []
    assert reranker.rerank("query", []) == []
    result = reranker.rerank("query", [CHUNKS[0]])
    assert result[0].score == 0.2
    assert calls[0][0] == "BAAI/bge-reranker-v2-m3"
    assert calls[0][1]["max_length"] == reranker.max_length
