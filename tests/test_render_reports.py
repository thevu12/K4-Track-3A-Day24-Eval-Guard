"""Offline renderer regressions; synthetic evidence only exists under tmp_path.

These fixtures exercise validation/rendering, not experiment quality or live API
provenance. Never copy their outputs into the repository's artifacts directory.
"""
from pathlib import Path
import shutil

import pytest

from config import ROOT_DIR
from scripts import render_reports
from src.experiment_support import read_json, write_json
from src.report_contracts import validate_markdown
from tests.test_report_contracts import answers, phase_a_data, phase_b_data
from tests.test_support import guard_report

pytestmark = pytest.mark.support


@pytest.fixture
def evidence_root(tmp_path, phase_a_data, phase_b_data, guard_report):
    root = tmp_path / "synthetic-renderer-only"
    source = Path(ROOT_DIR)
    shutil.copytree(source / "doc/schemas", root / "doc/schemas")
    shutil.copytree(source / "fixtures", root / "fixtures")
    write_json(root / "artifacts/answers_50q.json", phase_a_data[1])
    write_json(root / "artifacts/ragas_50q.json", phase_a_data[0])
    write_json(root / "artifacts/judge_results.json", phase_b_data[0])
    write_json(root / "artifacts/guard_results.json", guard_report[0])
    return root


def test_renders_all_three_from_validated_evidence(evidence_root):
    paths = render_reports.generate_reports(evidence_root)
    assert [path.name for path in paths] == list(render_reports.OUTPUT_NAMES)
    failure, bias, blueprint = [validate_markdown(path) for path in paths]
    assert "student diagnosis" in failure and "student fix" in failure
    assert failure.count("### Câu ") == 10
    assert "Context 1: context" in failure and "Reference:" in failure
    assert "0.8000" in failure and "SHA-256" in failure
    assert "### Cặp 41" in bias and "pipeline answer" in bias
    assert "κ = (p_o − p_e)/(1 − p_e) = 1.0000" in bias
    assert "Input P95=4.0000" in blueprint and "chưa đo" in blueprint
    for text in (failure, bias, blueprint):
        assert "Giang Thế Vũ — 02478" in text
        assert "Nguyễn Hoàng Lê Nguyên — 02472" in text
        assert "{{" not in text


@pytest.mark.parametrize("mutation", [
    "missing_answers", "missing_a", "missing_b", "missing_c", "bad_json",
    "nonlive", "stale_hash", "wrong_score", "bad_latency",
])
def test_invalid_or_missing_evidence_never_replaces_reports(evidence_root, mutation):
    output = evidence_root / "doc/reports"
    output.mkdir()
    original = "Existing reviewed content must remain unchanged.\n"
    for name in render_reports.OUTPUT_NAMES:
        (output / name).write_text(original, encoding="utf-8")
    paths = evidence_root / "artifacts"
    if mutation.startswith("missing"):
        filename = {"missing_answers": "answers_50q.json", "missing_a": "ragas_50q.json",
                    "missing_b": "judge_results.json", "missing_c": "guard_results.json"}[mutation]
        (paths / filename).unlink()
    elif mutation == "bad_json":
        (paths / "guard_results.json").write_text("{bad JSON", encoding="utf-8")
    else:
        filename = "guard_results.json" if mutation == "bad_latency" else "ragas_50q.json"
        report = read_json(paths / filename)
        if mutation == "nonlive":
            report["metadata"]["run_kind"] = "mock"
        elif mutation == "stale_hash":
            report["metadata"]["source_sha256"]["answers_50q.json"] = "0" * 64
        elif mutation == "wrong_score":
            report["per_distribution"]["factual"]["faithfulness"] = .1
        else:
            report["latency"]["samples"][0]["presidio_ms"] = 0
        write_json(paths / filename, report)
    assert render_reports.main(["--root", str(evidence_root), "--overwrite"]) == 1
    assert all((output / name).read_text(encoding="utf-8") == original
               for name in render_reports.OUTPUT_NAMES)
    assert sorted(p.name for p in output.iterdir()) == sorted(render_reports.OUTPUT_NAMES)


def test_existing_reports_require_explicit_overwrite(evidence_root):
    paths = render_reports.generate_reports(evidence_root)
    reviewed = "Human review added; keep until explicitly overwritten.\n"
    paths[0].write_text(reviewed, encoding="utf-8")
    with pytest.raises(ValueError, match="--overwrite"):
        render_reports.generate_reports(evidence_root)
    assert paths[0].read_text(encoding="utf-8") == reviewed
    render_reports.generate_reports(evidence_root, overwrite=True)
    assert "student diagnosis" in paths[0].read_text(encoding="utf-8")


def test_reports_failed_real_guard_runs_without_faking_success(evidence_root):
    path = evidence_root / "artifacts/guard_results.json"
    report = read_json(path)
    for row in report["adversarial"]["results"]:
        row.update(actual="allowed", blocked_by=None, passed=False)
    report["adversarial"].update(passed=0, pass_rate=0)
    for row in report["clean"]["results"]:
        row.update(actual="blocked", blocked_by="nemo_input", passed=False)
    report["clean"].update(passed=0, pass_rate=0, false_positive_rate=1)
    write_json(path, report)
    paths = render_reports.generate_reports(evidence_root)
    blueprint = paths[2].read_text(encoding="utf-8")
    assert "| Attacks passed | 0/20 |" in blueprint
    assert "| Clean allowed | 0/8 |" in blueprint
    assert "| Clean false-positive rate | 1.0000 |" in blueprint
    assert "| Guard quality bắt buộc lab (không gồm latency) | chưa đạt |" in blueprint
    assert "| attack |" in blueprint and "| clean |" in blueprint


def test_output_dir_preserves_human_review_and_no_network(evidence_root, tmp_path):
    alternative = tmp_path / "preview"
    assert render_reports.main(["--root", str(evidence_root), "--output-dir", str(alternative)]) == 0
    assert all((alternative / name).is_file() for name in render_reports.OUTPUT_NAMES)
    assert not (evidence_root / "doc/reports").exists()


def test_markdown_escapes_artifact_data():
    value = render_reports.cell("a|b\n<script>alert(1)</script> [link](http://invalid.test) _text_")
    assert "a&#124;b<br>" in value
    assert "<script>" not in value and "&lt;script&gt;" in value
    assert "[link]" not in value and "&#91;link&#93;" in value


def test_missing_evidence_does_not_create_output_dir(tmp_path):
    with pytest.raises(ValueError, match="Thiếu bằng chứng"):
        render_reports.generate_reports(tmp_path)
    assert not (tmp_path / "doc/reports").exists()
