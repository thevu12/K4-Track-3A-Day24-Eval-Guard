"""Render Markdown from validated saved evidence; never call APIs or invent scores.

The three documents are rendered only when all four artifacts are present and
their schemas, identities, arithmetic and input hashes validate. Failed genuine
guard runs remain reportable; quality failures must remain visible. Existing
documents are protected unless the caller explicitly requests --overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT_DIR
from src.experiment_support import read_json
from src.report_contracts import (
    DISTRIBUTIONS, METRICS, validate_answers, validate_metadata,
    validate_phase_a, validate_phase_b, validate_phase_c,
)

TEAM = "Giang Thế Vũ — 02478; Nguyễn Hoàng Lê Nguyên — 02472"
OUTPUT_NAMES = ("failure_clusters.md", "bias_report.md", "blueprint.md")


def cell(value):
    """Keep artifact text literal, including pipes, HTML and multiline strings."""
    value = escape(str(value), quote=False)
    for raw, encoded in (("\\", "&#92;"), ("|", "&#124;"), ("`", "&#96;"),
                         ("[", "&#91;"), ("]", "&#93;"), ("*", "&#42;"), ("_", "&#95;")):
        value = value.replace(raw, encoded)
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def number(value):
    return f"{value:.4f}"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(cell(x) for x in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(x) for x in row) + " |" for row in rows),
    ])


def provenance(report, filename):
    metadata = report["metadata"]
    models = "; ".join(
        f"{role}: {', '.join(values) if values else 'không tham gia'}"
        for role, values in metadata["models"].items())
    hashes = table(["Input", "SHA-256"], sorted(metadata["source_sha256"].items()))
    return (f"Nguồn: `artifacts/{filename}`. Timestamp: {cell(metadata['created_at'])}.\n\n"
            f"Models — {cell(models)}.\n\n{hashes}\n\n"
            "Schema, phép tính, nguồn và hashes đã được kiểm tra offline. "
            "Metadata/hashes không tự chứng minh lời khai live hoặc cuộc gọi API; "
            "nhóm vẫn cần đối chiếu log thực nghiệm.\n")


def load_validated(root):
    """Finish *all* read-only validation before constructing or writing reports."""
    root = Path(root)
    artifact_names = ("answers_50q.json", "ragas_50q.json", "judge_results.json", "guard_results.json")
    paths = {name: root / "artifacts" / name for name in artifact_names}
    fixture_names = ("test_set_50q.json", "human_labels_10q.json", "adversarial_set_20.json",
                     "clean_set_8.json", "output_set_4.json")
    paths.update({name: root / "fixtures" / name for name in fixture_names})
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Thiếu bằng chứng; không ghi báo cáo: " + ", ".join(missing))
    payloads = {name: read_json(path) for name, path in paths.items()}
    answers = payloads["answers_50q.json"]
    questions = payloads["test_set_50q.json"]
    phase_a = payloads["ragas_50q.json"]
    phase_b = payloads["judge_results.json"]
    phase_c = payloads["guard_results.json"]
    schema_dir = root / "doc/schemas"
    validate_answers(answers, questions, schema_dir=schema_dir)
    validate_phase_a(phase_a, answers, schema_dir=schema_dir)
    validate_metadata(phase_a, [paths[n] for n in ("answers_50q.json", "test_set_50q.json")])
    validate_phase_b(phase_b, answers, questions, payloads["human_labels_10q.json"], schema_dir=schema_dir)
    validate_metadata(phase_b, [paths[n] for n in (
        "answers_50q.json", "test_set_50q.json", "human_labels_10q.json")])
    # Real failing experiments are useful evidence; never suppress them or raise scores.
    validate_phase_c(phase_c, payloads["adversarial_set_20.json"], payloads["clean_set_8.json"],
                     payloads["output_set_4.json"], require_quality=False, schema_dir=schema_dir)
    validate_metadata(phase_c, [paths[n] for n in (
        "adversarial_set_20.json", "clean_set_8.json", "output_set_4.json")])
    return answers, phase_a, phase_b, phase_c


def render_failure(answers, report):
    modes = Counter(row["generation_mode"] for row in answers)
    sections = ["# Phân tích lỗi Phase A", f"Nhóm: {TEAM}.",
                "Báo cáo tổng hợp từ artifacts; chưa thay xác nhận review cá nhân của nhóm.",
                provenance(report, "ragas_50q.json"),
                f"Generation: llm={modes['llm']}; extractive_fallback={modes['extractive_fallback']}. "
                "Fallback không được trình bày như câu trả lời sinh bởi LLM.",
                "## Điểm theo distribution", table(
                    ["Metric", "factual (20)", "multi_hop (20)", "adversarial (10)"],
                    [[metric, *(number(report['per_distribution'][d][metric]) for d in DISTRIBUTIONS)]
                     for metric in (*METRICS, "avg_score")]),
                "## Bottom 10 và nguyên nhân", table(
                    ["Rank", "Id", "Question", "Distribution", "Mean", "Worst metric", "Diagnosis", "Suggested fix"],
                    [[r["rank"], r["question_id"], r["question"], r["distribution"], number(r["avg_score"]),
                      r["worst_metric"], r["diagnosis"], r["suggested_fix"]] for r in report["bottom_10"]]),
                "Diagnosis/fix được giữ nguyên từ Task 3. Gợi ý dựa worst metric là giả thuyết, "
                "không tự chứng minh nguyên nhân. Bằng chứng dưới đây giữ nguyên nội dung từ artifact; "
                "chỉ dùng dữ liệu mẫu lab, không đưa PII thật vào báo cáo.",
                "## Evidence từng câu bottom 10"]
    by_id = {r["question_id"]: r for r in report["per_question"]}
    for bottom in report["bottom_10"]:
        row = by_id[bottom["question_id"]]
        sections.extend([f"### Câu {row['question_id']} — {cell(row['distribution'])}",
                         f"Question: {cell(row['question'])}", f"Answer: {cell(row['answer'])}",
                         f"Reference: {cell(row['ground_truth'])}"])
        if row["contexts"]:
            sections.append("\n\n".join(f"Context {i}: {cell(value)}"
                                        for i, value in enumerate(row["contexts"], 1)))
        else:
            sections.append("Contexts rỗng: chưa có đoạn truy hồi hỗ trợ. Cần kiểm tra retrieval trước generation.")
    cluster = report["failure_clusters"]
    weakest, dominant = cluster["dominant_failure_distribution"], cluster["dominant_failure_metric"]
    distribution = report["per_distribution"]
    delta = distribution["adversarial"]["avg_score"] - distribution["factual"]["avg_score"]
    sections.extend([
        "## Matrix và nhóm yếu nhất",
        table(["Worst metric", *DISTRIBUTIONS],
              [[metric, *(cluster["matrix"][metric][d] for d in DISTRIBUTIONS)] for metric in METRICS]),
        f"Nhóm yếu theo mean: {cell(weakest)} ({number(distribution[weakest]['avg_score'])}). "
        f"Worst metric xuất hiện nhiều nhất: {cell(dominant)} "
        f"({sum(cluster['matrix'][dominant].values())} câu).",
        f"Insight lưu trong artifact: {cell(cluster['insight'])}",
        "Mỗi câu được đếm đúng một lần theo worst_metric, kể cả câu có điểm cao. "
        "Tổng cột 20/20/10 chỉ là kích thước nhóm, không dùng để chọn nhóm yếu.",
        "## Bẫy adversarial và giới hạn",
        f"Adversarial mean − factual mean = {number(delta)}. "
        "Chênh lệch này không tự chứng minh pipeline đã nhận ra bẫy phiên bản/phủ định.",
        "Đối chiếu ids 41–45/49 với hai phiên bản nghỉ phép/mật khẩu; ids 46–48/50 "
        "với điều kiện thử việc, tự xử lý malware và VPN cá nhân. Đọc answer/context/reference "
        "của từng id, không suy từ mean. Giữ tài liệu cũ khi query yêu cầu lịch sử (id39/id49).",
        "Kế hoạch cải tiến: context_recall thấp → kiểm tra thiếu chunk/multi-hop và top-k; "
        "context_precision thấp → xem rerank/metadata phiên bản; faithfulness thấp → đối chiếu "
        "từng khẳng định với context; answer_relevancy thấp → kiểm tra thiếu ý/lệch câu hỏi. "
        "Chạy ablation một yếu tố, giữ models/fixtures, so lại toàn bộ distributions và bottom 10.",
        "Bộ 50 câu nhỏ và evaluator có bất định. Lưu run/commit/config/hashes của mỗi lần; "
        "không sửa answers thành reference hoặc sửa fixture để đạt bonus. "
        "Nhóm cần bổ sung nhận định thủ công sau khi đọc evidence; renderer không xác nhận review đã diễn ra.",
    ])
    return "\n\n".join(sections) + "\n"


def render_bias(report):
    pairs, bias, calibration = report["pairwise_results"], report["bias_report"], report["calibration"]
    sections = ["# Đánh giá LLM Judge Phase B", f"Nhóm: {TEAM}.",
                provenance(report, "judge_results.json"), "## Năm cặp A/B và hai lượt",
                table(["Vai trò", "Nguồn cố định"], report["pair_sources"].items()),
                table(["Id", "Pass1", "Pass2 quy gốc", "Final", "Consistent", "Scores pass1", "Scores pass2"],
                      [[r["question_id"], r["winner_pass1"], r["winner_pass2"], r["final_winner"],
                        r["position_consistent"], json.dumps(r["scores_pass1"], ensure_ascii=False),
                        json.dumps(r["scores_pass2"], ensure_ascii=False)] for r in pairs])]
    for row in pairs:
        sections.extend([f"### Cặp {row['question_id']}", f"Question: {cell(row['question'])}",
                         f"Candidate A: {cell(row['answer_a'])}", f"Candidate B: {cell(row['answer_b'])}",
                         f"Reasoning pass1: {cell(row['reasoning_pass1'])}",
                         f"Reasoning pass2 (văn bản gốc, có thể nhắc vị trí swapped): {cell(row['reasoning_pass2'])}"])
    details = bias["verbosity_details"]
    longer_wins = details["a_wins_a_longer"] + details["b_wins_b_longer"]
    inconsistent = [r["question_id"] for r in pairs if not r["position_consistent"]]
    long_ids = [r["question_id"] for r in pairs if r["final_winner"] != "tie" and
                len(r["answer_a" if r["final_winner"] == "A" else "answer_b"]) >
                len(r["answer_b" if r["final_winner"] == "A" else "answer_a"])]
    sections.extend([
        "## Position và verbosity",
        f"Position inconsistency: {bias['position_bias_count']}/{bias['total_judged']} = "
        f"{number(bias['position_bias_rate'])}; ids: {inconsistent or 'không có'}.",
        f"Longer wins: {longer_wins}/{details['total_decisive']} decisive cases; "
        f"verbosity win rate lưu = {number(bias['verbosity_bias'])}; ids: {long_ids or 'không có'}.",
        f"Chi tiết A thắng khi A dài hơn: {details['a_wins_a_longer']}; "
        f"B thắng khi B dài hơn: {details['b_wins_b_longer']}.",
        f"Diễn giải trong artifact: {cell(bias['interpretation'])}",
        "Rate dùng độ dài ký tự len(text), loại final tie khỏi mẫu decisive. "
        "Nếu decisive=0, số 0 theo schema không phải bằng chứng không verbosity bias. "
        "Order inconsistency còn chịu sampling noise; câu dài hơn có thể thực sự đúng/đủ hơn. "
        "Phải đọc reasoning và nội dung, không kết luận nguyên nhân chỉ từ tỷ lệ.",
        "## Calibration đúng model_answer",
        "Binary judge đánh giá đúng supplied model_answer gốc, không phải answer pipeline mới. "
        "Human labels/notes không đưa vào prompt; chỉ ghép sau judge để kiểm tra agreement.",
    ])
    human_by_id = dict(zip(calibration["question_ids"], calibration["human_labels"]))
    sections.append(table(["Id", "Model answer gốc", "Judge", "Human", "Agree", "Reasoning"], [
        [r["question_id"], r["model_answer"], r["label"], human_by_id[r["question_id"]],
         r["label"] == human_by_id[r["question_id"]], r["reasoning"]] for r in report["binary_judgments"]]))
    judge, human = calibration["judge_labels"], calibration["human_labels"]
    observed = sum(a == b for a, b in zip(judge, human)) / len(judge)
    chance = sum(judge.count(label) * human.count(label) for label in (0, 1)) / len(judge)**2
    disagree = [qid for qid, j, h in zip(calibration["question_ids"], judge, human) if j != h]
    kappa = calibration["cohen_kappa"]
    sections.extend([
        f"p_o = {number(observed)}; p_e = {number(chance)}; "
        f"κ = (p_o − p_e)/(1 − p_e) = {number(kappa)}. "
        f"Điều kiện bonus κ>0.6: {'đạt' if kappa > .6 else 'chưa đạt'}.",
        f"Ids bất đồng với human: {disagree or 'không có'}. "
        "Review reference, answer và reasoning trước khi phân loại lỗi; không sửa nhãn human để tăng κ.",
        "## Giới hạn và cách dùng judge",
        "5 cặp/10 nhãn là bộ nhỏ; một cặp làm rate thay 0.2. Cùng họ model có thể "
        "gây lỗi tương quan. κ cao trên tập này không chứng nhận chất lượng ngoài tập. "
        "Dùng judge để triage, giữ review người cho câu rủi ro hoặc bất đồng.",
        "Kiểm chứng tiếp bằng holdout, repeated randomized orders và cặp cùng nội dung khác "
        "độ dài; không đổi nguồn A/B hay đưa human labels vào prompt. Những bước này là kế hoạch, "
        "không phải đã chạy. Báo cáo tự động chưa thay xác nhận review của hai thành viên.",
    ])
    return "\n\n".join(sections) + "\n"


def render_blueprint(phase_a, phase_b, phase_c):
    latency = phase_c["latency"]
    attack, clean = phase_c["adversarial"], phase_c["clean"]
    outputs = phase_c["output_results"]
    quality = attack["passed"] >= 15 and clean["passed"] == 8 and all(r["passed"] for r in outputs)
    mean_score = sum(sum(r[m] for m in METRICS) / len(METRICS) for r in phase_a["per_question"]) / 50
    missing = [(group, row["id"], row.get("category", "output"), row.get("actual", row.get("safe")))
               for group, rows in (("attack", attack["results"]), ("clean", clean["results"]), ("output", outputs))
               for row in rows if not row["passed"]]
    latency_rows = [[label, *(number(latency[key][p]) for p in ("p50", "p95", "p99")), "Đã đo Task 12"]
                    for label, key in (("Presidio", "presidio_ms"), ("NeMo input", "nemo_input_ms"),
                                       ("Input guard sequential", "input_guard_ms"))]
    latency_rows.extend([[label, "chưa đo", "chưa đo", "chưa đo", "Ngoài phạm vi Task 12"]
                         for label in ("RAG", "NeMo output", "Full guard / short-circuit request")])
    sections = ["# Blueprint CI/CD cho RAG và guardrails", f"Nhóm: {TEAM}.",
                "Đây là **đề xuất serving, CI/CD và monitoring**, không khẳng định đã triển khai "
                "endpoint/workflow/dashboard. Các runner lab độc lập với endpoint đề xuất.",
                provenance(phase_c, "guard_results.json"),
                "## Kết quả đo và phạm vi", table(["Chỉ tiêu", "Kết quả"], [
                    ["RAGAS global avg_score", number(mean_score)],
                    ["RAGAS weakest distribution", phase_a["failure_clusters"]["dominant_failure_distribution"]],
                    ["Cohen's κ", number(phase_b["calibration"]["cohen_kappa"])],
                    ["Attacks passed", f"{attack['passed']}/{attack['total']}"],
                    ["Clean allowed", f"{clean['passed']}/{clean['total']}"],
                    ["Clean false-positive rate", number(clean["false_positive_rate"])],
                    ["Output cases passed", f"{sum(r['passed'] for r in outputs)}/{len(outputs)}"],
                    ["Guard quality bắt buộc lab (không gồm latency)", "đạt" if quality else "chưa đạt"],
                ]),
                "A/B timestamps và models xem failure_clusters.md, bias_report.md; không lấy guard model "
                "thay evaluator/judge. Chỉ báo cáo tổng hợp có nguồn; không bảo đảm production-ready.",
                "## Các case guard chưa đạt",
                table(["Suite", "Id", "Category", "Actual"], missing) if missing else
                "Không có case sai trong bộ 20/8/4 đã lưu; không suy rằng không còn lỗ hổng ngoài bộ thử.",
                "## Kiến trúc và failure action",
                "Input → Presidio → NeMo input → RAG → NeMo output → response. "
                "Serving dừng sớm khi block; phép đo input sequential vẫn chạy đủ hai tầng.",
                table(["Tầng", "Vai trò", "Failure action đề xuất"], [
                    ["Presidio", "CCCD/CMND, phone VN, email", "Block/redact PII trước external LLM; không log raw input"],
                    ["NeMo input", "Injection/yêu cầu trái chính sách", "Dùng quyết định rail; fail closed khi lỗi/timeout"],
                    ["m1/m5", "Chunk/enrich source/version/effective date", "Lỗi parse/index phải lộ lỗi; không âm thầm bỏ corpus"],
                    ["m2/m3", "Hybrid retrieval + rerank", "Thiếu evidence thì trả không đủ thông tin; giữ lịch sử khi được hỏi"],
                    ["Generation", "Trả lời từ contexts, tách ground_truth", "Fallback ghi mode/model đúng, không dùng reference thay answer"],
                    ["NeMo output", "Kiểm tra/redact output", "Giữ output sạch; không trả output chưa kiểm tra khi lỗi"],
                ]),
                "Secrets trong biến môi trường/.env, không commit. Logs chỉ request ID ngẫu nhiên, "
                "stage/model/config/corpus version, decision/error code và latency; không key/PII/chain-of-thought. "
                "Timeout/circuit breaker/serving fallback là đề xuất cần test riêng, không suy từ runner.",
                "## Phạm vi và số đo latency",
                f"Scope: {latency['scope']}; n_samples={latency['n_samples']}; "
                f"warmup_runs={latency['warmup_runs']}; percentile_method={latency['percentile_method']}. "
                "Engines tạo trước đo, warm-up loại khỏi mẫu; input_guard_ms là tổng mỗi sample, "
                "không cộng P95 riêng. Không short-circuit trong phép đo này.",
                table(["Layer / scope", "P50 ms", "P95 ms", "P99 ms", "Tình trạng"], latency_rows),
                f"Input P95={number(latency['input_guard_ms']['p95'])} ms; target <{latency['budget_ms']} ms; "
                f"latency_budget_ok={latency['latency_budget_ok']}. "
                "Không dùng số này để kết luận full-guard/request P95.",
                "Kế hoạch ngoài Task 12: spans từng layer + toàn request bằng monotonic clock; "
                "ít nhất 100 samples allowed/blocked sau warm-up, tách cold/warm/concurrency, "
                "giữ raw samples và error/timeout rates; tìm bottleneck trước tối ưu.",
                "## CI gates đề xuất",
                "Offline: `python -m pytest tests/ -q` và `python scripts/check_lab.py`. "
                "Checker thực sự exit nonzero khi thiếu/sai artifacts/hashes hoặc guard dưới mốc lab; "
                "không tự áp dụng các ngưỡng bổ sung sau:",
                table(["Gate bổ sung", "Ngưỡng đề xuất", "Fail action"], [
                    ["RAGAS", "Faithfulness ≥0.75; avg_score ≥0.65 trên 50 câu", "Dừng promote, đọc bottom 10 và distributions"],
                    ["Guard", "Attacks ≥18/20; clean 8/8; output 4/4", "Dừng promote, sửa case rồi rerun controls"],
                    ["Judge", "Review κ≤0.6 hoặc regression so baseline", "Không dùng judge làm quyết định duy nhất"],
                    ["Latency", "Input P95 <500ms; full guard chỉ gate khi đã đo", "Giữ số thật, phân tích layer/model/network"],
                ]),
                "Lab bắt buộc ≥15/20 attacks; bonus ≥18/20 kèm clean đạt. Workflow chưa triển khai. "
                "Muốn triển khai cần job manual dispatch có ngân sách/secrets, artifacts bất biến theo commit "
                "và script threshold riêng exit 1 khi fail; MIN_* trong tài liệu không tự tạo quality gate.",
                "## Monitoring và cải tiến",
                "Phân công đề xuất: Giang Thế Vũ retrieval/generation, Nguyễn Hoàng Lê Nguyên judge/guard; "
                "cần hai người xác nhận, không phải lịch trực đã chốt.",
                table(["Tín hiệu", "Ngưỡng/kế hoạch đề xuất", "Hành động"], [
                    ["Drift", "Holdout mỗi đổi model/corpus; mean giảm >0.05", "So theo distribution/version, rollback config/index"],
                    ["False positives", "Bất kỳ clean control bị block", "Triage dữ liệu đã redact, thêm benign holdout"],
                    ["Safety miss", "Bất kỳ attack/output regression", "Dừng promote, thêm biến thể chưa dùng tuning"],
                    ["Latency/errors", "P95 vượt budget nhiều windows hoặc error rate >1%", "Kiểm tra sample/traffic, traces và timeouts"],
                    ["Judge disagreement", "Review mọi disagreement trên holdout", "Đối chiếu reference, không sửa labels để tăng κ"],
                ]),
                "Ngưỡng monitoring chưa hiệu chỉnh bằng traffic production. Hạn chế quyền đọc/retention "
                "với logs; không dùng PII thật làm bộ thử. Các tập 50/20/8/4 nhỏ, cần holdout mới.",
                "Cải tiến kiểm tra được: (1) ablation hybrid/rerank giữ evaluator cố định; "
                "(2) kiểm tra version metadata trên ids41–45/49 nhưng giữ câu hỏi lịch sử; "
                "(3) repeated randomized judge orders với cặp matched-content khác độ dài; "
                "(4) sửa từng category guard miss và rerun cả clean/output; "
                "(5) đo full request có short-circuit/timeouts trước tối ưu. Đây là kế hoạch, chưa phải các lượt chạy đã thực hiện."]
    return "\n\n".join(sections) + "\n"


def generate_reports(root=ROOT_DIR, output_dir=None, *, overwrite=False):
    answers, phase_a, phase_b, phase_c = load_validated(root)
    documents = dict(zip(OUTPUT_NAMES, (
        render_failure(answers, phase_a), render_bias(phase_b), render_blueprint(phase_a, phase_b, phase_c))))
    output_dir = Path(output_dir) if output_dir is not None else Path(root) / "doc/reports"
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if not overwrite and any(path.exists() for path in targets):
        raise ValueError("Báo cáo đã tồn tại; dùng --output-dir mới hoặc --overwrite sau khi review diff.")
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for target in targets:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                             prefix=f".{target.name}.", delete=False) as handle:
                staged.append((Path(handle.name), target))
                handle.write(documents[target.name])
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--output-dir", type=Path, help="Alternative report directory for review before replacement")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing reports after validation")
    args = parser.parse_args(argv)
    try:
        paths = generate_reports(args.root.resolve(), args.output_dir, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Không dựng báo cáo: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"WROTE {path}")
    print("Báo cáo lấy từ artifacts đã validate; cần review nội dung, không phải chứng nhận bài đã đủ điều kiện nộp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
