from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from numbers import Real

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import model_ids, write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DISTRIBUTIONS = ("factual", "multi_hop", "adversarial")

DIAGNOSTIC_TREE = {
    "faithfulness": (
        "Possible unsupported claims; verify the answer against its retrieved contexts.",
        "Inspect unsupported claims, then tighten grounding instructions if confirmed.",
    ),
    "answer_relevancy": (
        "Possible incomplete or off-question answer; inspect the question and response.",
        "Check refusal versus irrelevance, then revise the response prompt if needed.",
    ),
    "context_precision": (
        "Possibly irrelevant or poorly ranked contexts; inspect each retrieved chunk.",
        "Check distractor chunks and version metadata, then adjust reranking or filters.",
    ),
    "context_recall": (
        "Possibly missing reference evidence; compare the contexts with the reference.",
        "Verify missing evidence in the corpus, then improve chunking or hybrid retrieval.",
    ),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        # Equal scores use the same stable priority as report_contracts.METRICS.
        return min(METRICS, key=lambda name: getattr(self, name))


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: preserve measured metrics and actual evaluator model provenance.

    Missing rows, invalid scores and missing model IDs abort the experiment;
    they are never replaced with guessed model IDs or synthetic scores.
    """
    from src.m4_eval import evaluate_ragas
    if not isinstance(answers, list) or not answers:
        raise ValueError("RAGAS requires a non-empty list of saved answers.")
    raw = evaluate_ragas([a["question"] for a in answers], [a["answer"] for a in answers],
                         [a["contexts"] for a in answers], [a["ground_truth"] for a in answers])
    if not isinstance(raw, dict):
        raise ValueError("m4 must return an evaluation object.")
    per_question = raw.get("per_question")
    if not isinstance(per_question, list) or len(per_question) != len(answers):
        raise ValueError("m4 must return exactly one metric row per saved answer.")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("m4 must report evaluation metadata.")
    models = metadata.get("evaluator_models")
    model_ids(models, "metadata.evaluator_models")
    if not models:
        raise ValueError("m4 must report non-empty metadata.evaluator_models.")
    rows = []
    for answer, metric in zip(answers, per_question):
        values = []
        for name in METRICS:
            try:
                value = metric[name] if isinstance(metric, dict) else getattr(metric, name)
            except (KeyError, AttributeError) as exc:
                raise ValueError(f"Question {answer['id']}: missing RAGAS metric {name}.") from exc
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(
                    f"Question {answer['id']}: {name} must be finite numeric data in [0, 1]."
                )
            values.append(float(value))
        rows.append(RagasResult(answer["id"], answer["distribution"], answer["question"],
                                answer["answer"], answer["contexts"], answer["ground_truth"], *values))
    return RagasEvaluation(rows, list(models))


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    # Stable tie-breaking makes the report reproducible.
    rows = []
    for rank, result in enumerate(sorted(results, key=lambda r: (r.avg_score, r.question_id))[:10], 1):
        diagnosis, fix = DIAGNOSTIC_TREE[result.worst_metric]
        rows.append({"rank": rank, "question_id": result.question_id,
                     "distribution": result.distribution, "question": result.question,
                     "avg_score": round(result.avg_score, 4), "worst_metric": result.worst_metric,
                     "diagnosis": diagnosis, "suggested_fix": fix})
    return rows


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Matrix thống kê worst_metric của mọi câu, không coi mọi câu là một failure.
    Nhóm yếu nhất = distribution có avg_score trung bình thấp nhất (bỏ nhóm rỗng).
    Tổng matrix theo cột chỉ là kích thước nhóm 20/20/10, không đo chất lượng.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {metric: {dist: 0 for dist in DISTRIBUTIONS} for metric in METRICS}
    for result in results:
        if result.distribution not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {result.distribution!r}")
        matrix[result.worst_metric][result.distribution] += 1
    means = {dist: sum(r.avg_score for r in results if r.distribution == dist) /
             sum(r.distribution == dist for r in results)
             for dist in DISTRIBUTIONS if any(r.distribution == dist for r in results)}
    if not means:
        return {"matrix": matrix, "distribution_avg_scores": {},
                "dominant_failure_distribution": None, "dominant_failure_metric": None,
                "insight": "No results available."}
    weak_dist = min(means, key=means.get)
    weak_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    return {"matrix": matrix, "distribution_avg_scores": means,
            "dominant_failure_distribution": weak_dist, "dominant_failure_metric": weak_metric,
            "insight": (
                f"{weak_dist} has the lowest mean score; {weak_metric} occurs most often "
                "as the worst metric (including ties resolved by metric order). "
                "This is a diagnostic signal, not proof of a failure cause. "
                f"{DIAGNOSTIC_TREE[weak_metric][1]}"
            )}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
