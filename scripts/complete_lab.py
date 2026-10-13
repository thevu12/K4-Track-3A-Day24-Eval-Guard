"""Resume the real lab workflow without paying again for valid saved evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_support import read_json
from src.report_contracts import (
    validate_answers, validate_metadata, validate_phase_a, validate_phase_b, validate_phase_c,
)

PHASE_ARTIFACTS = {"a": "ragas_50q.json", "b": "judge_results.json", "c": "guard_results.json"}
VALIDATION_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError)


class WorkflowError(RuntimeError):
    """An expected workflow failure; do not continue to dependent steps."""


def api_key_present() -> bool:
    import config
    return isinstance(config.LLM_API_KEY, str) and bool(config.LLM_API_KEY.strip())


def _safe_provider_error(error: Exception) -> str:
    """Return diagnostics that cannot echo a provider body, request, or secret."""
    parts = [type(error).__name__]
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        parts.append(f"status={status}")
    code = getattr(error, "code", None)
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
        parts.append(f"code={code}")
    return ", ".join(parts)


def validate_provider_access() -> None:
    """Make exactly one read-only request before any costly workflow calls."""
    import config
    from src.provider_client import openai_client

    try:
        client = openai_client(max_retries=0, timeout=15.0)
        if config.LLM_PROVIDER == "gemini":
            available = {model.id.removeprefix("models/") for model in client.models.list()}
            if config.GENERATION_MODEL.removeprefix("models/") not in available:
                raise WorkflowError("Configured Gemini generation model is not available to this key.")
        else:
            client.models.retrieve(config.GENERATION_MODEL)
    except Exception as error:
        detail = _safe_provider_error(error)
        raise WorkflowError(
            f"Provider/model access check failed ({detail}). Replace or fix the key in local .env; "
            "no generation or experiment calls were made."
        ) from error


def run_script(root: Path, script: str, *args: str) -> None:
    """Use the invoking interpreter and repository cwd for every subprocess."""
    command = [sys.executable, str(root / "scripts" / script), *args]
    print("RUN " + " ".join([script, *args]), flush=True)
    try:
        result = subprocess.run(command, cwd=root, check=False)
    except OSError as error:
        raise WorkflowError(f"Cannot start {script} ({type(error).__name__}).") from error
    if result.returncode:
        raise WorkflowError(f"{script} exited with status {result.returncode}; later steps were not run.")


def validate_saved_answers(root: Path) -> list[dict]:
    answers = read_json(root / "artifacts/answers_50q.json")
    validate_answers(answers, read_json(root / "fixtures/test_set_50q.json"),
                     schema_dir=root / "doc/schemas")
    return answers


def validate_saved_phase(root: Path, phase: str) -> None:
    """Validate identity, arithmetic, schema and input hashes, including real failures.

    A truthful failed guard run is reusable evidence, not a reason to silently
    reroll an LLM. The unchanged final checker still enforces submission quality.
    """
    report = read_json(root / "artifacts" / PHASE_ARTIFACTS[phase])
    fixtures = root / "fixtures"
    answer_path = root / "artifacts/answers_50q.json"
    question_path = fixtures / "test_set_50q.json"
    schema_dir = root / "doc/schemas"
    if phase == "a":
        validate_phase_a(report, read_json(answer_path), schema_dir=schema_dir)
        sources = [answer_path, question_path]
    elif phase == "b":
        human_path = fixtures / "human_labels_10q.json"
        validate_phase_b(report, read_json(answer_path), read_json(question_path),
                         read_json(human_path), schema_dir=schema_dir)
        sources = [answer_path, question_path, human_path]
    else:
        sources = [fixtures / name for name in
                   ("adversarial_set_20.json", "clean_set_8.json", "output_set_4.json")]
        validate_phase_c(report, *(read_json(path) for path in sources),
                         require_quality=False, schema_dir=schema_dir)
    validate_metadata(report, sources)


def phase_is_reusable(root: Path, phase: str) -> bool:
    try:
        validate_saved_phase(root, phase)
    except VALIDATION_ERRORS as error:
        print(f"NEEDS RUN Phase {phase.upper()}: saved evidence missing/stale/invalid ({type(error).__name__}).")
        return False
    print(f"REUSE Phase {phase.upper()}: schema, provenance and input hashes validate.")
    return True


def complete(root: Path, *, regenerate: bool, rerun: bool, latency_samples: int) -> None:
    if type(latency_samples) is not int or latency_samples < 20:
        raise WorkflowError("Submission latency requires an integer sample count of at least 20.")
    if not api_key_present():
        raise WorkflowError("The configured provider API key is missing. Set it locally in .env before --live; no generation/API calls were made.")
    # Fail locally before generation can silently fall back due to a missing key.
    run_script(root, "preflight.py", "--live")
    validate_provider_access()
    answer_path = root / "artifacts/answers_50q.json"
    if regenerate or not answer_path.is_file():
        flags = ["--live"] + (["--overwrite"] if regenerate else [])
        run_script(root, "setup_answers.py", *flags)
    try:
        answers = validate_saved_answers(root)
    except VALIDATION_ERRORS as error:
        raise WorkflowError("Saved answers do not validate. Inspect the failure and use --regenerate only to explicitly replace them.") from error
    fallback_count = sum(row["generation_mode"] == "extractive_fallback" for row in answers)
    if fallback_count:
        print(f"WARNING {fallback_count}/{len(answers)} answers are extractive_fallback, not LLM generation. "
              "They are valid evidence; use --regenerate to request fresh generation.")

    pending = list(PHASE_ARTIFACTS) if rerun else [
        phase for phase in PHASE_ARTIFACTS if not phase_is_reusable(root, phase)
    ]
    # The existing runner already stops on failure. Use its all path only when
    # every phase needs work; otherwise leave expensive validated phases alone.
    for selected in (["all"] if pending == list(PHASE_ARTIFACTS) else pending):
        run_script(root, "run_experiments.py", "--phase", selected, "--live",
                   "--latency-samples", str(latency_samples))

    # A subprocess exit code alone is not evidence. Validate all outputs again
    # before allowing the renderer to overwrite reports.
    validate_saved_answers(root)
    for phase in PHASE_ARTIFACTS:
        validate_saved_phase(root, phase)
    run_script(root, "render_reports.py", "--overwrite")
    run_script(root, "check_lab.py", "--with-tests")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Acknowledge network/model/API use and potential charges")
    parser.add_argument("--regenerate", action="store_true", help="Explicitly replace saved answers (may invalidate A/B)")
    parser.add_argument("--rerun", action="store_true", help="Explicitly rerun all phases even when saved evidence is valid")
    parser.add_argument("--latency-samples", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Completion may incur API costs. Pass --live to acknowledge; preflight.py is read-only.")
    if args.latency_samples < 20:
        parser.error("Submission latency requires at least 20 timed samples.")
    try:
        complete(ROOT, regenerate=args.regenerate, rerun=args.rerun,
                 latency_samples=args.latency_samples)
    except (WorkflowError, *VALIDATION_ERRORS) as error:
        print(f"Completion stopped: {error}", file=sys.stderr)
        return 1
    print("All artifacts, rendered reports and offline submission checks passed. Review the reports and individual contributions before submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
