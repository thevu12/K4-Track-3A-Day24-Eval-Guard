from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GUARDRAILS_CONFIG_DIR, JUDGE_MODEL, LLM_API_KEY, LLM_BASE_URL, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Build the explicit PII scope required by this Vietnamese policy lab.

    Detect CCCD/CMND, Vietnamese phones and email. The phone library additionally
    covers formatted/+84 numbers. Do not register generic English NER, dates,
    URLs or unrelated national IDs: a policy date or Vietnamese phrase is not
    itself private identifying data. Those recognizers caused genuine false
    positives when used on Vietnamese text with an English spaCy model.

    This scoped detector does not claim Vietnamese personal-name recognition;
    requests to disclose somebody's records belong to the NeMo input rail.
    The NLP engine is retained for library compatibility/context enhancement,
    but its NER recognizer is deliberately absent from the nonempty registry.
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        supported_language=PRESIDIO_LANGUAGE,
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        supported_language=PRESIDIO_LANGUAGE,
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry(supported_languages=[PRESIDIO_LANGUAGE])
    registry.add_recognizer(EmailRecognizer(supported_language=PRESIDIO_LANGUAGE))
    registry.add_recognizer(PhoneRecognizer(
        supported_language=PRESIDIO_LANGUAGE, supported_regions=["VN"],
    ))
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer = AnalyzerEngine(registry=registry, supported_languages=[PRESIDIO_LANGUAGE])
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}
    return {"has_pii": True,
            "entities": [{"type": r.entity_type, "text": text[r.start:r.end],
                          "score": round(float(r.score), 3), "start": r.start, "end": r.end} for r in results],
            "anonymized": anonymizer.anonymize(text=text, analyzer_results=results).text}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — câu từ chối Colang 1.0; built-in self-check flows được bật trong YAML
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    config.models[0].model = JUDGE_MODEL
    from langchain_openai import ChatOpenAI
    from src.provider_client import langchain_provider_kwargs
    # Explicit chat client: NeMo otherwise classifies non-GPT names as legacy completions.
    class PacedChat(ChatOpenAI):
        async def _agenerate(self, *args, **kwargs):
            if LLM_BASE_URL:
                await asyncio.sleep(4.25)
            result = await super()._agenerate(*args, **kwargs)
            # A truncated/empty self-check is a provider failure, not a safety verdict.
            if not result.generations or any(
                not isinstance(generation.text, str) or not generation.text.strip()
                for generation in result.generations
            ):
                raise ValueError("Guard self-check returned empty text; no safety verdict recorded.")
            return result
    llm = PacedChat(model=JUDGE_MODEL, temperature=0, timeout=60,
                     **langchain_provider_kwargs())
    rails = LLMRails(config, llm=llm)
    return rails


def _field(value, name):
    """Read either a native NeMo/Pydantic object or its serialized mapping."""
    try:
        return value[name] if isinstance(value, dict) else getattr(value, name)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Malformed NeMo response: missing {name}.") from exc


def _rail_outcome(result, rail_type: str) -> tuple[bool, str]:
    """Interpret the logged decision, never the words in a generated response.

    NeMo 0.10's GenerationResponse/ActivatedRail schema is defined in
    nemoguardrails/rails/llm/options.py. Its processing_log.py records stop=True
    when an input/output flow halts. Input-only success may have empty content.
    Missing evidence is an error, not permission to bypass the guard.
    """
    activated = _field(_field(result, "log"), "activated_rails")
    if not isinstance(activated, list) or not activated:
        raise ValueError("NeMo did not record any activated rails; refusing to infer safety.")
    decisions = []
    for rail in activated:
        stop = _field(rail, "stop")
        if type(stop) is not bool:
            raise ValueError("Malformed NeMo rail: stop must be a boolean.")
        # Minimal injected rails expose stop only; native records also expose type.
        kind = rail.get("type", rail_type) if isinstance(rail, dict) else getattr(rail, "type", rail_type)
        if not isinstance(kind, str) or not kind:
            raise ValueError("Malformed NeMo rail: invalid type.")
        if kind == rail_type:
            decisions.append(stop)
    if not decisions:
        raise ValueError(f"NeMo did not record an executed {rail_type} rail.")
    blocked = any(decisions)

    response = _field(result, "response")
    if isinstance(response, str):
        content = response
    elif isinstance(response, list):
        messages = []
        for message in response:
            if (not isinstance(message, dict)
                    or message.get("role", "assistant") != "assistant"
                    or not isinstance(message.get("content"), str)):
                raise ValueError("NeMo returned a malformed or exception response.")
            messages.append(message["content"])
        content = "\n".join(messages)
    else:
        raise ValueError("NeMo returned neither text nor a list of messages.")
    if blocked and not content.strip():
        raise ValueError("NeMo stopped processing without a refusal response.")
    return blocked, content


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        rails = setup_nemo_rails()
    result = await rails.generate_async(messages=[{"role": "user", "content": text}],
                                        options={"rails": ["input"], "log": {"activated_rails": True}})
    blocked, response = _rail_outcome(result, "input")
    return {"allowed": not blocked, "blocked_reason": "nemo_input_rail" if blocked else None, "response": response}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra PII và nội dung nhạy cảm. Self-check này không thay RAGAS faithfulness
    hoặc kiểm tra factual accuracy với retrieved contexts.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        rails = setup_nemo_rails()
    result = await rails.generate_async(messages=[{"role": "user", "content": question},
                                                   {"role": "assistant", "content": answer}],
                                        options={"rails": ["output"], "log": {"activated_rails": True}})
    blocked, response = _rail_outcome(result, "output")
    return {"safe": not blocked, "flagged_reason": "nemo_output_rail" if blocked else None,
            "final_answer": response if blocked else answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii, không dùng category/expected
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()
    async def run_all():
        rows = []
        for item in adversarial_set:
            pii = pii_scan(item["input"], analyzer, anonymizer)
            blocked_by = "presidio" if pii["has_pii"] else None
            if blocked_by is None:
                rail = await check_input_rail(item["input"], rails)
                if not rail["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            rows.append({"id": item["id"], "category": item["category"], "input": item["input"],
                         "expected": item["expected"], "actual": actual, "blocked_by": blocked_by,
                         "passed": actual == item["expected"]})
        return rows
    return asyncio.run(run_all())


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: measure INPUT layers sequentially, not end-to-end guard/RAG latency.

    Reuse engines. Inside ONE event loop, execute two real warm-up calls per layer,
    excluded from samples. Then collect exactly n_runs timed samples (cycle clean
    inputs), always measuring BOTH PII and NeMo input even if PII blocks.
    Use perf_counter; keep raw milliseconds. summarize_latency() supplies linear
    percentiles at (n - 1) * q. Submission requires n_runs >= 20.
    Short-circuit request timing, RAG and output timing are outside this scope.
    """
    from src.experiment_support import summarize_latency
    if (not isinstance(test_inputs, list) or not test_inputs
            or any(not isinstance(text, str) or not text.strip() for text in test_inputs)):
        raise ValueError("Need a non-empty list of non-blank latency test inputs.")
    if type(n_runs) is not int or n_runs < 1:
        raise ValueError("n_runs must be a positive integer (submission requires >= 20).")
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()
    async def measure():
        for i in range(2):
            text = test_inputs[i % len(test_inputs)]
            pii_scan(text, analyzer, anonymizer)
            await check_input_rail(text, rails)
        samples = []
        for i in range(n_runs):
            text = test_inputs[i % len(test_inputs)]
            started = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            pii_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - started) * 1000
            if any(not math.isfinite(value) or value <= 0 for value in (pii_ms, nemo_ms)):
                raise ValueError("Timing clock must produce positive, finite raw latency samples.")
            samples.append({"presidio_ms": pii_ms, "nemo_input_ms": nemo_ms, "input_guard_ms": pii_ms + nemo_ms})
        return samples
    return summarize_latency(asyncio.run(measure()), warmup_runs=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "c", *sys.argv[1:]]))
