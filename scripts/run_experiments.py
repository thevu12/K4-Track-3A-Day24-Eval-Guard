"""Run submitted experiments explicitly. Default tests never call paid APIs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LLM_API_KEY
from src.experiment_support import require_completed
from src.experiment_runners import run_phase_a, run_phase_b, run_phase_c


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG eval/guard experiments (paid API, network required)")
    parser.add_argument("--phase", choices=("a", "b", "c", "all"), required=True)
    parser.add_argument("--live", action="store_true", help="Acknowledge API use and network dependency")
    parser.add_argument("--latency-samples", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Real experiments require --live. Use python -m pytest for offline unit tests.")
    if not LLM_API_KEY:
        parser.error("Set the configured provider API key in .env; never put the key in source.")
    from src import phase_a_ragas, phase_b_judge, phase_c_guard
    modules = {"a": phase_a_ragas, "b": phase_b_judge, "c": phase_c_guard}
    selected = list(modules) if args.phase == "all" else [args.phase]
    for name in selected:
        require_completed(modules[name])
    runners = {"a": run_phase_a, "b": run_phase_b, "c": lambda: run_phase_c(args.latency_samples)}
    for name in selected:
        runners[name]()
        print(f"Phase {name.upper()} saved to artifacts/.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit(f"Experiment failed: {error}")
