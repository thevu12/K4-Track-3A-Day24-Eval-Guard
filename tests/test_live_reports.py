"""Optional integration batch: paid/network calls only with --run-live."""
import pytest
from config import (LLM_API_KEY,ADVERSARIAL_SET_PATH,CLEAN_SET_PATH,OUTPUT_SET_PATH,
                    ANSWERS_PATH,TEST_SET_PATH,HUMAN_LABELS_PATH)
from src.experiment_support import read_json,require_completed
from src.experiment_runners import run_phase_b,run_phase_c
from src.report_contracts import validate_phase_b,validate_phase_c

pytestmark=pytest.mark.integration


@pytest.fixture(scope="module")
def live_batch(request):
    if not request.config.getoption("--run-live") or not LLM_API_KEY:
        pytest.skip("Requires --run-live and a provider key; may incur API costs.")
    from src import phase_b_judge,phase_c_guard
    require_completed(phase_b_judge)
    require_completed(phase_c_guard)
    return run_phase_b(),run_phase_c()  # exactly once each for every test in this module


def test_live_judge(live_batch):
    validate_phase_b(live_batch[0],read_json(ANSWERS_PATH),read_json(TEST_SET_PATH),read_json(HUMAN_LABELS_PATH))


def test_live_guards(live_batch):
    validate_phase_c(live_batch[1],read_json(ADVERSARIAL_SET_PATH),read_json(CLEAN_SET_PATH),read_json(OUTPUT_SET_PATH))


def test_live_bonus_evidence_is_countable(live_batch):
    assert live_batch[1]["adversarial"]["total"]==20
    assert 0 <= live_batch[1]["clean"]["false_positive_rate"] <= 1
