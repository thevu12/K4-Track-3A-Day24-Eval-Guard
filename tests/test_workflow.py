"""Offline tests for preflight and resumable orchestration; no provider calls."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import complete_lab as workflow
from scripts import preflight


pytestmark = pytest.mark.support


@pytest.fixture
def ready_preflight(monkeypatch):
    configuration = SimpleNamespace(
        OPENAI_API_KEY="private-test-value", LLM_API_KEY="private-test-value",
        LLM_PROVIDER="openai", QDRANT_HOST="localhost", QDRANT_PORT=6333)
    monkeypatch.setattr(preflight, "DEPENDENCIES", (("example_module", "example-package"),))
    monkeypatch.setattr(preflight, "check_import", lambda *args: preflight.Check("dependency", "PASS", "import OK"))
    monkeypatch.setattr(preflight, "check_spacy_model", lambda: preflight.Check("spaCy", "PASS", "installed"))
    monkeypatch.setattr(preflight.importlib, "import_module", Mock(return_value=configuration))
    # sys.version_info supports both attributes and slicing.
    class Version(tuple):
        major, minor, micro = 3, 11, 16
    monkeypatch.setattr(preflight.sys, "version_info", Version((3, 11, 16)))
    health = Mock(return_value=preflight.Check("Qdrant", "PASS", "healthy"))
    monkeypatch.setattr(preflight, "check_qdrant", health)
    return configuration, health


def test_preflight_default_does_not_use_network_or_expose_key(ready_preflight, capsys):
    configuration, health = ready_preflight
    assert preflight.main([]) == 0
    health.assert_not_called()
    output = capsys.readouterr().out
    assert "SKIP Qdrant" in output
    assert "OPENAI API key: present" in output
    assert configuration.OPENAI_API_KEY not in output
    assert "No provider API calls" in output


def test_preflight_live_only_uses_configured_health_probe(ready_preflight):
    configuration, health = ready_preflight
    assert preflight.main(["--live"]) == 0
    health.assert_called_once_with(configuration.QDRANT_HOST, configuration.QDRANT_PORT)


@pytest.mark.parametrize("key", ["", " ", None])
def test_preflight_fails_for_missing_key(ready_preflight, key):
    configuration, _ = ready_preflight
    configuration.LLM_API_KEY = key
    assert preflight.main([]) == 1


def test_preflight_rejects_wrong_python(ready_preflight, monkeypatch):
    class Version(tuple):
        major, minor, micro = 3, 13, 0
    monkeypatch.setattr(preflight.sys, "version_info", Version((3, 13, 0)))
    assert preflight.main([]) == 1


def test_preflight_import_failure_does_not_print_exception_secrets(monkeypatch):
    monkeypatch.setattr(preflight.importlib, "import_module", Mock(side_effect=ImportError("sensitive exception contents")))
    result = preflight.check_import("module", "package")
    assert result.status == "FAIL"
    assert "ImportError" in result.detail
    assert "sensitive" not in result.detail


def test_spacy_check_does_not_load_or_download_model(monkeypatch):
    find = Mock(return_value=object())
    monkeypatch.setattr(preflight.importlib.util, "find_spec", find)
    load = Mock(side_effect=AssertionError("must not import the model"))
    monkeypatch.setattr(preflight.importlib, "import_module", load)
    assert preflight.check_spacy_model().status == "PASS"
    find.assert_called_once_with("en_core_web_lg")
    load.assert_not_called()


def test_qdrant_preflight_is_read_only(monkeypatch):
    response = SimpleNamespace(status=200, read=Mock(return_value=b"healthz check passed"))
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    open_url = Mock(return_value=context)
    monkeypatch.setattr(preflight, "urlopen", open_url)
    result = preflight.check_qdrant("localhost", 6333)
    assert result.status == "PASS"
    request = open_url.call_args.args[0]
    assert request.full_url == "http://localhost:6333/healthz"
    assert request.method == "GET" and request.data is None
    assert open_url.call_args.kwargs == {"timeout": 5}


@pytest.mark.parametrize("host", ["http://example.test", "user:secret@example.test", "example.test/path", "example.test?key=secret"])
def test_qdrant_rejects_embedded_credentials_or_non_host_config(monkeypatch, host):
    opened = Mock(side_effect=AssertionError("Unexpected request"))
    monkeypatch.setattr(preflight, "urlopen", opened)
    result = preflight.check_qdrant(host, 6333)
    assert result.status == "FAIL" and "secret" not in result.detail
    opened.assert_not_called()


@pytest.fixture
def ready_workflow(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/answers_50q.json").touch()
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "api_key_present", Mock(return_value=True))
    monkeypatch.setattr(workflow, "validate_provider_access", Mock())
    answers = [{"generation_mode": "llm"}]
    answer_validation = Mock(return_value=answers)
    monkeypatch.setattr(workflow, "validate_saved_answers", answer_validation)
    phase_validation = Mock()
    monkeypatch.setattr(workflow, "validate_saved_phase", phase_validation)
    commands = Mock()
    monkeypatch.setattr(workflow, "run_script", commands)
    return tmp_path, commands, answer_validation, phase_validation


def command_args(commands):
    return [call.args[1:] for call in commands.call_args_list]


def test_completion_requires_explicit_live_before_commands(ready_workflow):
    _, commands, _, _ = ready_workflow
    with pytest.raises(SystemExit) as failure:
        workflow.main([])
    assert failure.value.code != 0
    commands.assert_not_called()


def test_completion_missing_key_fails_before_any_subprocess(ready_workflow, monkeypatch, capsys):
    _, commands, _, _ = ready_workflow
    monkeypatch.setattr(workflow, "api_key_present", Mock(return_value=False))
    assert workflow.main(["--live"]) == 1
    commands.assert_not_called()
    assert "provider API key is missing" in capsys.readouterr().err


def test_completion_requires_twenty_latency_samples(ready_workflow):
    _, commands, _, _ = ready_workflow
    with pytest.raises(SystemExit) as failure:
        workflow.main(["--live", "--latency-samples", "19"])
    assert failure.value.code != 0
    commands.assert_not_called()


@pytest.mark.parametrize("samples", [19, True, 20.5, "20"])
def test_programmatic_workflow_also_rejects_invalid_latency(ready_workflow, samples):
    root, commands, _, _ = ready_workflow
    with pytest.raises(workflow.WorkflowError, match="at least 20"):
        workflow.complete(root, regenerate=False, rerun=False, latency_samples=samples)
    commands.assert_not_called()


def test_completion_reuses_all_valid_evidence(ready_workflow):
    root, commands, answers, phases = ready_workflow
    assert workflow.main(["--live"]) == 0
    assert command_args(commands) == [
        ("preflight.py", "--live"), ("render_reports.py", "--overwrite"),
        ("check_lab.py", "--with-tests"),
    ]
    assert answers.call_count == 2
    assert [call.args for call in phases.call_args_list] == [(root, phase) for phase in ("a", "b", "c") * 2]


def test_completion_only_runs_stale_phase(ready_workflow):
    _, commands, _, phases = ready_workflow
    phases.side_effect = [None, ValueError("stale hash"), None, None, None, None]
    assert workflow.main(["--live", "--latency-samples", "24"]) == 0
    assert command_args(commands) == [
        ("preflight.py", "--live"),
        ("run_experiments.py", "--phase", "b", "--live", "--latency-samples", "24"),
        ("render_reports.py", "--overwrite"), ("check_lab.py", "--with-tests"),
    ]


def test_completion_uses_all_runner_when_all_evidence_missing(ready_workflow):
    _, commands, _, phases = ready_workflow
    phases.side_effect = [FileNotFoundError(), FileNotFoundError(), FileNotFoundError(), None, None, None]
    assert workflow.main(["--live"]) == 0
    assert command_args(commands)[1] == (
        "run_experiments.py", "--phase", "all", "--live", "--latency-samples", "20")


def test_completion_generates_missing_answers(ready_workflow):
    root, commands, _, _ = ready_workflow
    (root / "artifacts/answers_50q.json").unlink()
    assert workflow.main(["--live"]) == 0
    assert command_args(commands)[1] == ("setup_answers.py", "--live")


def test_regenerate_is_explicit_overwrite(ready_workflow):
    _, commands, _, _ = ready_workflow
    assert workflow.main(["--live", "--regenerate"]) == 0
    assert command_args(commands)[1] == ("setup_answers.py", "--live", "--overwrite")


def test_rerun_is_explicit_even_when_evidence_valid(ready_workflow):
    _, commands, _, phases = ready_workflow
    assert workflow.main(["--live", "--rerun"]) == 0
    assert command_args(commands)[1] == (
        "run_experiments.py", "--phase", "all", "--live", "--latency-samples", "20")
    assert phases.call_count == 3  # post-run checks only; --rerun bypasses cache decisions


def test_completion_preserves_valid_extractive_answers_with_warning(ready_workflow, capsys):
    _, commands, answers, _ = ready_workflow
    answers.return_value = [{"generation_mode": "extractive_fallback"}]
    assert workflow.main(["--live"]) == 0
    assert all(args[0] != "setup_answers.py" for args in command_args(commands))
    assert "extractive_fallback" in capsys.readouterr().out


def test_completion_never_silently_overwrites_invalid_answers(ready_workflow):
    _, commands, answers, _ = ready_workflow
    answers.side_effect = ValueError("invalid answer evidence")
    assert workflow.main(["--live"]) == 1
    assert command_args(commands) == [("preflight.py", "--live")]


def test_completion_stops_after_failed_preflight(ready_workflow):
    _, commands, answers, phases = ready_workflow
    commands.side_effect = workflow.WorkflowError("preflight failed")
    assert workflow.main(["--live"]) == 1
    assert command_args(commands) == [("preflight.py", "--live")]
    answers.assert_not_called()
    phases.assert_not_called()


def test_completion_stops_after_invalid_provider_auth(ready_workflow, monkeypatch, capsys):
    _, commands, answers, phases = ready_workflow
    provider_check = Mock(side_effect=workflow.WorkflowError(
        "Provider/model access check failed (AuthenticationError, status=401, code=invalid_api_key)."
    ))
    monkeypatch.setattr(workflow, "validate_provider_access", provider_check)
    assert workflow.main(["--live", "--regenerate"]) == 1
    assert command_args(commands) == [("preflight.py", "--live")]
    provider_check.assert_called_once_with()
    answers.assert_not_called()
    phases.assert_not_called()
    assert "secret-provider-body" not in capsys.readouterr().err


def test_provider_error_summary_never_exposes_body_or_unsafe_code():
    class ProviderFailure(Exception):
        status_code = 401
        code = "invalid_api_key secret-provider-body"

    summary = workflow._safe_provider_error(ProviderFailure("secret-provider-body"))
    assert summary == "ProviderFailure, status=401"
    assert "secret-provider-body" not in summary


def test_completion_does_not_render_unvalidated_new_reports(ready_workflow):
    _, commands, _, phases = ready_workflow
    phases.side_effect = [None, FileNotFoundError(), None, None, ValueError("bad new metrics")]
    assert workflow.main(["--live"]) == 1
    assert [args[0] for args in command_args(commands)] == ["preflight.py", "run_experiments.py"]


def test_completion_stops_at_failed_runner_and_does_not_render(ready_workflow):
    _, commands, _, phases = ready_workflow
    phases.side_effect = [FileNotFoundError(), FileNotFoundError(), FileNotFoundError()]
    commands.side_effect = [None, workflow.WorkflowError("runner failed")]
    assert workflow.main(["--live"]) == 1
    assert [args[0] for args in command_args(commands)] == ["preflight.py", "run_experiments.py"]


def test_completion_propagates_final_checker_quality_failure(ready_workflow):
    _, commands, _, _ = ready_workflow
    commands.side_effect = [None, None, workflow.WorkflowError("quality gate failed")]
    assert workflow.main(["--live"]) == 1
    assert command_args(commands)[-1] == ("check_lab.py", "--with-tests")


def test_subprocesses_use_same_interpreter_and_repo_cwd(tmp_path, monkeypatch):
    process = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(workflow.subprocess, "run", process)
    workflow.run_script(tmp_path, "preflight.py", "--live")
    process.assert_called_once_with([
        workflow.sys.executable, str(tmp_path / "scripts/preflight.py"), "--live"],
        cwd=tmp_path, check=False)


def test_subprocess_nonzero_is_a_workflow_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=7)))
    with pytest.raises(workflow.WorkflowError, match="status 7"):
        workflow.run_script(tmp_path, "run_experiments.py", "--phase", "a", "--live")


@pytest.mark.parametrize("phase,source_names", [
    ("a", ("answers_50q.json", "test_set_50q.json")),
    ("b", ("answers_50q.json", "test_set_50q.json", "human_labels_10q.json")),
    ("c", ("adversarial_set_20.json", "clean_set_8.json", "output_set_4.json")),
])
def test_phase_reuse_checks_exact_metadata_hash_sources(tmp_path, monkeypatch, phase, source_names):
    monkeypatch.setattr(workflow, "read_json", lambda path: {"path": Path(path).name})
    validators = {}
    for name in ("a", "b", "c"):
        validators[name] = Mock()
        monkeypatch.setattr(workflow, "validate_phase_" + name, validators[name])
    metadata_check = Mock()
    monkeypatch.setattr(workflow, "validate_metadata", metadata_check)
    workflow.validate_saved_phase(tmp_path, phase)
    validators[phase].assert_called_once()
    assert [path.name for path in metadata_check.call_args.args[1]] == list(source_names)
    if phase == "c":
        assert validators[phase].call_args.kwargs["require_quality"] is False


def test_reuse_rejects_stale_metadata_even_when_schema_is_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "read_json", lambda path: {})
    monkeypatch.setattr(workflow, "validate_phase_a", Mock())
    monkeypatch.setattr(workflow, "validate_metadata", Mock(side_effect=ValueError("stale source")))
    assert workflow.phase_is_reusable(tmp_path, "a") is False
