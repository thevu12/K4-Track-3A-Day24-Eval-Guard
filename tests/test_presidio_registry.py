"""Offline wiring tests: policy guard registers identifiers, not generic NER."""
import re
import sys
from types import SimpleNamespace

import pytest

from src import phase_c_guard as phase


@pytest.fixture
def fake_presidio(monkeypatch):
    class Registry:
        def __init__(self, supported_languages):
            self.supported_languages = supported_languages
            self.recognizers = []

        def add_recognizer(self, recognizer):
            self.recognizers.append(recognizer)

        def load_predefined_recognizers(self):
            pytest.fail("Do not register all English NER/country-specific recognizers")

    def pattern_recognizer(**kwargs):
        return SimpleNamespace(supported_entities=[kwargs["supported_entity"]], **kwargs)

    def email_recognizer(**kwargs):
        return SimpleNamespace(supported_entities=["EMAIL_ADDRESS"], **kwargs)

    def phone_recognizer(**kwargs):
        return SimpleNamespace(supported_entities=["PHONE_NUMBER"], **kwargs)

    analyzer_calls = []
    def analyzer_factory(**kwargs):
        analyzer_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    anonymizer = object()
    monkeypatch.setitem(sys.modules, "presidio_analyzer", SimpleNamespace(
        AnalyzerEngine=analyzer_factory, RecognizerRegistry=Registry,
        Pattern=lambda name, regex, score: SimpleNamespace(name=name, regex=regex, score=score),
        PatternRecognizer=pattern_recognizer,
    ))
    monkeypatch.setitem(sys.modules, "presidio_analyzer.predefined_recognizers", SimpleNamespace(
        EmailRecognizer=email_recognizer, PhoneRecognizer=phone_recognizer,
    ))
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", SimpleNamespace(AnonymizerEngine=lambda: anonymizer))
    return analyzer_calls, anonymizer


def test_registry_contains_only_intended_identifiers(fake_presidio):
    calls, expected_anonymizer = fake_presidio
    analyzer, anonymizer = phase.setup_presidio()
    entities = {entity for rec in analyzer.registry.recognizers for entity in rec.supported_entities}
    assert entities == {"VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    assert len(analyzer.registry.recognizers) == 4
    assert anonymizer is expected_anonymizer
    assert calls[0]["supported_languages"] == [phase.PRESIDIO_LANGUAGE]
    phone = next(rec for rec in analyzer.registry.recognizers if rec.supported_entities == ["PHONE_NUMBER"])
    assert phone.supported_regions == ["VN"]


def test_registry_language_is_consistent(fake_presidio, monkeypatch):
    monkeypatch.setattr(phase, "PRESIDIO_LANGUAGE", "en")
    analyzer, _ = phase.setup_presidio()
    assert analyzer.registry.supported_languages == analyzer.supported_languages == ["en"]
    assert all(rec.supported_language == "en" for rec in analyzer.registry.recognizers)


def test_custom_patterns_keep_ids_but_do_not_match_dates_or_policy_versions(fake_presidio):
    analyzer, _ = phase.setup_presidio()
    custom = {rec.supported_entity: rec for rec in analyzer.registry.recognizers if hasattr(rec, "patterns")}
    assert any(re.search(p.regex, "034095001234") for p in custom["VN_CCCD"].patterns)
    assert any(re.search(p.regex, "012345678") for p in custom["VN_CCCD"].patterns)
    assert any(re.search(p.regex, "0987654321") for p in custom["VN_PHONE"].patterns)
    safe = "Policy v2024, v2.0, effective 01/01/2024; Senior/Junior salary ranges, VPN and MFA."
    assert not any(re.search(p.regex, safe) for rec in custom.values() for p in rec.patterns)
