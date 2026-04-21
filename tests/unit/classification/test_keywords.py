"""Unit tests for tidus.classification.keywords — topic detection + veto."""
from __future__ import annotations

from tidus.classification.keywords import complexity_veto, flatten, match


class TestMatch:
    def test_medical_diagnosis_detected(self):
        hits = match("Can you help me diagnose my symptoms?")
        assert "medical" in hits
        assert "diagnose" in hits["medical"]
        assert "symptoms" in hits["medical"]

    def test_legal_attorney_detected(self):
        hits = match("I need an attorney to review this NDA")
        assert "legal" in hits
        assert "attorney" in hits["legal"]
        assert "nda" in hits["legal"]

    def test_financial_tax_return_detected(self):
        hits = match("Help me understand my tax return W-2 earnings")
        assert "financial" in hits
        assert "tax return" in hits["financial"]
        assert "w-2" in hits["financial"]

    def test_hr_complaint_detected(self):
        hits = match("filing an employee complaint about harassment")
        assert "hr" in hits
        assert "employee complaint" in hits["hr"]
        assert "harassment" in hits["hr"]

    def test_benign_text_no_hits(self):
        hits = match("what's a good recipe for chocolate chip cookies?")
        assert hits == {}

    def test_case_insensitive(self):
        hits = match("ATTORNEY-client PRIVILEGE covers NDAs")
        assert "legal" in hits
        # Canonicalized to lowercase
        assert "attorney" in hits["legal"]
        assert "privilege" in hits["legal"]


class TestFlatten:
    def test_flat_format(self):
        hits = {"medical": ["diagnose"], "legal": ["attorney"]}
        flat = flatten(hits)
        assert "medical:diagnose" in flat
        assert "legal:attorney" in flat

    def test_empty_dict(self):
        assert flatten({}) == []


class TestComplexityVeto:
    def test_medical_forces_critical(self):
        assert complexity_veto({"medical": ["diagnose"]}) == "critical"

    def test_legal_forces_complex(self):
        assert complexity_veto({"legal": ["attorney"]}) == "complex"

    def test_financial_forces_complex(self):
        assert complexity_veto({"financial": ["tax return"]}) == "complex"

    def test_hr_forces_complex(self):
        assert complexity_veto({"hr": ["harassment"]}) == "complex"

    def test_no_hits_returns_none(self):
        assert complexity_veto({}) is None

    def test_medical_wins_over_legal(self):
        # If both hit, medical is the more serious escalation
        assert complexity_veto({"medical": ["symptom"], "legal": ["attorney"]}) == "critical"
