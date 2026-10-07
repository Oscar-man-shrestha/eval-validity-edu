"""Guards for the final, evidence-backed project scope."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs_junyi" / "phases" / "research_integrity_audit.json"


def test_audit_marks_local_sequence_probes_as_exploratory():
    audit = json.loads(AUDIT.read_text())
    probe = audit["exploratory_sequence_probes"]
    assert probe["status"] == "not_comparable_to_published_GRU4Rec_or_SASRec"
    assert probe["reason"]["published_gru4rec_ranking_loss_present"] is False


def test_audit_uses_practical_thresholds_for_final_claims():
    audit = json.loads(AUDIT.read_text())
    results = audit["validated_findings"]
    assert results["ragr_vs_recent5"]["status"] == "not_practically_superior"
    assert results["readiness_correctness"]["status"] == "detectable_but_negligible"
    assert results["forgetting_gap_ge_1_day"]["status"] == "unresolved"


def test_canonical_report_does_not_call_the_prototype_a_winner():
    text = (ROOT / "docs" / "NeuroTrace-DAG_Paper.md").read_text().lower()
    assert "exploratory diagnostic" in text
    assert "state-of-the-art" not in text
