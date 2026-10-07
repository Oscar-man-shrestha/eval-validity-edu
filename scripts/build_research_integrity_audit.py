"""Build a single, machine-readable source of truth for final project claims.

This script deliberately distinguishes (a) validated findings, (b) exploratory
probes, and (c) claims that must not appear in the final paper or demo. It reads
the existing experiment JSON and the probe implementation configuration, then
writes ``outputs_junyi/phases/research_integrity_audit.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs_junyi" / "phases"
OUT = PHASE / "research_integrity_audit.json"
PRACTICAL_R5 = 0.01


def load(name: str) -> dict:
    return json.loads((PHASE / name).read_text())


def ci_contains_zero(metric: dict) -> bool:
    lo, hi = metric.get("ci", [None, None])
    return lo is not None and lo <= 0 <= hi


def main() -> None:
    ranking = load("review_ranking.json")
    readiness = load("review_readiness.json")
    forgetting = load("followup3_forgetting.json")
    concept = load("followup3_concept_repetition.json")
    follow2 = load("followup2_ranking_reconcile.json")
    rank_probe = load("followup3_rank_reversal.json")

    recent_delta = follow2["canonical_run"]["paired_ragr_minus_recent5_r5"]
    ragr_r5 = ranking["metrics"]["ragr_r5"]["mean"]
    recent_r5 = ranking["metrics"]["recent5_r5"]["mean"]

    primary_readiness = readiness["dag_helps_correctness"]["primary_delta_auc"]
    practical_ready = readiness["dag_helps_correctness"]["practical_threshold_delta_auc"]

    source = (ROOT / "run_followup3.py").read_text()
    epochs = re.search(r"epochs=(\d+)", source)
    cap = re.search(r"pairs = pairs\[:([\d_]+)\]", source)
    has_ce = "nn.CrossEntropyLoss" in source
    has_original_ranking_loss = "BPR" in source or "TOP1" in source

    audit = {
        "schema_version": 1,
        "positioning": {
            "final_project_type": "multi-dataset evaluation-validity and protocol study",
            "permitted_method_claim": "exploratory next-action prototype",
            "prohibited_method_claims": [
                "state-of-the-art recommender",
                "RAGR practically outperforms simple recency baselines",
                "the current probe is a faithful GRU4Rec or SASRec reproduction",
                "DAG readiness provides a practically meaningful correctness gain",
                "forgetting features improve prediction on the tested data",
            ],
        },
        "validated_findings": {
            "cross_dataset_concept_repetition": concept,
            "ragr_vs_recent5": {
                "ragr_r5": ragr_r5,
                "recent5_r5": recent_r5,
                "paired_delta_r5": recent_delta,
                "practical_threshold_r5": PRACTICAL_R5,
                "status": (
                    "practically_superior"
                    if recent_delta["mean"] >= PRACTICAL_R5
                    and recent_delta.get("ci", [None, None])[0] is not None
                    and recent_delta["ci"][0] > 0
                    else "not_practically_superior"
                ),
            },
            "readiness_correctness": {
                "primary": primary_readiness,
                "practical_threshold_delta_auc": practical_ready,
                "status": readiness["dag_helps_correctness"]["status"],
            },
            "forgetting_gap_ge_1_day": {
                "status": forgetting["status"],
                "auc": forgetting["auc"],
                "delta_log_gap_minus_success": forgetting["paired_deltas"]["log1p_gap_minus_success"],
                "n_test_rows": forgetting["n_test_rows"],
                "n_test_learners": forgetting["n_test_learners"],
            },
        },
        "exploratory_sequence_probes": {
            "status": "not_comparable_to_published_GRU4Rec_or_SASRec",
            "reason": {
                "epochs": int(epochs.group(1)) if epochs else None,
                "training_pair_cap": int(cap.group(1).replace("_", "")) if cap else None,
                "loss": "cross_entropy" if has_ce else "unknown",
                "published_gru4rec_ranking_loss_present": has_original_ranking_loss,
                "result_file": "followup3_rank_reversal.json",
            },
            "allowed_label": "GRU-sequence probe / attention-sequence probe",
            "prohibited_labels": ["GRU4Rec baseline", "SASRec baseline", "deep benchmark winner"],
        },
        "references": [
            {
                "key": "repeatnet",
                "citation": "Ren et al. (2019), RepeatNet: A Repeat Aware Neural Recommendation Machine.",
                "url": "https://ojs.aaai.org/index.php/AAAI/article/view/4408",
                "use": "Repeat/explore gating predates RAGR; cite it instead of claiming the gate as novel.",
            },
            {
                "key": "gru4rec",
                "citation": "Hidasi et al. (2016), Session-based Recommendations with Recurrent Neural Networks.",
                "url": "https://hidasi.eu/assets/pdf/gru4rec_iclr16.pdf",
                "use": "Original GRU4Rec uses a session-recommendation ranking protocol; the local two-epoch probe is not a reproduction.",
            },
            {
                "key": "sasrec",
                "citation": "Kang and McAuley (2018), Self-Attentive Sequential Recommendation.",
                "url": "https://arxiv.org/abs/1808.09781",
                "use": "Original SASRec is a separate validated method; the local attention probe is not a published SASRec baseline.",
            },
        ],
        "rendering_rules": {
            "numbers_must_come_from": "outputs_junyi/phases/*.json",
            "legacy_documents_are_historical": True,
            "do_not_use_rank_probe_as_final_comparison": True,
        },
    }
    OUT.write_text(json.dumps(audit, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
