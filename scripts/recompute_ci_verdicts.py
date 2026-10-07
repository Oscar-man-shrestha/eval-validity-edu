#!/usr/bin/env python3
"""Recompute paper_paired_verdicts.json with CI-threshold verdict rule (no 'tie')."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junyi_pipeline import OUT

PHASE = OUT / "phases"
T_R5 = 0.01
T_AUC = 0.005


def ci_verdict(ci: list | None, t: float) -> dict:
    """CI rule (no 'tie'):
    - supported_A if entire CI > +t  (lo > t)
    - supported_B if entire CI < -t  (hi < -t)
    - negligible if entire CI ⊂ (−t, +t)  (lo > −t and hi < t)
    - inconclusive otherwise
    """
    if not ci or ci[0] is None or ci[1] is None:
        return {"verdict": "unavailable", "lo": None, "hi": None, "t": t}
    lo, hi = float(ci[0]), float(ci[1])
    if lo > t:
        v = "supported_A"
    elif hi < -t:
        v = "supported_B"
    elif lo > -t and hi < t:
        v = "negligible"
    else:
        v = "inconclusive"
    return {"verdict": v, "lo": lo, "hi": hi, "t": t}


def annotate(block: dict, t: float, a: str, b: str) -> dict:
    out = dict(block)
    cv = ci_verdict(block.get("ci"), t)
    out["verdict"] = cv["verdict"]
    out["verdict_rule"] = (
        f"supported_A if CI entirely > +{t}; supported_B if entirely < −{t}; "
        f"negligible if entirely inside (−{t},+{t}); else inconclusive"
    )
    out["A"] = a
    out["B"] = b
    out.pop("call", None)  # remove old tie/A/B label
    return out


def main() -> None:
    full = json.loads((PHASE / "paper_ranking_full.json").read_text())
    adv_tmp_path = PHASE / "paper_advance_paired_tmp.json"
    if adv_tmp_path.exists():
        adv_pack = json.loads(adv_tmp_path.read_text())
    else:
        adv_pack = (json.loads((PHASE / "paper_paired_verdicts.json").read_text())
                    .get("junyi_ranking_full_test", {})
                    .get("advance_slice_recomputed") or {})

    p3 = json.loads((PHASE / "phase3_memory_fulltest.json").read_text())
    gru_mk = json.loads((PHASE / "gru_markov_paired_ci.json").read_text())

    existing = full.get("paired_deltas") or {}
    overall = {
        "ragr_minus_recent5_r5": annotate(
            existing.get("ragr_minus_recent5_r5") or {}, T_R5, "RAGR", "Recent-5"
        ),
        "ragr_minus_markov_r5": annotate(
            existing.get("ragr_minus_markov_r5") or {}, T_R5, "RAGR", "Markov"
        ),
        "ragr_minus_ragr_no_dag_r5": annotate(
            existing.get("ragr_minus_ragr_no_dag_r5") or {}, T_R5, "RAGR", "RAGR-no-DAG"
        ),
    }

    paired = dict(adv_pack.get("paired") or {})
    # Drop advance Recent-5 contrast (mechanical near-zero)
    paired.pop("ragr_minus_recent5_r5_advance", None)
    paired_ann = {}
    name_map = {
        "ragr_minus_recent5_r5_all": ("RAGR", "Recent-5"),
        "ragr_minus_ragr_no_dag_r5_all": ("RAGR", "RAGR-no-DAG"),
        "ragr_minus_markov_r5_all": ("RAGR", "Markov"),
        "ragr_minus_ragr_no_dag_r5_advance": ("RAGR", "RAGR-no-DAG"),
        "ragr_minus_markov_r5_advance": ("RAGR", "Markov"),
    }
    for k, block in paired.items():
        if k not in name_map:
            continue
        a, b = name_map[k]
        paired_ann[k] = annotate(block, T_R5, a, b)

    phase3 = p3["evaluate_on_test_json"]["paired_logistic_step_minus_success"]
    phase3_block = annotate(
        {
            "mean": phase3["mean_delta_auc"],
            "ci": phase3["ci"],
            "n_test_rows": p3["evaluate_on_test_json"]["n_test"],
            "bootstrap": phase3.get("bootstrap"),
        },
        T_AUC,
        "logistic_step_dt",
        "success_rate",
    )

    gru_rows = []
    for ds, units in (gru_mk.get("datasets") or {}).items():
        if not isinstance(units, dict):
            continue
        for unit in ("native", "cluster"):
            block = units.get(unit)
            if not isinstance(block, dict):
                continue
            nr = block.get("nonrepeat")
            if not isinstance(nr, dict):
                nr = block.get("all")
            if not isinstance(nr, dict) or "mean_delta_gru_minus_markov" not in nr:
                continue
            row = annotate(
                {
                    "mean": nr["mean_delta_gru_minus_markov"],
                    "ci": nr["ci"],
                    "n_sequences": nr.get("n_sequences"),
                    "mean_gru": nr.get("mean_gru"),
                    "mean_markov": nr.get("mean_markov"),
                },
                T_R5,
                "GRU",
                "Markov",
            )
            row["dataset"] = ds
            row["unit"] = unit
            row["slice"] = "nonrepeat" if isinstance(block.get("nonrepeat"), dict) else "all"
            # primary H-rev uses these nonrepeat deltas
            gru_rows.append(row)

    # Within-dataset native vs cluster sign flip (exploratory summary aid)
    h_rev_primary = {}
    by_ds = {}
    for r in gru_rows:
        by_ds.setdefault(r["dataset"], {})[r["unit"]] = r
    for ds, u in by_ds.items():
        if "native" in u and "cluster" in u:
            n, c = u["native"], u["cluster"]
            h_rev_primary[ds] = {
                "native_verdict": n["verdict"],
                "cluster_verdict": c["verdict"],
                "native_mean": n["mean"],
                "cluster_mean": c["mean"],
                "sign_flip": (n["mean"] > 0) != (c["mean"] > 0),
                "both_supported": n["verdict"].startswith("supported_")
                and c["verdict"].startswith("supported_"),
                "primary_h_rev_claim_supported": (
                    (n["mean"] > 0) != (c["mean"] > 0)
                    and n["verdict"].startswith("supported_")
                    and c["verdict"].startswith("supported_")
                ),
            }

    out = {
        "status": "EXPLORATORY_pre_prereg",
        "prereg_v1_tagged": False,
        "thresholds": {"recall_at_5": T_R5, "auc": T_AUC},
        "verdict_rule": (
            f"supported_A if CI entirely > +t; supported_B if CI entirely < −t; "
            f"negligible if CI entirely inside (−t,+t); else inconclusive. "
            f"R@5 t={T_R5}; AUC t={T_AUC}. No 'tie' label."
        ),
        "junyi_ranking_full_test": {
            "from_paper_ranking_full_paired_deltas": overall,
            "advance_and_all_recomputed": {
                "eval": adv_pack.get("eval"),
                "paired": paired_ann,
                "note": "ragr_minus_recent5_r5_advance dropped (mechanical near-zero on advance)",
            },
        },
        "phase3_logistic_minus_success": phase3_block,
        "gru_minus_markov": {
            "source": "gru_markov_paired_ci.json",
            "rows": gru_rows,
            "h_rev_primary_native_vs_cluster": h_rev_primary,
            "note": (
                "Validation paired CIs (freeze-era). Primary H-rev = sign(GRU−Markov) differs "
                "native vs cluster with both verdicts supported_*. Confirmatory: freeze test "
                "partition after prereg-v1 (see docs/PREREGISTRATION_REPLICATION.md)."
            ),
        },
        "exploratory_summary_for_pdf": {
            "slice_reversals_flagged_under_h_rev_required_models": False,
            "xes_native_vs_cluster_sign_flip": h_rev_primary.get("xes3g5m", {}).get(
                "sign_flip"
            ),
            "xes_primary_h_rev_supported": h_rev_primary.get("xes3g5m", {}).get(
                "primary_h_rev_claim_supported"
            ),
            "markov_beats_ragr_on_advance": paired_ann.get(
                "ragr_minus_markov_r5_advance", {}
            ).get("verdict")
            == "supported_B",
            "dag_features_negligible": paired_ann.get(
                "ragr_minus_ragr_no_dag_r5_all", {}
            ).get("verdict")
            == "negligible",
            "phase3_logistic_vs_success": phase3_block["verdict"],
        },
    }

    path = PHASE / "paper_paired_verdicts.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path)
    print("phase3", phase3_block["verdict"], phase3_block["ci"])
    print("advance RAGR-Markov", paired_ann.get("ragr_minus_markov_r5_advance", {}).get("verdict"))
    print("DAG", paired_ann.get("ragr_minus_ragr_no_dag_r5_all", {}).get("verdict"))
    print("h_rev_primary", json.dumps(h_rev_primary, indent=2))


if __name__ == "__main__":
    main()
