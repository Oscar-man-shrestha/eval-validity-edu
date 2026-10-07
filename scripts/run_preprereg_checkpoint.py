#!/usr/bin/env python3
"""Pre-prereg checkpoint: exploratory tags, paired verdicts, rank-reversal diagnostics.

Does NOT create git tag prereg-v1.
"""

from __future__ import annotations

import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import build_heads
from run_claude_fixes import EVAL_POS, N_BOOT, action_label, paired_seq_delta
from run_valid_eval import recall_at, score_features

PHASE = OUT / "phases"
K = 5
FREEZE_SEED = 20261004
THRESH_R5 = 0.01
THRESH_AUC = 0.005


def write_json(name: str, obj: dict) -> None:
    path = PHASE / name
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print("wrote", path, flush=True)


def load(name: str) -> dict:
    return json.loads((PHASE / name).read_text())


def verdict_r5(delta_mean: float | None, ci: list | None) -> str:
    if delta_mean is None or ci is None or ci[0] is None:
        return "unavailable"
    lo, hi = ci
    if abs(delta_mean) < THRESH_R5 or (lo <= 0 <= hi):
        return "tie"
    return "A" if delta_mean > 0 else "B"


def verdict_auc(delta_mean: float | None, ci: list | None) -> str:
    if delta_mean is None or ci is None or ci[0] is None:
        return "unavailable"
    lo, hi = ci
    if abs(delta_mean) < THRESH_AUC or (lo <= 0 <= hi):
        return "tie"
    return "A" if delta_mean > 0 else "B"


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """Kendall tau-b on shared items by rank position (higher rank index = worse)."""
    common = [m for m in order_a if m in order_b]
    if len(common) < 2:
        return None
    ra = {m: i for i, m in enumerate(order_a)}
    rb = {m: i for i, m in enumerate(order_b)}
    xs = [ra[m] for m in common]
    ys = [rb[m] for m in common]
    n = len(common)
    conc = disc = 0
    for i, j in combinations(range(n), 2):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        if dx == 0 or dy == 0:
            continue
        if dx * dy > 0:
            conc += 1
        else:
            disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom else None


# ---------------------------------------------------------------------------
# 1. Mark paper_rank_reversal_full EXPLORATORY
# ---------------------------------------------------------------------------

def mark_exploratory_rank_reversal() -> dict:
    rev = load("paper_rank_reversal_full.json")
    freeze = load("probe_freeze.json")
    status = {
        "status": "EXPLORATORY",
        "prereg_v1_existed_before_run": False,
        "evidence": {
            "git_tags_prereg": [],
            "preregistration_md": "draft only; explicitly not tagged",
            "freeze_seed": FREEZE_SEED,
            "paper_eval_seed_used": int(SEED),
            "n_test_sizes_match_freeze": {
                ds: rev["datasets"][ds]["split"]["n_test"]
                == freeze["datasets"][ds]["split"]["n_test"]
                for ds in rev["datasets"]
            },
            "test_uid_hashes_match_freeze": False,
            "reason_hashes_differ": (
                f"run_paper_eval.py called learner_split(..., seed=junyi_pipeline.SEED={SEED}); "
                f"freeze used seed={FREEZE_SEED}. Same 70/15/15 sizes (900/439/2711) but different learners."
            ),
        },
        "replication_plan": {
            "register_seed": "new seed chosen only after OSF/Zenodo + git tag prereg-v1",
            "steps": [
                "Tag prereg-v1 and register protocol",
                "learner_split with the registered seed (do not reuse 0 or 20261004 for confirmatory test)",
                "Freeze train/val; hash test uids; touch test once for confirmatory H-rev",
                "Recompute rank-reversal + paired CIs on that fresh test only",
            ],
        },
        "h_rev_reversal_rule": {
            "required_models": ["popularity", "recency", "markov_order1", "gru_probe"],
            "recency_on_advance": "N/A (exclude from advance orderings; not 0.000)",
            "flag_reversal_only_if": (
                "ordering of required models on slice differs from all-query ordering "
                "AND Kendall tau < 1 with bootstrap flip probability support (see diagnostics)"
            ),
        },
    }
    rev["status"] = "EXPLORATORY"
    rev["exploratory_disclosure"] = status
    rev["seed_note"] = (
        f"EXPLORATORY: evaluated with seed={SEED}; freeze partition seed={FREEZE_SEED}; "
        "sizes match freeze n_test but uid hashes do not."
    )

    # Fix advance tables: recency → N/A; exclude from advance orderings; Kendall + flip placeholders
    for ds, pack in rev["datasets"].items():
        ranks = pack["rankings_by_slice_r5"]
        # rewrite advance with recency N/A
        adv = []
        for row in ranks.get("advance") or []:
            if row["method"] == "recency":
                adv.append({
                    "method": "recency",
                    "r5": None,
                    "r5_display": "N/A",
                    "n_sequences": row.get("n_sequences"),
                    "note": "recency undefined / not applicable on advance (next never in history)",
                })
            else:
                adv.append(row)
        # sort advance excluding recency for ordering
        adv_orderable = [r for r in adv if r["method"] != "recency" and r.get("r5") is not None]
        adv_orderable.sort(key=lambda r: r["r5"], reverse=True)
        ranks["advance"] = adv
        ranks["advance_ordering_excluding_recency"] = [r["method"] for r in adv_orderable]

        all_order = [r["method"] for r in ranks["all"]]
        # H-rev required subset orderings
        required = ["popularity", "recency", "markov_order1", "gru_probe"]
        all_req = [m for m in all_order if m in required]
        adv_req = [m for m in ranks["advance_ordering_excluding_recency"] if m in required]
        # recency excluded from advance required ordering
        adv_req = [m for m in adv_req if m != "recency"]
        all_req_adv_comparable = [m for m in all_req if m != "recency"]

        kendall = {}
        for sl in ("revisit", "advance", "nonrepeat"):
            if sl == "advance":
                order = ranks["advance_ordering_excluding_recency"]
                base = [m for m in all_order if m != "recency"]
            else:
                order = [r["method"] for r in ranks.get(sl) or [] if r.get("r5") is not None]
                base = all_order
            kendall[sl] = {
                "kendall_tau_vs_all": kendall_tau(base, order),
                "all_order": base,
                "slice_order": order,
            }

        # H-rev flag: required-model ordering change on advance vs all (excl recency)
        h_rev_advance = {
            "all_required_excl_recency": all_req_adv_comparable,
            "advance_required_excl_recency": adv_req,
            "ordering_changed": all_req_adv_comparable != adv_req,
            "kendall_tau": kendall_tau(all_req_adv_comparable, adv_req),
        }
        pack["kendall_tau_by_slice"] = kendall
        pack["h_rev_diagnostics"] = {
            "advance": h_rev_advance,
            "flip_probability_note": (
                "Per-sequence bootstrap flip probabilities require stored per-seq R@5; "
                "not available from the original paper_rank_reversal_full aggregates. "
                "Confirmatory replication will store per-seq scores and report P(order flip)."
            ),
            "flip_probability_by_pair": None,
            "reversals_flagged_under_h_rev_rule": (
                [{"slice": "advance", **h_rev_advance}]
                if h_rev_advance["ordering_changed"]
                else []
            ),
        }
        # rewrite legacy rank_reversals note
        pack["rank_reversals_legacy_point_estimate"] = pack.get("rank_reversals")
        pack["rank_reversals"] = pack["h_rev_diagnostics"]["reversals_flagged_under_h_rev_rule"]
        pack["status"] = "EXPLORATORY"

    write_json("paper_rank_reversal_full.json", rev)
    return status


# ---------------------------------------------------------------------------
# 3. Paired CIs + verdicts (full-test ranking advance; Phase3; GRU-Markov)
# ---------------------------------------------------------------------------

def compute_advance_paired_from_ranking() -> dict:
    """Re-score full test to get advance-slice paired CIs (sequence bootstrap)."""
    print("recomputing advance-slice paired CIs on full test…", flush=True)
    names = load_names()
    graph, similar_set = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    bundle = joblib.load(OUT / "serve_models.joblib")
    parent_lists = [list(graph.predecessors(i)) for i in range(len(names))]
    popularity = np.asarray(bundle["popularity"])
    feat_fn = build_heads(embeddings, graph, similar_set, popularity, parent_lists, bundle["logreg"])
    mode_clf, review_clf, advance_clf = bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"]
    test_kt = load_sequences(DATA / "test.json")
    train_kt = load_sequences(DATA / "train.json")
    n_items = len(names)
    trans = np.ones((n_items, n_items), dtype=np.float64) * 1e-3
    for seq in train_kt:
        ids = [c for c, _ in seq]
        for a, b in zip(ids, ids[1:]):
            if 0 <= a < n_items and 0 <= b < n_items:
                trans[a, b] += 1.0
    trans = np.log(trans / trans.sum(axis=1, keepdims=True))

    buckets = {
        "ragr": [],
        "recent5": [],
        "ragr_no_dag": [],
        "markov": [],
        "ragr_adv": [],
        "recent5_adv": [],
        "ragr_no_dag_adv": [],
        "markov_adv": [],
    }
    rng = np.random.default_rng(SEED)
    for j, seq in enumerate(test_kt):
        if j % 1000 == 0:
            print(f"  paired pass {j}/{len(test_kt)}", flush=True)
        cuts = np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int)
        local = {k: [] for k in buckets}
        for cut in cuts:
            cut = int(cut)
            pack = score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings)
            pack_nd = score_features(
                feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings,
                mask_cols={"advance": [1, 3]},
            )
            label = pack["label"]
            last = pack["last"]
            stats, last_step, recent = history_state(seq, cut)
            act = action_label(label, last_step, last)
            distinct = []
            for x in reversed(recent):
                if x not in distinct:
                    distinct.append(x)
                if len(distinct) >= 5:
                    break
            r5_scores = np.full(n_items, -1e9)
            for rank, item in enumerate(distinct):
                if 0 <= item < n_items:
                    r5_scores[item] = 5 - rank
            mk = trans[last].copy() if 0 <= last < n_items else popularity.copy()
            local["ragr"].append(recall_at(pack["mixed"], label, K))
            local["ragr_no_dag"].append(recall_at(pack_nd["mixed"], label, K))
            local["recent5"].append(recall_at(r5_scores, label, K))
            local["markov"].append(recall_at(mk, label, K))
            if act == "advance":
                local["ragr_adv"].append(recall_at(pack["mixed"], label, K))
                local["ragr_no_dag_adv"].append(recall_at(pack_nd["mixed"], label, K))
                local["recent5_adv"].append(recall_at(r5_scores, label, K))
                local["markov_adv"].append(recall_at(mk, label, K))
        for k in buckets:
            if local[k]:
                buckets[k].append(local[k])

    paired = {
        "ragr_minus_recent5_r5_all": paired_seq_delta(buckets["ragr"], buckets["recent5"], rng),
        "ragr_minus_ragr_no_dag_r5_all": paired_seq_delta(buckets["ragr"], buckets["ragr_no_dag"], rng),
        "ragr_minus_markov_r5_all": paired_seq_delta(buckets["ragr"], buckets["markov"], rng),
        "ragr_minus_recent5_r5_advance": paired_seq_delta(buckets["ragr_adv"], buckets["recent5_adv"], rng),
        "ragr_minus_ragr_no_dag_r5_advance": paired_seq_delta(
            buckets["ragr_adv"], buckets["ragr_no_dag_adv"], rng
        ),
        "ragr_minus_markov_r5_advance": paired_seq_delta(buckets["ragr_adv"], buckets["markov_adv"], rng),
    }
    # recent5 on advance is ~0 by construction — still report but flag
    paired["ragr_minus_recent5_r5_advance"]["note"] = (
        "Recent-5 on advance is near-zero by construction (label not in history); "
        "prefer RAGR−Markov and RAGR−RAGR-no-DAG for advance validity claims."
    )
    return {
        "eval": {
            "n_sequences": len(test_kt),
            "cuts": EVAL_POS,
            "n_boot": N_BOOT,
            "seed": int(SEED),
            "full_test": True,
            "status": "EXPLORATORY_pre_prereg",
        },
        "thresholds": {"r5": THRESH_R5, "auc": THRESH_AUC},
        "paired": paired,
        "verdicts": {
            name: {
                "delta_mean": p.get("mean"),
                "ci": p.get("ci"),
                "call": verdict_r5(p.get("mean"), p.get("ci")),
                "A": name.split("_minus_")[0],
                "B": name.split("_minus_")[1].replace("_r5_all", "").replace("_r5_advance", ""),
            }
            for name, p in paired.items()
        },
    }


def assemble_verdicts(advance_pack: dict) -> dict:
    full = load("paper_ranking_full.json")
    p3 = load("phase3_memory_fulltest.json")
    gru_mk = load("gru_markov_paired_ci.json") if (PHASE / "gru_markov_paired_ci.json").exists() else {}

    # merge overall paired from existing full ranking if present
    existing = full.get("paired_deltas") or {}
    phase3 = p3["evaluate_on_test_json"]["paired_logistic_step_minus_success"]
    phase3_block = {
        "mean": phase3["mean_delta_auc"],
        "ci": phase3["ci"],
        "n_test_rows": p3["evaluate_on_test_json"]["n_test"],
        "bootstrap": phase3.get("bootstrap"),
        "call": verdict_auc(phase3["mean_delta_auc"], phase3["ci"]),
        "threshold_auc": THRESH_AUC,
        "A": "logistic_step_dt",
        "B": "success_rate",
        "note": (
            f"|Δ|={abs(phase3['mean_delta_auc']):.4f}; "
            f"threshold={THRESH_AUC}; CI excludes 0 but |Δ|<threshold ⇒ tie under verdict rule"
            if abs(phase3["mean_delta_auc"]) < THRESH_AUC
            else "CI and magnitude vs threshold applied"
        ),
    }

    gru_verdicts = []
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
            if not isinstance(nr, dict):
                continue
            if "mean_delta_gru_minus_markov" not in nr:
                continue
            mean = nr["mean_delta_gru_minus_markov"]
            ci = nr.get("ci")
            call = verdict_r5(mean, ci)
            if call == "A":
                call_label = "GRU"
            elif call == "B":
                call_label = "Markov"
            else:
                call_label = "tie"
            gru_verdicts.append({
                "dataset": ds,
                "unit": unit,
                "slice": "nonrepeat" if isinstance(block.get("nonrepeat"), dict) else "all",
                "mean_delta_gru_minus_markov": mean,
                "ci": ci,
                "call": call_label,
                "threshold_r5": THRESH_R5,
                "n_sequences": nr.get("n_sequences"),
            })

    out = {
        "status": "EXPLORATORY_pre_prereg",
        "prereg_v1_tagged": False,
        "thresholds": {"recall_at_5": THRESH_R5, "auc": THRESH_AUC},
        "verdict_rule": (
            f"For R@5: call A if mean(Δ)>0, CI excludes 0, and |Δ|≥{THRESH_R5}; "
            f"call B symmetrically; else tie. "
            f"For AUC: same with threshold {THRESH_AUC}."
        ),
        "junyi_ranking_full_test": {
            "from_paper_ranking_full_paired_deltas": {
                k: {
                    **v,
                    "call": verdict_r5(v.get("mean"), v.get("ci")),
                }
                for k, v in existing.items()
            },
            "advance_slice_recomputed": advance_pack,
        },
        "phase3_logistic_minus_success": phase3_block,
        "gru_minus_markov": {
            "source": "gru_markov_paired_ci.json",
            "rows": gru_verdicts,
            "note": "Validation paired CIs from freeze-era retrain; confirmatory test must use registered seed.",
        },
    }
    write_json("paper_paired_verdicts.json", out)

    # also attach advance paired into paper_ranking_full
    full["paired_deltas_advance"] = advance_pack.get("paired")
    full["verdicts"] = advance_pack.get("verdicts")
    full["status"] = full.get("status") or "EXPLORATORY_pre_prereg_full_test"
    write_json("paper_ranking_full.json", full)
    return out


# ---------------------------------------------------------------------------
# 5. Calibration: dBrier for p_recall; lowest-bin note
# ---------------------------------------------------------------------------

def update_calibration() -> dict:
    cal = load("paper_calibration.json")
    mem = cal.get("p_recall_on_continue_queries") or {}
    if mem.get("n") and mem.get("brier") is not None and mem.get("base_rate") is not None:
        # recompute ΔBrier vs constant base-rate using stored n/base_rate/brier
        # need y,p — reconstruct base-rate brier = base*(1-base) for binary
        br = float(mem["base_rate"])
        brier_base = br * (1.0 - br)
        mem["brier_constant_base_rate"] = brier_base
        mem["base_rate_adjusted_brier"] = float(mem["brier"]) - brier_base
        mem["base_rate_adjusted_brier_note"] = (
            "Brier(model) − base_rate*(1−base_rate); negative ⇒ better than constant base rate"
        )
    rel = mem.get("reliability") or []
    nonempty = [r for r in rel if r.get("n")]
    if nonempty:
        # lowest predicted bin among nonempty
        lowest = min(nonempty, key=lambda r: r["pred"] if r["pred"] is not None else 1.0)
        mem["lowest_pred_bin"] = {
            **lowest,
            "gap_obs_minus_pred": (
                None
                if lowest.get("obs") is None or lowest.get("pred") is None
                else float(lowest["obs"]) - float(lowest["pred"])
            ),
        }
    cal["p_recall_on_continue_queries"] = mem
    write_json("paper_calibration.json", cal)
    return mem


# ---------------------------------------------------------------------------
# Fix followup3 concept table method B
# ---------------------------------------------------------------------------

def fix_concept_xes_method_b() -> None:
    concept = load("followup3_concept_repetition.json")
    xes = load("paper_xes_kc_methods.json")
    b = xes["method_b_expand_all_kc_tags"]
    new_rows = []
    for r in concept["rows"]:
        unit = r["dataset_unit"]
        if "method B" in unit and "drop multi-KC" in unit:
            new_rows.append({
                "dataset_unit": "XES3G5M KC (method B: expand-all KC tags)",
                "share_next_equals_last": b["share_next_equals_last"],
                "three_way_mix": b["three_way_mix"],
                "supersedes": "old method B was first-KC only (=A); replaced by expand-all",
            })
        elif "method B parquet" in unit:
            # keep as sensitivity footnote row but mark
            r = dict(r)
            r["note"] = "parquet is_repeat sensitivity; not primary method B"
            new_rows.append(r)
        else:
            new_rows.append(r)
    concept["rows"] = new_rows
    concept["xes_method_b_source"] = "paper_xes_kc_methods.json method_b_expand_all_kc_tags"
    write_json("followup3_concept_repetition.json", concept)


def main():
    t0 = time.time()
    print("=== 1 exploratory mark ===", flush=True)
    status = mark_exploratory_rank_reversal()
    print(json.dumps(status["evidence"], indent=2))

    print("=== fix XES concept B ===", flush=True)
    fix_concept_xes_method_b()

    print("=== calibration dBrier ===", flush=True)
    mem = update_calibration()
    print("p_recall ΔBrier", mem.get("base_rate_adjusted_brier"), "lowest bin", mem.get("lowest_pred_bin"))

    print("=== advance paired CIs (slow) ===", flush=True)
    adv = compute_advance_paired_from_ranking()
    write_json("paper_advance_paired_tmp.json", adv)  # survive assemble failures
    assemble_verdicts(adv)

    print("done checkpoint", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
