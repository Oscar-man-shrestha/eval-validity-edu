#!/usr/bin/env python3
"""Recompute Phase 3 test metrics on ALL revisit memory rows.

Headline 0.877 / CM used the first 80,000 test.json memory rows in file order
with a row-level bootstrap. This script:

- fits the same train protocol (first 200k train.json memory rows, sequence split)
- scores EVERY test.json memory row (revisit attempts only; first attempt on a
  concept emits no row)
- reports sequence-level bootstrap CIs (resample sequences, take all their rows)

Does not write a preregistration.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junyi_pipeline import DATA, SEED, fit_hlr, hlr_predict, load_sequences
from run_all_phases import PHASE_OUT, write_json
from run_valid_eval import assert_no_leak, memory_rows_by_sequence, split_by_id

N_BOOT = 400
TRAIN_ROW_CAP = 200_000
OLD_TEST_CAP = 80_000


def _groups(sids: np.ndarray) -> dict[int, np.ndarray]:
    g: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(sids):
        g[int(s)].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}


def seq_boot(y, p, sids, metric, n_boot=N_BOOT, seed=SEED):
    groups = _groups(sids)
    keys = np.array(list(groups.keys()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[int(s)] for s in draw])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(float(metric(yy, pp)))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "boot_mean": float(arr.mean()),
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_boot_kept": int(len(arr)),
        "n_boot_requested": n_boot,
        "unit": "sequence",
        "statistic": "mean_of_resampled_full_sequences",
    }


def pack_model(name, y, p, sids, extra=None):
    out = {
        "auc_point": float(roc_auc_score(y, p)),
        "pr_auc_point": float(average_precision_score(y, p)),
        "auc_sequence_bootstrap": seq_boot(y, p, sids, roc_auc_score, seed=SEED + 11),
        "pr_auc_sequence_bootstrap": seq_boot(y, p, sids, average_precision_score, seed=SEED + 13),
    }
    # Headline fields used by the report (point estimate + sequence CI)
    out["auc"] = out["auc_point"]
    out["auc_ci"] = out["auc_sequence_bootstrap"]["ci"]
    out["pr_auc"] = out["pr_auc_point"]
    out["pr_auc_ci"] = out["pr_auc_sequence_bootstrap"]["ci"]
    if extra:
        out.update(extra)
    return out


def cm_block(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "n": int(len(y)),
        "predicted_positive_rate": float(pred.mean()),
        "accuracy": float((pred == y).mean()),
    }


def main() -> None:
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train_kt = load_sequences(DATA / "train.json")
    test_kt = load_sequences(DATA / "test.json")

    feats, labels, sids = [], [], []
    for feat, y, sid in memory_rows_by_sequence(train_kt):
        feats.append(feat)
        labels.append(y)
        sids.append(sid)
        if len(feats) >= TRAIN_ROW_CAP:
            break
    x = np.vstack(feats)
    y = np.array(labels, dtype=np.float64)
    sids = np.array(sids)
    tr_ids, te_ids = split_by_id(sids)
    assert_no_leak(tr_ids, te_ids, "ktbd-sequence")
    tr_m = np.array([s in tr_ids for s in sids])
    x_tr, y_tr = x[tr_m], y[tr_m]

    feats_te, y_te, sid_te = [], [], []
    for feat, yy, sid in memory_rows_by_sequence(test_kt):
        feats_te.append(feat)
        y_te.append(yy)
        sid_te.append(sid)
    x_te = np.vstack(feats_te)
    y_te = np.array(y_te, dtype=np.float64)
    sid_te = np.array(sid_te, dtype=np.int64)

    n_test_seq = int(len(np.unique(sid_te)))
    print(
        f"train_fit_rows={len(y_tr)} test_memory_rows={len(y_te)} "
        f"test_sequences_with_revisits={n_test_seq} correct_rate={y_te.mean():.6f}",
        flush=True,
    )

    # Same five models as eval_memory_table
    models = {}
    mean_p = np.full_like(y_te, y_tr.mean())
    models["mean"] = pack_model("mean", y_te, mean_p, sid_te)

    logreg_step = LogisticRegression(max_iter=500).fit(x_tr[:, [1, 2, 3]], y_tr)
    p_step = logreg_step.predict_proba(x_te[:, [1, 2, 3]])[:, 1]
    models["logistic_step_dt"] = pack_model("logistic_step_dt", y_te, p_step, sid_te)

    logreg_day = LogisticRegression(max_iter=500).fit(x_tr[:, [1, 2, 4]], y_tr)
    p_day = logreg_day.predict_proba(x_te[:, [1, 2, 4]])[:, 1]
    models["logistic_day_dt"] = pack_model("logistic_day_dt", y_te, p_day, sid_te)

    logreg_both = LogisticRegression(max_iter=500).fit(x_tr[:, 1:], y_tr)
    p_both = logreg_both.predict_proba(x_te[:, 1:])[:, 1]
    models["logistic_both"] = pack_model("logistic_both", y_te, p_both, sid_te)

    p_succ = x_te[:, 2]
    models["success_rate"] = pack_model("success_rate", y_te, p_succ, sid_te)

    hlr_w = fit_hlr(x_tr[:, :4], y_tr)
    p_hlr = hlr_predict(hlr_w, x_te[:, :4])
    models["hlr_step"] = pack_model("hlr_step", y_te, p_hlr, sid_te)

    # Paired sequence bootstrap: logistic_step − success_rate
    groups = _groups(sid_te)
    keys = np.array(list(groups.keys()), dtype=np.int64)
    deltas = []
    for _ in range(N_BOOT):
        draw = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[int(s)] for s in draw])
        yy = y_te[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(roc_auc_score(yy, p_step[idx]) - roc_auc_score(yy, p_succ[idx]))
    d = np.asarray(deltas, dtype=np.float64)

    winner = max(
        ("logistic_step_dt", "logistic_day_dt", "logistic_both", "success_rate", "hlr_step"),
        key=lambda k: models[k]["auc_point"],
    )
    p_win = {
        "logistic_step_dt": p_step,
        "logistic_day_dt": p_day,
        "logistic_both": p_both,
        "success_rate": p_succ,
        "hlr_step": p_hlr,
    }[winner]

    # 80k file-order slice (for the superseded comparison, same fit)
    x_80, y_80 = x_te[:OLD_TEST_CAP], y_te[:OLD_TEST_CAP]
    p_80 = p_step[:OLD_TEST_CAP]
    superseded_80k = {
        "n_rows": int(len(y_80)),
        "selection": "first_80000_test_json_memory_rows_in_file_order",
        "correct_rate": float(y_80.mean()),
        "logistic_step_dt_auc_point": float(roc_auc_score(y_80, p_80)),
        "logistic_step_dt_pr_auc_point": float(average_precision_score(y_80, p_80)),
        "confusion_matrix_counts": cm_block(y_80, p_80),
        "note": (
            "This is the slice that produced the published headline ~0.877 "
            "(row-level bootstrap mean in phase3_memory.json). Superseded."
        ),
    }

    payload = {
        "phase": 3,
        "task": "correctness_on_revisit_attempts_only",
        "task_definition": (
            "A memory row is emitted only when a concept is attempted again in the same "
            "sequence. The label is whether that revisit attempt is correct (0/1). "
            "The first attempt on a concept does not create a row."
        ),
        "headline_0_877_used_80k_file_order_slice": True,
        "headline_0_877_superseded": True,
        "split": "by_sequence_id_within_train_json",
        "leakage_check": "pass",
        "train_protocol": {
            "source": "train.json",
            "memory_row_cap": TRAIN_ROW_CAP,
            "n_rows_collected": int(len(y)),
            "n_fit_rows": int(len(y_tr)),
            "fit_on": "sequence-id split of capped train memory rows (same as run_valid_eval)",
        },
        "test_protocol": {
            "source": "test.json",
            "rows": "ALL memory (revisit) rows; no 80k cap",
            "n_rows": int(len(y_te)),
            "n_sequences_with_at_least_one_revisit": n_test_seq,
            "n_test_json_sequences": int(len(test_kt)),
            "correct_rate": float(y_te.mean()),
            "bootstrap": "sequence-level: resample sequences with replacement; pool their rows",
            "n_boot": N_BOOT,
            "registered_sample": None,
        },
        "evaluate_on_test_json": {
            **models,
            "paired_logistic_step_minus_success": {
                "mean_delta_auc": float(d.mean()),
                "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                "bootstrap": "sequence",
            },
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "winner": winner,
            "confusion_matrix_winner": cm_block(y_te, p_win),
            "confusion_matrix_logistic_step_dt": cm_block(y_te, p_step),
        },
        "superseded_first_80k_file_order": superseded_80k,
    }
    write_json("phase3_memory_fulltest.json", payload)

    # Patch phase3_memory.json: keep old block labeled, replace headline block
    old_path = PHASE_OUT / "phase3_memory.json"
    old = json.loads(old_path.read_text())
    old_eval = old.get("evaluate_on_test_json")
    old["evaluate_on_test_json_first_80k_file_order_superseded"] = old_eval
    old["evaluate_on_test_json"] = payload["evaluate_on_test_json"]
    old["test_eval_protocol"] = payload["test_protocol"]
    old["task"] = payload["task"]
    old["task_definition"] = payload["task_definition"]
    old["headline_0_877_used_80k_file_order_slice"] = True
    old["headline_0_877_superseded"] = True
    old["fulltest_path"] = "outputs_junyi/phases/phase3_memory_fulltest.json"
    write_json("phase3_memory.json", old)

    # Patch base-rate report
    br_path = PHASE_OUT / "phase3_base_rate.json"
    if br_path.exists():
        br = json.loads(br_path.read_text())
    else:
        br = {}
    br["headline_0_877_used_80k_file_order_slice"] = True
    br["headline_replaced_by"] = "phase3_memory_fulltest.json"
    br["task"] = payload["task"]
    br["logistic_step_dt_on_all_test_memory_rows"] = {
        "n_rows": int(len(y_te)),
        "correct_rate": float(y_te.mean()),
        "auc_point": models["logistic_step_dt"]["auc_point"],
        "auc_ci_sequence_bootstrap": models["logistic_step_dt"]["auc_ci"],
        "pr_auc_point": models["logistic_step_dt"]["pr_auc_point"],
        "confusion_matrix_counts": cm_block(y_te, p_step),
    }
    br_path.write_text(json.dumps(br, indent=2) + "\n")

    win = models[winner]
    print(
        f"WINNER {winner} auc={win['auc_point']:.6f} "
        f"ci={win['auc_ci']} pr_auc={win['pr_auc_point']:.6f} "
        f"n={len(y_te)}",
        flush=True,
    )
    print("wrote phase3_memory_fulltest.json and patched phase3_memory.json", flush=True)


if __name__ == "__main__":
    main()
