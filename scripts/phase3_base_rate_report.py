#!/usr/bin/env python3
"""Phase 3 base-rate / confusion diagnostics + integer-labelled confusion plot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from junyi_pipeline import DATA, load_sequences
from run_valid_eval import assert_no_leak, memory_rows_by_sequence, split_by_id

PHASE = ROOT / "outputs_junyi" / "phases"
FIG = ROOT / "outputs_junyi" / "figures"
TRAIN_MEM_CAP = 200_000
TEST_MEM_CAP = 80_000  # exact rows used for 0.877 AUC / confusion in validated protocol


def attempt_stats(seqs, max_seq=None):
    use = seqs if max_seq is None else seqs[:max_seq]
    ys = [c for s in use for _, c in s]
    return {"n_attempts": len(ys), "correct_rate": float(np.mean(ys)) if ys else None, "n_sequences": len(use)}


def main() -> None:
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")

    feats, labels, sids = [], [], []
    for feat, y, sid in memory_rows_by_sequence(train):
        feats.append(feat)
        labels.append(y)
        sids.append(sid)
        if len(feats) >= TRAIN_MEM_CAP:
            break
    x = np.vstack(feats)
    y = np.array(labels)
    sids = np.array(sids)
    tr_ids, te_ids = split_by_id(sids)
    assert_no_leak(tr_ids, te_ids, "ktbd-sequence")
    tr_m = np.array([s in tr_ids for s in sids])

    feats_te, y_te = [], []
    for feat, yy, _ in memory_rows_by_sequence(test):
        feats_te.append(feat)
        y_te.append(yy)
        if len(feats_te) >= TEST_MEM_CAP:
            break
    x_pub = np.vstack(feats_te)
    y_pub = np.array(y_te)

    # full test memory rows (no cap) for comparison
    y_test_all = np.array([yy for _, yy, _ in memory_rows_by_sequence(test)])

    clf = LogisticRegression(max_iter=400).fit(x[tr_m][:, 1:4], y[tr_m])
    p_log = clf.predict_proba(x_pub[:, 1:4])[:, 1]
    fpr, tpr, _ = roc_curve(y_pub, p_log)
    auc_val = float(auc(fpr, tpr))
    yhat = (p_log >= 0.5).astype(int)
    cm = confusion_matrix(y_pub, yhat)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    report = {
        "sampling_rule": {
            "memory_row_definition": (
                "A memory row is emitted only when a concept is revisited within a sequence "
                "(first attempt on a concept produces no row). Features = "
                "[bias, log1p(attempts), success_rate, log1p(delta_steps), log1p(delta_days)]."
            ),
            "train_fit_rows": (
                f"Iterate train.json in file order via memory_rows_by_sequence; keep first "
                f"{TRAIN_MEM_CAP:,} rows; split by sequence id 80/20 (leakage_check=pass); "
                "fit logistic on the train-side sequences only."
            ),
            "test_auc_and_confusion_rows": (
                f"Iterate test.json in file order via memory_rows_by_sequence; keep first "
                f"{TEST_MEM_CAP:,} rows. These exact rows produce the headline ~0.877 AUC and the confusion matrix."
            ),
            "why_not_attempt_level_0_51": (
                "The ~0.51 figure is attempt-level correct rate on a 4k-sequence train sample "
                "(every attempt counts). The confusion matrix uses revisit-only memory rows from "
                "the first 80k test memory rows — a harder, non-iid slice with lower correct rate."
            ),
        },
        "attempt_level_correct_rate": {
            "train_json_all": attempt_stats(train),
            "train_json_first_4000_sequences": attempt_stats(train, 4000),
            "test_json_all": attempt_stats(test),
        },
        "memory_row_correct_rate": {
            "train_first_200k": {
                "n_rows": int(len(y)),
                "correct_rate": float(y.mean()),
                "fit_rows_n": int(tr_m.sum()),
                "fit_correct_rate": float(y[tr_m].mean()),
                "within_train_holdout_n": int((~tr_m).sum()),
                "within_train_holdout_correct_rate": float(y[~tr_m].mean()),
            },
            "test_first_80k_used_for_auc_and_confusion": {
                "n_rows": int(len(y_pub)),
                "correct_rate": float(y_pub.mean()),
                "cap_hit": len(y_pub) == TEST_MEM_CAP,
            },
            "test_all_memory_rows_uncapped": {
                "n_rows": int(len(y_test_all)),
                "correct_rate": float(y_test_all.mean()),
            },
        },
        "gap_explanation": (
            "Attempt-level train sample ≈0.51 includes first-time attempts. Memory rows are revisits only "
            f"(train first-200k rate ≈{float(y.mean()):.3f}). The matrix/AUC slice is the first "
            f"{TEST_MEM_CAP:,} test memory rows in file order (correct_rate="
            f"{float(y_pub.mean()):.3f}), not a random sample of all test memory rows "
            f"(uncapped test memory correct_rate={float(y_test_all.mean()):.3f}). "
            "File-order capping over-represents harder early sequences → ~0.33 vs ~0.45."
        ),
        "logistic_step_dt_on_capped_test": {
            "auc": auc_val,
            "n_rows": int(len(y_pub)),
            "threshold": 0.5,
            "confusion_matrix_counts": {
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "note": "Printed as integers (not scientific notation).",
            },
        },
    }

    PHASE.mkdir(parents=True, exist_ok=True)
    path = PHASE / "phase3_base_rate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", path)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    disp = ConfusionMatrixDisplay(cm, display_labels=["wrong (0)", "correct (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Phase 3 logistic @0.5 — test.json first 80k memory rows")
    # force integer annotations if library still formats oddly
    for t in ax.texts:
        try:
            t.set_text(f"{int(float(t.get_text())):,}")
        except ValueError:
            pass
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "memory_confusion.png", dpi=140)
    plt.close(fig)
    print("wrote", FIG / "memory_confusion.png", "TN,FP,FN,TP=", tn, fp, fn, tp)


if __name__ == "__main__":
    main()
