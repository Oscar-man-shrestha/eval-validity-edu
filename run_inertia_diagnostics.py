"""Inertia-Illusion diagnostics recommended by the novelty audit.

Adds three analyses on top of phase6_eval.json:
  1) Failure-conditioned next action (after wrong vs after correct)
  2) Pedagogical cost of hard-gate violations (accuracy unlocked vs skipped)
  3) Gate threshold regret curve (overall R@5 vs advance R@5)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import build_heads, mode_features, write_json
from run_valid_eval import EVAL_POS, EVAL_SEQS, recall_at, score_features

BUNDLE = OUT / "serve_models.joblib"


def main() -> None:
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, similar = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    train_kt = load_sequences(DATA / "train.json")
    test_kt = load_sequences(DATA / "test.json")
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]
    bundle = joblib.load(BUNDLE)
    popularity = np.asarray(bundle["popularity"])
    feat_fn = build_heads(embeddings, graph, similar, popularity, parent_lists, bundle["logreg"])
    mode_clf, review_clf, advance_clf = bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"]

    # --- 1) Failure-conditioned transitions on train prefixes ---
    after = {
        "wrong": {"n": 0, "review": 0, "last": 0, "advance": 0, "violate": 0},
        "correct": {"n": 0, "review": 0, "last": 0, "advance": 0, "violate": 0},
    }
    for seq in train_kt[:8000]:
        for cut in range(8, len(seq)):
            prev_y = seq[cut - 1][1]
            key = "wrong" if prev_y == 0 else "correct"
            label = seq[cut][0]
            stats, last_step, recent = history_state(seq, cut)
            mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
            bucket = after[key]
            bucket["n"] += 1
            if label == recent[-1]:
                bucket["last"] += 1
            if label in last_step:
                bucket["review"] += 1
            else:
                bucket["advance"] += 1
                parents = parent_lists[label]
                if parents and any(mastery.get(p, 0.0) < 0.5 for p in parents):
                    bucket["violate"] += 1

    failure_conditioned = {}
    for key, b in after.items():
        n = max(b["n"], 1)
        failure_conditioned[key] = {
            "n": b["n"],
            "p_review": b["review"] / n,
            "p_last": b["last"] / n,
            "p_advance": b["advance"] / n,
            "p_violate_among_all": b["violate"] / n,
        }

    # --- 2) Pedagogical cost: next-attempt accuracy when advance is unlocked vs violated ---
    unlock_y, violate_y = [], []
    for seq in train_kt[:8000]:
        for cut in range(8, min(len(seq), 80)):
            label, y = seq[cut]
            stats, last_step, _ = history_state(seq, cut)
            if label in last_step:
                continue
            mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
            parents = parent_lists[label]
            unlocked = (not parents) or all(mastery.get(p, 0.0) >= 0.5 for p in parents)
            if unlocked:
                unlock_y.append(y)
            else:
                violate_y.append(y)
    pedagogical_cost = {
        "advance_unlocked": {
            "n": len(unlock_y),
            "accuracy": float(np.mean(unlock_y)) if unlock_y else None,
        },
        "advance_hard_gate_violation": {
            "n": len(violate_y),
            "accuracy": float(np.mean(violate_y)) if violate_y else None,
        },
    }
    if unlock_y and violate_y:
        pedagogical_cost["accuracy_gap_unlocked_minus_violate"] = float(np.mean(unlock_y) - np.mean(violate_y))

    # --- 3) Gate threshold regret on test sequences ---
    idx = rng.choice(len(test_kt), size=min(EVAL_SEQS, len(test_kt)), replace=False)
    thresholds = np.round(np.linspace(0.2, 0.9, 8), 2)
    curve = []
    # collect packs once
    packs = []
    for si in idx:
        seq = test_kt[int(si)]
        cuts = np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int)
        for cut in cuts:
            packs.append(
                score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, int(cut), graph, embeddings)
            )
    for tau in thresholds:
        hits_all, hits_adv, n_adv = [], [], 0
        for pack in packs:
            # use review head if p_rev >= tau else advance head
            scores = pack["review"] if pack["p_rev"] >= tau else pack["advance"]
            hit = recall_at(scores, pack["label"], 5)
            hits_all.append(hit)
            if not pack["is_review"]:
                n_adv += 1
                hits_adv.append(hit)
        curve.append(
            {
                "tau_review_if_p_ge": float(tau),
                "r5_all": float(np.mean(hits_all)),
                "r5_advance": float(np.mean(hits_adv)) if hits_adv else 0.0,
                "n_advance": n_adv,
                "n_all": len(hits_all),
            }
        )

    out = {
        "phase": "inertia_diagnostics",
        "thesis": "Inertia Illusion: monolithic next-item metrics on Junyi are dominated by session inertia",
        "failure_conditioned_next_action": failure_conditioned,
        "pedagogical_cost_of_violations": pedagogical_cost,
        "gate_threshold_regret_curve": curve,
        "n_test_queries_for_curve": len(packs),
    }
    write_json("phase6_inertia_diagnostics.json", out)

    # append readable lines
    summary = OUT / "all_phases_results.txt"
    lines = summary.read_text().splitlines() if summary.exists() else []
    lines = [ln for ln in lines if not ln.startswith("Inertia ") and not ln.startswith("         after_")]
    fw, fc = failure_conditioned["wrong"], failure_conditioned["correct"]
    lines.append(
        f"Inertia after_wrong  p_last={fw['p_last']:.3f} p_review={fw['p_review']:.3f} p_advance={fw['p_advance']:.3f}"
    )
    lines.append(
        f"Inertia after_correct p_last={fc['p_last']:.3f} p_review={fc['p_review']:.3f} p_advance={fc['p_advance']:.3f}"
    )
    pc = pedagogical_cost
    lines.append(
        f"Inertia DAG pedagogy  unlocked_acc={pc['advance_unlocked']['accuracy']:.3f} "
        f"violate_acc={pc['advance_hard_gate_violation']['accuracy']:.3f} "
        f"gap={pc.get('accuracy_gap_unlocked_minus_violate')}"
    )
    best_adv = max(curve, key=lambda r: r["r5_advance"])
    best_all = max(curve, key=lambda r: r["r5_all"])
    lines.append(
        f"Inertia gate_curve  best_all@τ={best_all['tau_review_if_p_ge']} R@5={best_all['r5_all']:.3f}; "
        f"best_advance@τ={best_adv['tau_review_if_p_ge']} R@5={best_adv['r5_advance']:.3f}"
    )
    summary.write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2)[:2000])
    print("wrote", OUT / "phases" / "phase6_inertia_diagnostics.json")


if __name__ == "__main__":
    main()
