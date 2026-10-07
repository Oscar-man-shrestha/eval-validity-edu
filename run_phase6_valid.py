"""Finish Phase 6 only: sequence-bootstrap CIs + baselines (faster settings)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from junyi_pipeline import DATA, OUT, SEED, load_graph, load_names, load_sequences
from run_all_phases import train_models, write_json
from run_valid_eval import (
    assert_no_leak,
    evaluate_ranking,
    memory_rows_by_sequence,
    split_by_id,
)

# Faster than full run_valid_eval defaults
import run_valid_eval as R

R.EVAL_SEQS = 400
R.EVAL_POS = 4
R.BOOT_SEQ = 300


def main() -> None:
    print("loading", flush=True)
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, similar = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    train_kt = load_sequences(DATA / "train.json")
    test_kt = load_sequences(DATA / "test.json")
    popularity = np.zeros(len(names))
    for seq in train_kt:
        for c, _ in seq:
            popularity[c] += 1
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]

    print("fitting memory + ranker", flush=True)
    feats, labels, sids = [], [], []
    for feat, y, sid in memory_rows_by_sequence(train_kt):
        feats.append(feat)
        labels.append(y)
        sids.append(sid)
        if len(feats) >= 200_000:
            break
    x = np.vstack(feats)
    y = np.array(labels)
    sids = np.array(sids)
    tr_ids, te_ids = split_by_id(sids)
    assert_no_leak(tr_ids, te_ids, "ktbd-sequence")
    tr_m = np.array([s in tr_ids for s in sids])
    logreg = LogisticRegression(max_iter=500).fit(x[tr_m][:, [1, 2, 3]], y[tr_m])

    mode_clf, review_clf, advance_clf, feat_fn, p5 = train_models(
        train_kt, embeddings, graph, similar, popularity, parent_lists, logreg, rng
    )
    write_json("phase5_ranker.json", p5)
    print("ranking eval…", flush=True)
    ranking = evaluate_ranking(
        test_kt,
        embeddings,
        graph,
        similar,
        popularity,
        (mode_clf, review_clf, advance_clf, feat_fn),
        parent_lists,
        rng,
    )
    phase6 = {
        "phase": 6,
        "bootstrap": "over_sequences",
        "eval_sequences": R.EVAL_SEQS,
        "cuts_per_sequence": R.EVAL_POS,
        "ranking": ranking,
    }
    write_json("phase6_eval.json", phase6)

    # Refresh headline from all phase JSON
    p2 = json.loads((OUT / "phases" / "phase2_embeddings.json").read_text())
    p3 = json.loads((OUT / "phases" / "phase3_memory.json").read_text())
    p4 = json.loads((OUT / "phases" / "phase4_graph.json").read_text())
    te = p3["evaluate_on_test_json"]
    td = p3.get("timed_learner_split") or {}
    lines = [
        "NeuroTrace-DAG — VALIDATED results (sequence/learner splits, sequence bootstrap)",
        f"Phase 2  pair-AUC vs random={p2['pair_auc_vs_random']:.3f}  vs same-topic={p2['pair_auc_vs_same_topic']:.3f}  lexical-AUC={p2['lexical_jaccard_auc_vs_random']:.3f}",
        f"Phase 3  test.json logistic_step={te['logistic_step_dt']['auc']:.3f} CI={te['logistic_step_dt']['auc_ci']}  success={te['success_rate']['auc']:.3f}  hlr={te['hlr_step']['auc']:.3f}",
    ]
    if td:
        lines.append(
            f"Phase 3 timed(learner)  day_dt={td['logistic_day_dt']['auc']:.3f}  "
            f"success={td['success_rate']['auc']:.3f}  "
            f"delta_day_minus_success={td['paired_logistic_day_minus_success']['mean_delta_auc']:.4f} "
            f"CI={td['paired_logistic_day_minus_success']['ci']}  "
            f"forgetting_helps={td['paired_logistic_day_minus_success']['forgetting_helps']}"
        )
    mix = p4["next_item_mix_train_sample"]
    lines.append(
        f"Phase 4  review={mix['review_frac']:.3f}  violate={mix['advance_violation_frac']:.3f}  "
        f"DAG={p4['nx.is_directed_acyclic_graph']}"
    )
    r = ranking
    lines.append(
        f"Phase 6  RAGR R@5={r['ragr_all']['mean']:.3f} CI={r['ragr_all']['ci']}  "
        f"n_seq={r['ragr_all']['n_sequences']} n_q={r['ragr_all']['n_queries']}"
    )
    lines.append(
        f"         force_review R@5={r['force_review_all']['mean']:.3f} CI={r['force_review_all']['ci']}  "
        f"delta={r.get('ragr_minus_force_review')}"
    )
    if "ragr_advance" in r:
        lines.append(f"         advance R@5={r['ragr_advance']['mean']:.3f} CI={r['ragr_advance']['ci']}")
    lines.append(f"         share next==last={r['share_next_equals_last']:.3f}")
    if "gate" in r:
        lines.append(
            f"Gate  AUC={r['gate']['auc']:.3f}  PR-AUC={r['gate']['pr_auc']:.3f}  base={r['gate']['base_rate']:.3f}"
        )
    if "advance_ablation" in r:
        for k, v in r["advance_ablation"].items():
            lines.append(f"  advance_ablation {k}: R@5={v['mean']:.3f} CI={v['ci']}")
    for key in [
        "base_last_review",
        "base_prefix_pop_review",
        "base_global_pop_advance",
        "base_dag_child_advance",
        "base_sim_advance",
    ]:
        if key in r:
            lines.append(f"  baseline {key}: R@5={r[key]['mean']:.3f} CI={r[key]['ci']}")
    (OUT / "all_phases_results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
