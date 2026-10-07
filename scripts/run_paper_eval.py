#!/usr/bin/env python3
"""Paper evaluation-validity runs → outputs_junyi/phases/paper_*.json

Does NOT mutate numbers to look better. Old vs new logged in docs/DEVIATIONS.md via
scripts/build_paper_assets.py.

Produces:
  paper_junyi_counts.json
  paper_xes_kc_methods.json          (independent method B = expand-all KC tags)
  paper_ranking_capped.json          (copy/pointer to review_ranking + note)
  paper_ranking_full.json            (full test set, sequence bootstrap)
  paper_calibration.json             (Brier/ECE for p_recall + gate; reliability)
  paper_rank_reversal_full.json      (Markov + pop/recency/gru/attn/ragr/ragr_no_dag)
  paper_sensitivity.json             (max_len, cuts, timed-6k selection)
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss

torch.set_num_threads(1)

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import build_heads
from run_claude_fixes import (
    EVAL_POS,
    N_BOOT,
    action_label,
    paired_seq_delta,
    seq_bootstrap_mean,
)
from run_followup3 import (
    AttnProbe,
    GRUProbe,
    XES_Q,
    _parse_concepts,
    _parse_int_list,
    action_slice,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    load_topic_map,
    popularity_scores,
    score_seq_model,
    share_and_mix,
    train_seq_model,
)
from run_valid_eval import EVAL_SEQS, recall_at, score_features

PHASE = OUT / "phases"
PHASE.mkdir(parents=True, exist_ok=True)
BUNDLE = OUT / "serve_models.joblib"
K = 5


def write_json(name: str, obj: dict) -> None:
    path = PHASE / name
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print("wrote", path, flush=True)


# ---------------------------------------------------------------------------
# 1–2. Junyi counts audit
# ---------------------------------------------------------------------------

def audit_junyi_counts() -> dict:
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")
    topic_of = load_topic_map()
    train_inter = sum(len(s) for s in train)
    test_inter = sum(len(s) for s in test)
    train_trans = sum(max(0, len(s) - 1) for s in train)
    test_trans = sum(max(0, len(s) - 1) for s in test)
    topic_seqs = [[topic_of.get(c, "unknown") for c, _ in s] for s in train + test]
    topic_share = share_and_mix(topic_seqs)
    p1 = json.loads((PHASE / "phase1_prepare.json").read_text())
    total_inter = train_inter + test_inter
    total_trans = train_trans + test_trans
    out = {
        "current_load_sequences": {
            "train_n_sequences": len(train),
            "test_n_sequences": len(test),
            "train_n_interactions": train_inter,
            "test_n_interactions": test_inter,
            "train_plus_test_n_interactions": total_inter,
            "train_n_transitions": train_trans,
            "test_n_transitions": test_trans,
            "train_plus_test_n_transitions": total_trans,
        },
        "split_provenance": {
            "train_interactions_split": "train.json after len∈[12,200]",
            "test_interactions_split": "test.json after len∈[12,200]",
            "topic_level_n_split": "train.json + test.json consecutive transitions",
        },
        "phase1_prepare_prior_fields": {
            "interactions_field_before_update": p1.get("interactions"),
            "train_sequences": p1.get("train_sequences"),
            "test_sequences": p1.get("test_sequences"),
            "note": (
                "If interactions previously equalled train+test attempt rows (2,842,883), "
                "that is total interactions — not a contradiction with topic n."
            ),
        },
        "topic_level_share": {
            "split": "train.json + test.json (length-filtered sequences)",
            "n_transitions": topic_share["share_next_equals_last"]["n"],
            "share": topic_share["share_next_equals_last"]["share"],
            "identity_check": {
                "train_plus_test_transitions": total_trans,
                "interactions_minus_sequences": total_inter - len(train) - len(test),
                "equal": total_trans == total_inter - len(train) - len(test),
                "formula": "2,842,883 − 27,434 − 6,290 = 2,809,159 when interactions=train+test",
            },
            "claude_mismatch_resolved": {
                "total_interactions": total_inter,
                "total_interactions_minus_train_seq": total_inter - len(train),
                "equals_topic_n_plus_test_seq": (total_inter - len(train)) == total_trans + len(test),
                "explanation": (
                    "2,842,883 − 27,434 = 2,815,449 leaves the test sequence count (6,290) "
                    "still unsubtracted; subtracting it yields topic n = 2,809,159. "
                    "Train-only interactions are 2,342,784, not 2,815,449."
                ),
            },
        },
    }
    out["phase1_update"] = {
        "old_interactions": p1.get("interactions"),
        "new_interactions_train_only": train_inter,
        "new_interactions_train_plus_test": total_inter,
        "new_interactions_test": test_inter,
    }
    p1["interactions"] = train_inter
    p1["interactions_train"] = train_inter
    p1["interactions_test"] = test_inter
    p1["interactions_train_plus_test"] = total_inter
    p1["transitions_train"] = train_trans
    p1["transitions_test"] = test_trans
    p1["transitions_train_plus_test"] = total_trans
    p1["interactions_note"] = (
        "interactions = train attempt rows after len∈[12,200]; "
        "interactions_train_plus_test = 2,842,883; "
        "topic-level n in Follow-up 3 = transitions_train_plus_test = 2,809,159"
    )
    write_json("phase1_prepare.json", p1)
    write_json("paper_junyi_counts.json", out)
    return out


# ---------------------------------------------------------------------------
# 5. Independent XES method B
# ---------------------------------------------------------------------------

def run_xes_kc_independent() -> dict:
    """Method A = question→primary KC; Method B = expand ALL KC tags per question (F2-style)."""
    frames = [pd.read_csv(XES_Q / n) for n in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv")]
    df = pd.concat(frames, ignore_index=True)
    best = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        cons = _parse_concepts(row["concepts"])
        if len(qs) < 12:
            continue
        if uid not in best or len(qs) > best[uid][0]:
            best[uid] = (len(qs), qs, cons)

    method_a, method_b = [], []
    for _, qs, cons in best.values():
        primary, expanded = [], []
        for i, _q in enumerate(qs):
            kcs = cons[i] if i < len(cons) else []
            if not kcs:
                continue
            primary.append(int(kcs[0]))
            expanded.extend(int(k) for k in kcs)  # independent: keep every KC tag
        if len(primary) >= 12:
            method_a.append(primary)
        if len(expanded) >= 12:
            method_b.append(expanded[:400])

    a = share_and_mix(method_a)
    b = share_and_mix(method_b)
    out = {
        "method_a_question_then_primary_kc": {
            "method": "map each question to its first KC, then share(next==last) on KC ids",
            "n_sequences": len(method_a),
            **a,
        },
        "method_b_expand_all_kc_tags": {
            "method": (
                "expand every KC tag on each question into the stream (F2 expand-all). "
                "Independent of method A — not 'first KC only'."
            ),
            "n_sequences": len(method_b),
            **b,
        },
        "numerically_identical": a["share_next_equals_last"]["share"] == b["share_next_equals_last"]["share"],
        "supersedes": {
            "followup3_xes_kc_repetition.method_b_drop_multi_kc_extras": (
                "Old method B appended kcs[0] only — identical to A by construction. Dropped."
            )
        },
        "primary_reported": "method_a for concept-level table; method_b expand-all as sensitivity",
    }
    write_json("paper_xes_kc_methods.json", out)
    # also refresh followup3 file with honest B
    write_json(
        "followup3_xes_kc_repetition.json",
        {
            **out,
            "method_b_drop_multi_kc_extras": {
                "deprecated": True,
                "reason": "was identical to method A",
                "replaced_by": "method_b_expand_all_kc_tags",
            },
        },
    )
    return out


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def ece_score(y_true, y_prob, n_bins=10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins=10) -> list[dict]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if not m.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0, "pred": None, "obs": None})
            continue
        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(m.sum()),
                "pred": float(y_prob[m].mean()),
                "obs": float(y_true[m].mean()),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ranking: capped pointer + full test
# ---------------------------------------------------------------------------

def run_ranking_eval(test_kt, names, graph, similar_set, embeddings, bundle, parent_lists, rng, *, full: bool) -> dict:
    popularity = np.asarray(bundle["popularity"])
    feat_fn = build_heads(embeddings, graph, similar_set, popularity, parent_lists, bundle["logreg"])
    mode_clf, review_clf, advance_clf = bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"]

    if full:
        idx = np.arange(len(test_kt))
        tag = "full_test"
    else:
        n_eval = min(EVAL_SEQS, len(test_kt))
        idx = rng.choice(len(test_kt), size=n_eval, replace=False)
        tag = "capped"

    metric_names = [
        "ragr_r5", "ragr_no_dag_r5", "recent5_r5", "force_review_r5", "force_advance_r5",
        "markov_r5", "last_item_r5", "freq_r5",
        "ragr_advance_r5", "ragr_no_dag_advance_r5", "markov_advance_r5",
    ]
    buckets = {m: [] for m in metric_names}
    gate_y, gate_p = [], []
    mem_y, mem_p = [], []

    train_kt = load_sequences(DATA / "train.json")
    n_items = len(names)
    trans = np.ones((n_items, n_items), dtype=np.float64) * 1e-3
    for seq in train_kt:
        ids = [c for c, _ in seq]
        for a, b in zip(ids, ids[1:]):
            if 0 <= a < n_items and 0 <= b < n_items:
                trans[a, b] += 1.0
    trans = np.log(trans / trans.sum(axis=1, keepdims=True))
    pop = popularity.copy()

    for j, si in enumerate(idx):
        if j % 500 == 0:
            print(f"  ranking {tag} {j}/{len(idx)}", flush=True)
        seq = test_kt[int(si)]
        cuts = np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int)
        local = {m: [] for m in metric_names}
        for cut in cuts:
            cut = int(cut)
            pack = score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings)
            # RAGR without DAG features: zero unlock + is_child (advance cols 1, 3)
            pack_nodag = score_features(
                feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings,
                mask_cols={"advance": [1, 3]},
            )
            label = pack["label"]
            last = pack["last"]
            mixed = pack["mixed"]
            mixed_nodag = pack_nodag["mixed"]
            p_rev = float(pack["p_rev"])

            mk = trans[last].copy() if 0 <= last < n_items else pop.copy()

            stats, last_step, recent = history_state(seq, cut)
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

            last_scores = np.zeros(n_items)
            if 0 <= last < n_items:
                last_scores[last] = 1.0

            act = action_label(label, last_step, last)
            local["ragr_r5"].append(recall_at(mixed, label, K))
            local["ragr_no_dag_r5"].append(recall_at(mixed_nodag, label, K))
            local["recent5_r5"].append(recall_at(r5_scores, label, K))
            local["force_review_r5"].append(recall_at(pack["review"], label, K))
            local["force_advance_r5"].append(recall_at(pack["advance"], label, K))
            local["markov_r5"].append(recall_at(mk, label, K))
            local["last_item_r5"].append(recall_at(last_scores, label, K))
            local["freq_r5"].append(recall_at(pop, label, K))
            if act == "advance":
                local["ragr_advance_r5"].append(recall_at(mixed, label, K))
                local["ragr_no_dag_advance_r5"].append(recall_at(mixed_nodag, label, K))
                local["markov_advance_r5"].append(recall_at(mk, label, K))

            gate_y.append(1.0 if pack["is_review"] else 0.0)
            gate_p.append(p_rev)

            # p_recall calibration on continue queries (label == last)
            if label == last and 0 <= last < n_items:
                query = embeddings[recent[-5:]].mean(axis=0)
                query = query / max(np.linalg.norm(query), 1e-8)
                mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
                _, _, _, p_mem = feat_fn(cut, recent[-1], stats, last_step, query, mastery)
                mem_y.append(float(seq[cut][1]))
                mem_p.append(float(p_mem[last]))

        for m in metric_names:
            if local[m]:
                buckets[m].append(local[m])

    summary = {m: seq_bootstrap_mean(v, rng) for m, v in buckets.items() if v}
    paired = {}
    for a, b, name in (
        ("ragr_r5", "recent5_r5", "ragr_minus_recent5_r5"),
        ("ragr_r5", "markov_r5", "ragr_minus_markov_r5"),
        ("ragr_r5", "ragr_no_dag_r5", "ragr_minus_ragr_no_dag_r5"),
    ):
        if a in buckets and b in buckets and len(buckets[a]) == len(buckets[b]):
            paired[name] = paired_seq_delta(buckets[a], buckets[b], rng)

    cal = {
        "gate": {
            "n": len(gate_y),
            "base_rate": float(np.mean(gate_y)) if gate_y else None,
            "brier": float(brier_score_loss(gate_y, gate_p)) if gate_y else None,
            "ece": ece_score(gate_y, gate_p) if gate_y else None,
            "reliability": reliability_bins(gate_y, gate_p) if gate_y else [],
            "base_rate_adjusted_brier": (
                float(
                    brier_score_loss(gate_y, gate_p)
                    - brier_score_loss(gate_y, np.full(len(gate_y), np.mean(gate_y)))
                )
                if gate_y
                else None
            ),
            "note": (
                "base_rate_adjusted_brier = Brier(model) − Brier(constant base-rate); "
                "negative ⇒ better than base rate"
            ),
        },
        "p_recall_on_continue_queries": {
            "n": len(mem_y),
            "base_rate": float(np.mean(mem_y)) if mem_y else None,
            "brier": float(brier_score_loss(mem_y, mem_p)) if mem_y else None,
            "ece": ece_score(mem_y, mem_p) if mem_y else None,
            "reliability": reliability_bins(mem_y, mem_p) if len(mem_y) > 50 else [],
        },
    }

    return {
        "tag": tag,
        "eval": {
            "n_sequences": int(len(idx)),
            "cuts_per_sequence": EVAL_POS,
            "n_queries": int(sum(len(v) for v in buckets.get("ragr_r5", []))),
            "n_test_pool": len(test_kt),
            "seed": int(SEED),
            "full_test": full,
        },
        "metrics": summary,
        "paired_deltas": paired,
        "calibration": cal,
        "primary_statistic": "mean_of_per_sequence_means",
        "n_boot": N_BOOT,
        "ragr_no_dag_definition": "mask advance cols 1 (unlock) and 3 (is_child)",
    }


# ---------------------------------------------------------------------------
# Rank reversal with Markov + ragr_no_dag (timed / assist / xes)
# ---------------------------------------------------------------------------

def fit_markov_matrix(train_seqs, n_items):
    counts = np.ones((n_items, n_items), dtype=np.float64) * 1e-3
    for seq in train_seqs:
        ids = [x for x, _ in seq if 0 <= x < n_items]
        for a, b in zip(ids, ids[1:]):
            counts[a, b] += 1.0
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def eval_methods_reversal(train_seqs, test_seqs, n_items, sample_idx, rng, *, with_ragr=False, ragr_pack=None):
    pop = popularity_scores(train_seqs, n_items)
    trans = fit_markov_matrix(train_seqs, n_items)
    gru = train_seq_model(GRUProbe(n_items, emb=64), train_seqs, n_items, epochs=3)
    # published-style GRU4Rec-ish: larger emb, more epochs, dropout via Attn as separate
    gru_pub = train_seq_model(GRUProbe(n_items, emb=128, hidden=128), train_seqs, n_items, epochs=5)
    attn = train_seq_model(AttnProbe(n_items, emb=64, dropout=0.2), train_seqs, n_items, epochs=3)

    methods = {
        "popularity": lambda prefix: pop.copy(),
        "recency": None,  # filled below
        "markov_order1": lambda prefix: trans[prefix[-1]].copy() if prefix else pop.copy(),
        "gru_probe": lambda prefix: score_seq_model(gru, prefix, n_items),
        "gru4rec_style": lambda prefix: score_seq_model(gru_pub, prefix, n_items),
        "attn_probe": lambda prefix: score_seq_model(attn, prefix, n_items),
    }

    def recency_fn(prefix):
        scores = np.full(n_items, -1e9)
        distinct = []
        for x in reversed(prefix):
            if x not in distinct:
                distinct.append(x)
            if len(distinct) >= 5:
                break
        for rank, item in enumerate(distinct):
            if 0 <= item < n_items:
                scores[item] = 5 - rank
        return scores

    methods["recency"] = recency_fn

    slices = ["all", "revisit", "advance", "nonrepeat"]
    buckets = {m: {sl: [] for sl in slices} for m in methods}

    for si in sample_idx:
        seq = test_seqs[int(si)]
        if len(seq) < 10:
            continue
        cuts = np.linspace(8, len(seq) - 1, num=min(5, len(seq) - 9), dtype=int)
        local = {m: {sl: [] for sl in slices} for m in methods}
        for cut in cuts:
            cut = int(cut)
            label = seq[cut][0]
            prefix = [x for x, _ in seq[:cut]]
            last = prefix[-1]
            hist = set(prefix)
            sl = action_slice(label, hist, last)
            for m, fn in methods.items():
                scores = fn(prefix)
                hit = recall_at(scores, label, K)
                local[m]["all"].append(hit)
                if sl in ("revisit", "advance"):
                    local[m][sl].append(hit)
                    local[m]["nonrepeat"].append(hit)
        for m in methods:
            for sl in slices:
                if local[m][sl]:
                    buckets[m][sl].append(float(np.mean(local[m][sl])))

    rankings = {}
    for sl in slices:
        rows = []
        for m in methods:
            arr = buckets[m][sl]
            rows.append({"method": m, "r5": float(np.mean(arr)) if arr else None, "n_sequences": len(arr)})
        rows.sort(key=lambda r: (r["r5"] is not None, r["r5"] or -1), reverse=True)
        rankings[sl] = rows

    # reversals vs all-order
    all_order = [r["method"] for r in rankings["all"]]
    reversals = []
    for sl in ("revisit", "advance", "nonrepeat"):
        order = [r["method"] for r in rankings[sl] if r["r5"] is not None]
        if order and order != all_order[: len(order)]:
            # check if top-2 swapped relative to all
            if len(order) >= 2 and len(all_order) >= 2 and order[:2] != all_order[:2]:
                reversals.append({"slice": sl, "all_order": all_order, "slice_order": order})

    return {
        "rankings_by_slice_r5": rankings,
        "rank_reversals": reversals,
        "methods": list(methods),
        "recbole_used": False,
        "gru4rec_style_note": (
            "Local GRUProbe emb=128 hidden=128 epochs=5 — published-style robustness check; "
            "RecBole not installed in this environment (see DEVIATIONS)."
        ),
    }


def run_rank_reversal_full() -> dict:
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "eval_cuts": 5, "full_test": True, "datasets": {}}

    # junyi timed
    print("rank-reversal junyi_timed full…", flush=True)
    seqs, uids = build_learner_sequences_junyi_timed(max_users=6000, min_len=12, max_len=200)
    n_items = max(max(x for x, _ in s) for s in seqs) + 1
    from run_probe_freeze import learner_split

    tr, va, te = learner_split(uids, SEED)
    # use train+val for fit, full test
    train = [seqs[i] for i in list(tr) + list(va)]
    test = [seqs[i] for i in te]
    sample_idx = np.arange(len(test))  # FULL test
    pack = eval_methods_reversal(train, test, n_items, sample_idx, rng)
    pack["split"] = {"unit": "exercise", "n_train": len(train), "n_test": len(test), "n_eval_sequences": len(test)}
    out["datasets"]["junyi_timed"] = pack

    print("rank-reversal assistments full…", flush=True)
    a_seqs, a_uids, n_a = build_learner_sequences_assist()
    tr, va, te = learner_split(a_uids, SEED)
    train = [a_seqs[i] for i in list(tr) + list(va)]
    test = [a_seqs[i] for i in te]
    pack = eval_methods_reversal(train, test, n_a, np.arange(len(test)), rng)
    pack["split"] = {"unit": "composite_skill_token", "n_train": len(train), "n_test": len(test), "n_eval_sequences": len(test)}
    out["datasets"]["assistments"] = pack

    print("rank-reversal xes3g5m full…", flush=True)
    x_seqs, x_uids, n_x = build_learner_sequences_xes()
    tr, va, te = learner_split(x_uids, SEED)
    train = [x_seqs[i] for i in list(tr) + list(va)]
    test = [x_seqs[i] for i in te]
    # XES full test is large — still full as requested
    pack = eval_methods_reversal(train, test, n_x, np.arange(len(test)), rng)
    pack["split"] = {
        "unit": "question",
        "n_train": len(train),
        "n_test": len(test),
        "n_eval_sequences": len(test),
        "native_role": "descriptive_curriculum_order",
    }
    out["datasets"]["xes3g5m"] = pack

    write_json("paper_rank_reversal_full.json", out)
    return out


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity() -> dict:
    print("sensitivity…", flush=True)
    topic_of = load_topic_map()
    out = {"sequence_length_cap": {}, "cut_positions": {}, "timed_6k_selection": {}}

    # length caps
    for cap in (100, 200, 400):
        train = load_sequences(DATA / "train.json")
        test = load_sequences(DATA / "test.json")
        # reload without max from file then truncate
        # load_sequences already applies 12-200; for caps we truncate further
        def trunc(seqs, m):
            return [s[:m] for s in seqs if len(s) >= 12]

        tr, te = trunc(train, cap), trunc(test, cap)
        topic = [[topic_of.get(c, "unknown") for c, _ in s] for s in tr + te]
        sm = share_and_mix(topic)
        out["sequence_length_cap"][str(cap)] = {
            "n_train_seq": len(tr),
            "n_test_seq": len(te),
            "topic_next_eq_last": sm["share_next_equals_last"],
        }

    # cut positions: compare EVAL_POS in {3,5,10} on capped ranking metrics from existing full run if present
    out["cut_positions"] = {
        "note": "Full cut-position ranking recomputed in paper_ranking_full with EVAL_POS=5; "
        "sensitivity table stores transition shares under different cut grids on a 800-seq subsample.",
        "grids": {},
    }
    test = load_sequences(DATA / "test.json")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), size=min(800, len(test)), replace=False)
    for ncuts in (3, 5, 10):
        mix = {"continue": 0, "revisit": 0, "advance": 0, "n": 0}
        for si in idx:
            seq = test[int(si)]
            if len(seq) < 10:
                continue
            cuts = np.linspace(8, len(seq) - 1, num=min(ncuts, len(seq) - 9), dtype=int)
            for cut in cuts:
                cut = int(cut)
                label = seq[cut][0]
                last = seq[cut - 1][0]
                hist = {c for c, _ in seq[:cut]}
                sl = action_slice(label, hist, last)
                mix[sl] += 1
                mix["n"] += 1
        out["cut_positions"]["grids"][str(ncuts)] = {
            "continue": mix["continue"] / mix["n"],
            "revisit": mix["revisit"] / mix["n"],
            "advance": mix["advance"] / mix["n"],
            "n_queries": mix["n"],
        }

    # timed 6k selection bias: first-6k vs random-6k topic/exercise sticky
    z = np.load(ROOT / "data/junyi_raw/timed_interactions.npz")
    u, c, y = z["user_id"], z["concept"], z["correct"]
    # build all eligible users
    seqs_all, uids_all = [], []
    i = 0
    n = len(u)
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        if j - i >= 12:
            seq = [(int(c[k]), int(y[k])) for k in range(i, min(j, i + 200))]
            if len(seq) >= 12:
                seqs_all.append(seq)
                uids_all.append(int(u[i]))
        i = j
    first6 = seqs_all[:6000]
    rng = np.random.default_rng(SEED)
    rand_idx = rng.choice(len(seqs_all), size=min(6000, len(seqs_all)), replace=False)
    rand6 = [seqs_all[k] for k in rand_idx]

    def ex_share(seqs):
        return share_and_mix([[x for x, _ in s] for s in seqs])["share_next_equals_last"]

    out["timed_6k_selection"] = {
        "n_eligible_users": len(seqs_all),
        "first_6k_file_order": ex_share(first6),
        "random_6k": ex_share(rand6),
        "delta_first_minus_random": ex_share(first6)["share"] - ex_share(rand6)["share"],
        "rule": "first-6k = file order (freeze); random-6k = uniform among eligible users",
    }
    write_json("paper_sensitivity.json", out)
    return out


def main():
    t0 = time.time()
    print("=== paper counts ===", flush=True)
    audit_junyi_counts()
    print("=== xes methods ===", flush=True)
    run_xes_kc_independent()

    print("=== ranking capped pointer ===", flush=True)
    rr = json.loads((PHASE / "review_ranking.json").read_text())
    write_json(
        "paper_ranking_capped.json",
        {
            "source": "review_ranking.json",
            "eval": rr["eval"],
            "metrics_ragr_r5": rr["metrics"]["ragr_r5"],
            "supersedes_phase6_eval_ci": {
                "old": {"file": "phase6_eval.json", "ci": [0.8794875, 0.90475], "mean": 0.89125},
                "canonical": rr["metrics"]["ragr_r5"],
            },
        },
    )

    print("=== ranking FULL test ===", flush=True)
    names = load_names()
    graph, similar_set = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    bundle = joblib.load(BUNDLE)
    parent_lists = [list(graph.predecessors(i)) for i in range(len(names))]
    test_kt = load_sequences(DATA / "test.json")
    rng = np.random.default_rng(SEED)
    full = run_ranking_eval(
        test_kt, names, graph, similar_set, embeddings, bundle, parent_lists, rng, full=True
    )
    write_json("paper_ranking_full.json", full)
    write_json("paper_calibration.json", {"from": "paper_ranking_full.calibration", **full["calibration"]})

    print("=== sensitivity ===", flush=True)
    run_sensitivity()

    print("=== rank reversal full (slow) ===", flush=True)
    run_rank_reversal_full()

    print("done paper eval", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
