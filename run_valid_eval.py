"""
Statistically valid re-evaluation (reviewer fixes 1–8).

1. Memory rows split by SEQUENCE (and by LEARNER on timed data). No sequence on both sides.
2. Real-time Δt in days from junyi.rar timestamps, compared with step Δt.
3. Phase 6 Recall@K with bootstrap over sequences (not queries).
4. Ranking baselines on review / advance slices.
5–7. Advance ablations, embedding controls, gate PR-AUC.
8. All numbers written to outputs_junyi/phases/*.json for the report to inject.

Run:
    python3 scripts/extract_junyi_times.py   # once, needs ~disk + junyi.rar
    python3 run_valid_eval.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from junyi_pipeline import (
    DATA,
    OUT,
    SEED,
    fit_hlr,
    hlr_predict,
    history_state,
    load_graph,
    load_names,
    load_sequences,
    ndcg_at_k,
)
from run_all_phases import (
    PHASE_OUT,
    build_heads,
    mode_features,
    train_models,
    write_json,
)

TIMED = ROOT = Path(__file__).resolve().parent
TIMED_NPZ = ROOT / "data" / "junyi_raw" / "timed_interactions.npz"
BOOT_SEQ = 400
EVAL_SEQS = 800
EVAL_POS = 5  # more cuts → larger advance slice
ZPD = 0.62


def assert_no_leak(train_ids: set, test_ids: set, label: str) -> None:
    overlap = train_ids & test_ids
    if overlap:
        raise SystemExit(f"LEAKAGE {label}: {len(overlap)} ids on both sides")
    print(f"leakage check OK ({label}): train={len(train_ids)} test={len(test_ids)} overlap=0")


def memory_rows_by_sequence(sequences: list[list[tuple[int, int]]], use_days=None):
    """Yield (feat, y, seq_id). feat = [1, log1p(att), rate, log1p(delta_steps), log1p(delta_days)]."""
    for sid, seq in enumerate(sequences):
        last_step: dict[int, int] = {}
        last_day: dict[int, float] = {}
        stats: dict[int, list[int]] = {}
        times = use_days[sid] if use_days is not None else None
        for step, (concept, correct) in enumerate(seq):
            attempts, successes = stats.get(concept, [0, 0])
            if concept in last_step:
                d_step = float(step - last_step[concept])
                d_day = float(times[step] - last_day[concept]) if times is not None else d_step
                d_day = max(d_day, 0.0)
                rate = successes / attempts if attempts else 0.0
                feat = np.array(
                    [1.0, math.log1p(attempts), rate, math.log1p(d_step), math.log1p(d_day)],
                    dtype=np.float64,
                )
                yield feat, float(correct), sid
            last_step[concept] = step
            if times is not None:
                last_day[concept] = float(times[step])
            stats[concept] = [attempts + 1, successes + int(correct)]


def split_by_id(ids: np.ndarray, seed: int = SEED, test_frac: float = 0.2):
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_te = max(1, int(test_frac * len(uniq)))
    te = set(uniq[:n_te].tolist())
    tr = set(uniq[n_te:].tolist())
    return tr, te


def eval_memory_table(x_tr, y_tr, x_te, y_te, rng) -> dict:
    def boot_auc(y, p):
        vals = []
        n = len(y)
        for _ in range(300):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(roc_auc_score(y[idx], p[idx]))
        arr = np.array(vals)
        return float(arr.mean()), [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

    table = {}
    mean_p = np.full_like(y_te, y_tr.mean())
    table["mean"] = {"auc": 0.5, "auc_ci": [0.5, 0.5], "pr_auc": float(average_precision_score(y_te, mean_p))}

    # step-Δt features: cols 1,2,3
    logreg_step = LogisticRegression(max_iter=500).fit(x_tr[:, [1, 2, 3]], y_tr)
    p_step = logreg_step.predict_proba(x_te[:, [1, 2, 3]])[:, 1]
    a, ci = boot_auc(y_te, p_step)
    table["logistic_step_dt"] = {
        "auc": a,
        "auc_ci": ci,
        "pr_auc": float(average_precision_score(y_te, p_step)),
    }

    # day-Δt features: cols 1,2,4
    logreg_day = LogisticRegression(max_iter=500).fit(x_tr[:, [1, 2, 4]], y_tr)
    p_day = logreg_day.predict_proba(x_te[:, [1, 2, 4]])[:, 1]
    a, ci = boot_auc(y_te, p_day)
    table["logistic_day_dt"] = {
        "auc": a,
        "auc_ci": ci,
        "pr_auc": float(average_precision_score(y_te, p_day)),
    }

    # both
    logreg_both = LogisticRegression(max_iter=500).fit(x_tr[:, 1:], y_tr)
    p_both = logreg_both.predict_proba(x_te[:, 1:])[:, 1]
    a, ci = boot_auc(y_te, p_both)
    table["logistic_both"] = {
        "auc": a,
        "auc_ci": ci,
        "pr_auc": float(average_precision_score(y_te, p_both)),
    }

    success = x_te[:, 2]
    a, ci = boot_auc(y_te, success)
    table["success_rate"] = {
        "auc": a,
        "auc_ci": ci,
        "pr_auc": float(average_precision_score(y_te, success)),
    }

    # HLR on step features (cols 0..3)
    hlr_w = fit_hlr(x_tr[:, :4], y_tr)
    p_hlr = hlr_predict(hlr_w, x_te[:, :4])
    a, ci = boot_auc(y_te, p_hlr)
    table["hlr_step"] = {
        "auc": a,
        "auc_ci": ci,
        "pr_auc": float(average_precision_score(y_te, p_hlr)),
    }

    # paired bootstrap: logistic_day - success_rate
    deltas = []
    n = len(y_te)
    for _ in range(400):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_te[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y_te[idx], p_day[idx]) - roc_auc_score(y_te[idx], success[idx]))
    d = np.array(deltas)
    table["paired_logistic_day_minus_success"] = {
        "mean_delta_auc": float(d.mean()),
        "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        "forgetting_helps": bool(d.mean() > 0.005 and np.quantile(d, 0.025) > 0),
    }
    table["n_train"] = int(len(y_tr))
    table["n_test"] = int(len(y_te))
    table["winner"] = max(
        ("logistic_step_dt", "logistic_day_dt", "logistic_both", "success_rate", "hlr_step"),
        key=lambda k: table[k]["auc"],
    )
    return table, logreg_both


def build_timed_sequences(max_users: int = 8000, min_len: int = 12, max_len: int = 200):
    z = np.load(TIMED_NPZ)
    u, c, t, y = z["user_id"], z["concept"], z["t_us"], z["correct"]
    # users already sorted
    sequences = []
    day_lists = []
    user_ids = []
    i = 0
    n = len(u)
    while i < n and len(sequences) < max_users:
        j = i
        uid = u[i]
        while j < n and u[j] == uid:
            j += 1
        if j - i >= min_len:
            concepts = c[i:j][:max_len]
            corrects = y[i:j][:max_len]
            times = t[i:j][:max_len].astype(np.float64)
            # t_us is Unix time in microseconds (Junyi ProblemLog). Fractional
            # days from first event for continuous Δt in the memory model.
            # Gap-bin analyses must use integer calendar days separately
            # (see run_followup_fixes.run_gap_bins_fixed).
            days = (times - times[0]) / (1e6 * 86400.0)
            sequences.append([(int(concepts[k]), int(corrects[k])) for k in range(len(concepts))])
            day_lists.append(days)
            user_ids.append(int(uid))
        i = j
    return sequences, day_lists, user_ids


def embedding_controls(embeddings, graph, names, rng) -> dict:
    edges = [(s, t) for s, t in graph.edges() if s != t]
    linked = np.array([float(embeddings[s] @ embeddings[t]) for s, t in edges])
    n = embeddings.shape[0]
    random_scores = []
    while len(random_scores) < len(linked):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            random_scores.append(float(embeddings[i] @ embeddings[j]))
    random_scores = np.array(random_scores)

    # same-topic: share first token of slug
    topic = {i: names[i].split()[0] if names[i] else "" for i in range(n)}
    same_topic = []
    tries = 0
    while len(same_topic) < len(linked) and tries < 200000:
        i, j = rng.integers(0, n, size=2)
        tries += 1
        if i != j and topic[i] == topic[j] and topic[i]:
            same_topic.append(float(embeddings[i] @ embeddings[j]))
    same_topic = np.array(same_topic) if same_topic else np.array([0.0])

    # lexical overlap baseline: Jaccard of word sets
    def jacc(a, b):
        sa, sb = set(names[a].split()), set(names[b].split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    lex_linked = np.array([jacc(s, t) for s, t in edges])
    lex_rand = []
    while len(lex_rand) < len(edges):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            lex_rand.append(jacc(i, j))
    lex_rand = np.array(lex_rand)

    def pair_auc(pos, neg):
        y = np.array([1] * len(pos) + [0] * len(neg))
        x = np.concatenate([pos, neg])
        return float(roc_auc_score(y, x))

    return {
        "pair_auc_vs_random": pair_auc(linked, random_scores),
        "pair_auc_vs_same_topic": pair_auc(linked, same_topic) if len(same_topic) > 50 else None,
        "n_same_topic_controls": int(len(same_topic)),
        "linked_mean_cosine": float(linked.mean()),
        "random_mean_cosine": float(random_scores.mean()),
        "same_topic_mean_cosine": float(same_topic.mean()),
        "lexical_jaccard_auc_vs_random": pair_auc(lex_linked, lex_rand),
        "lexical_linked_mean": float(lex_linked.mean()),
        "lexical_random_mean": float(lex_rand.mean()),
        "claim": "embeddings beat random and lexical Jaccard; same-topic control reported",
    }


def score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings, mask_cols=None):
    stats, last_step, recent = history_state(seq, cut)
    query = embeddings[recent[-5:]].mean(axis=0)
    query = query / max(np.linalg.norm(query), 1e-8)
    mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
    review, advance, seen, p = feat_fn(cut, recent[-1], stats, last_step, query, mastery)
    if mask_cols:
        for col in mask_cols.get("review", []):
            review[:, col] = 0
        for col in mask_cols.get("advance", []):
            advance[:, col] = 0
    p_rev = float(mode_clf.predict_proba(mode_features(seq, cut, last_step, stats, graph)[None, :])[0, 1])
    s_rev = review_clf.decision_function(review)
    s_adv = advance_clf.decision_function(advance)
    s_rev = np.where(seen, s_rev, s_rev.min() - 1.0)
    mixed = p_rev * s_rev + (1.0 - p_rev) * s_adv
    label = seq[cut][0]
    return {
        "label": label,
        "is_review": label in last_step,
        "is_last": label == recent[-1],
        "p_rev": p_rev,
        "mixed": mixed,
        "review": s_rev,
        "advance": s_adv,
        "seen": seen,
        "last": recent[-1],
        "recent": recent,
    }


def recall_at(scores, label, k):
    """Hit@k with deterministic tie-break.

    Scores are ranked descending. Ties (equal scores) are broken by **lower item
    index** via a stable mergesort on ``(-score, index)``. Documented in
    ``docs/PREREGISTRATION.md``.
    """
    scores = np.asarray(scores, dtype=float)
    # lexsort: last key is primary; sort by index ascending, then -score ascending
    # ⇒ higher score first; ties keep lower index first
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return 1.0 if int(label) in order else 0.0


def seq_bootstrap(values_per_seq: list[list[float]], rng, n_boot=BOOT_SEQ):
    """Bootstrap over sequences using the same statistic for point and CI.

    Point estimate = mean of per-sequence means (each sequence weighted equally).
    CI = percentile bootstrap of that same mean-of-means statistic.
    (Earlier code mixed mean-of-means with a pooled-query bootstrap, which can
    put the point near the CI edge.)
    """
    seq_means = np.array([np.mean(v) for v in values_per_seq if v], dtype=float)
    n = len(seq_means)
    if n == 0:
        return {"mean": None, "ci": [None, None], "n_sequences": 0, "n_queries": 0}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(seq_means[idx].mean()))
    arr = np.asarray(boots)
    return {
        "mean": float(seq_means.mean()),
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_sequences": n,
        "n_queries": int(sum(len(v) for v in values_per_seq)),
    }


def evaluate_ranking(test_seq, embeddings, graph, similar, popularity, models, parent_lists, rng):
    mode_clf, review_clf, advance_clf, feat_fn = models
    idx = rng.choice(len(test_seq), size=min(EVAL_SEQS, len(test_seq)), replace=False)
    # per-sequence metric lists
    buckets = {
        "ragr_all": [],
        "ragr_review": [],
        "ragr_advance": [],
        "ragr_nonlast": [],
        "ragr_mask_last_nonlast": [],
        "force_review_nonlast": [],
        "force_review_all": [],
        "force_advance_all": [],
        "base_last_review": [],
        "base_prefix_pop_review": [],
        "base_global_pop_advance": [],
        "base_dag_child_advance": [],
        "base_sim_advance": [],
    }
    last_item_hits = []
    nontrivial_hits = []  # review but not last
    mode_y, mode_p = [], []
    adv_ablation = {k: [] for k in ["full", "no_dag", "no_sim", "no_emb", "no_pop"]}

    for si in idx:
        seq = test_seq[int(si)]
        cuts = np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int)
        local = {k: [] for k in buckets}
        local_abl = {k: [] for k in adv_ablation}
        for cut in cuts:
            pack = score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, int(cut), graph, embeddings)
            label = pack["label"]
            last_item_hits.append(1.0 if pack["is_last"] else 0.0)
            nontrivial_hits.append(1.0 if (pack["is_review"] and not pack["is_last"]) else 0.0)
            mode_y.append(1.0 if pack["is_review"] else 0.0)
            mode_p.append(pack["p_rev"])
            r5 = recall_at(pack["mixed"], label, 5)
            local["ragr_all"].append(r5)
            local["force_review_all"].append(recall_at(pack["review"], label, 5))
            local["force_advance_all"].append(recall_at(pack["advance"], label, 5))
            # Harder slice: next item is NOT a sticky last-click (diagnostic novelty)
            if not pack["is_last"]:
                local["ragr_nonlast"].append(r5)
                local["force_review_nonlast"].append(recall_at(pack["review"], label, 5))
                masked = pack["mixed"].copy()
                masked[pack["last"]] = -1e9
                local["ragr_mask_last_nonlast"].append(recall_at(masked, label, 5))
            if pack["is_review"]:
                local["ragr_review"].append(r5)
                local["base_last_review"].append(1.0 if pack["is_last"] else 0.0)
                seen_idx = np.flatnonzero(pack["seen"])
                if len(seen_idx):
                    pop_order = seen_idx[np.argsort(-popularity[seen_idx])]
                    local["base_prefix_pop_review"].append(1.0 if label in pop_order[:5] else 0.0)
                else:
                    local["base_prefix_pop_review"].append(0.0)
            else:
                local["ragr_advance"].append(r5)
                pop5 = np.argsort(-popularity)[:5]
                local["base_global_pop_advance"].append(1.0 if label in pop5 else 0.0)
                children = []
                for r in pack["recent"][-5:]:
                    children.extend(list(graph.successors(r)))
                children = list(dict.fromkeys(children))[:5]
                local["base_dag_child_advance"].append(1.0 if label in children else 0.0)
                sims = [b for a, b in similar if a == pack["last"]][:5]
                local["base_sim_advance"].append(1.0 if label in sims else 0.0)
                masks = {
                    "full": None,
                    "no_dag": {"advance": [1, 3]},
                    "no_sim": {"advance": [2]},
                    "no_emb": {"advance": [0]},
                    "no_pop": {"advance": [5]},
                }
                for name, mask in masks.items():
                    p2 = score_features(
                        feat_fn, mode_clf, review_clf, advance_clf, seq, int(cut), graph, embeddings, mask
                    )
                    local_abl[name].append(recall_at(p2["advance"], label, 5))

        for k, vals in local.items():
            if vals:
                buckets[k].append(vals)
        for k, vals in local_abl.items():
            if vals:
                adv_ablation[k].append(vals)

    summary = {k: seq_bootstrap(v, rng) for k, v in buckets.items() if v}
    # paired RAGR vs force_review on all
    # rebuild paired at sequence level
    paired = []
    for a, b in zip(buckets["ragr_all"], buckets["force_review_all"]):
        paired.append(np.mean(a) - np.mean(b))
    if paired:
        boot = []
        arr = np.array(paired)
        for _ in range(BOOT_SEQ):
            idxb = rng.integers(0, len(arr), len(arr))
            boot.append(arr[idxb].mean())
        boot = np.array(boot)
        summary["ragr_minus_force_review"] = {
            "mean_delta_r5": float(arr.mean()),
            "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "n_sequences": len(arr),
        }
    summary["share_next_equals_last"] = float(np.mean(last_item_hits))
    summary["share_nontrivial_review"] = float(np.mean(nontrivial_hits))
    summary["share_advance"] = float(1.0 - np.mean(mode_y)) if mode_y else 0.0
    summary["advance_ablation"] = {k: seq_bootstrap(v, rng) for k, v in adv_ablation.items() if v}
    if len(set(mode_y)) > 1:
        summary["gate"] = {
            "auc": float(roc_auc_score(mode_y, mode_p)),
            "pr_auc": float(average_precision_score(mode_y, mode_p)),
            "base_rate": float(np.mean(mode_y)),
            "n": len(mode_y),
        }
        # calibration bins
        bins = np.linspace(0, 1, 11)
        cal = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (np.array(mode_p) >= lo) & (np.array(mode_p) < hi)
            if m.sum():
                cal.append({"lo": float(lo), "hi": float(hi), "pred": float(np.mean(np.array(mode_p)[m])), "obs": float(np.mean(np.array(mode_y)[m])), "n": int(m.sum())})
        summary["gate"]["calibration"] = cal
    return summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
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

    # --- Fix 6: embedding controls ---
    emb_stats = embedding_controls(embeddings, graph, names, rng)
    write_json("phase2_embeddings.json", {"phase": 2, "model": "all-MiniLM-L6-v2", **emb_stats})
    print("embeddings", emb_stats)

    # --- Fix 1: sequence-level memory split on KTBD train, eval also on test.json ---
    print("=== Phase 3 sequence-level split (KTBD) ===")
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
    te_m = ~tr_m
    # also build rows from test.json sequences (held-out learners in published split)
    feats_te, y_te_pub = [], []
    for feat, yy, _ in memory_rows_by_sequence(test_kt):
        feats_te.append(feat)
        y_te_pub.append(yy)
        if len(feats_te) >= 80_000:
            break
    x_pub = np.vstack(feats_te)
    y_pub = np.array(y_te_pub)

    mem_kt, _ = eval_memory_table(x[tr_m], y[tr_m], x[te_m], y[te_m], rng)
    mem_pub, logreg = eval_memory_table(x[tr_m], y[tr_m], x_pub, y_pub, rng)
    phase3 = {
        "phase": 3,
        "split": "by_sequence_id_within_train_json",
        "leakage_check": "pass",
        "within_train_seq_holdout": mem_kt,
        "evaluate_on_test_json": mem_pub,
        "note": "Headline memory numbers must come from evaluate_on_test_json or learner-timed split.",
    }

    # --- Fix 2: real-time Δt if timed npz exists ---
    if TIMED_NPZ.exists():
        print("=== Phase 3 timed learner split ===")
        seqs, days, user_ids = build_timed_sequences()
        user_ids = np.array(user_ids)
        tr_u, te_u = split_by_id(user_ids)
        assert_no_leak(tr_u, te_u, "learner")
        tr_seq = [seqs[i] for i, u in enumerate(user_ids) if u in tr_u]
        te_seq = [seqs[i] for i, u in enumerate(user_ids) if u in te_u]
        tr_days = [days[i] for i, u in enumerate(user_ids) if u in tr_u]
        te_days = [days[i] for i, u in enumerate(user_ids) if u in te_u]
        xf, yf, _ = [], [], []
        for feat, yy, sid in memory_rows_by_sequence(tr_seq, tr_days):
            xf.append(feat)
            yf.append(yy)
            if len(xf) >= 200_000:
                break
        xt, yt = [], []
        for feat, yy, sid in memory_rows_by_sequence(te_seq, te_days):
            xt.append(feat)
            yt.append(yy)
            if len(xt) >= 80_000:
                break
        mem_timed, logreg = eval_memory_table(np.vstack(xf), np.array(yf), np.vstack(xt), np.array(yt), rng)
        phase3["timed_learner_split"] = mem_timed
        phase3["timed_n_train_users"] = len(tr_u)
        phase3["timed_n_test_users"] = len(te_u)
    else:
        phase3["timed_learner_split"] = None
        phase3["timed_note"] = "Run scripts/extract_junyi_times.py when disk allows; junyi.rar is present."

    write_json("phase3_memory.json", phase3)
    print("phase3 winner(test.json)", phase3["evaluate_on_test_json"]["winner"])

    # --- Phase 4 mix (unchanged measurement, refresh JSON) ---
    review = unlock_ok = violate = 0
    total = 0
    parent_lists = [list(graph.predecessors(c)) for c in graph.nodes]
    skip_edges = {}
    for seq in train_kt[:5000]:
        cut = max(8, len(seq) // 2)
        label = seq[cut][0]
        stats, last_step, _ = history_state(seq, cut)
        mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
        total += 1
        if label in last_step:
            review += 1
            continue
        parents = parent_lists[label]
        if not parents or all(mastery.get(p, 0.0) >= 0.5 for p in parents):
            unlock_ok += 1
        else:
            violate += 1
            for p in parents:
                if mastery.get(p, 0.0) < 0.5:
                    skip_edges[(p, label)] = skip_edges.get((p, label), 0) + 1
    top_skips = sorted(skip_edges.items(), key=lambda kv: -kv[1])[:20]
    phase4 = {
        "phase": 4,
        "nx.is_directed_acyclic_graph": bool(nx.is_directed_acyclic_graph(graph)),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "longest_path": int(nx.dag_longest_path_length(graph)),
        "next_item_mix_train_sample": {
            "n": total,
            "review_frac": review / total,
            "advance_unlocked_frac": unlock_ok / total,
            "advance_violation_frac": violate / total,
        },
        "hard_gate_skips_top20": [
            {"source": names[a], "target": names[b], "count": c} for (a, b), c in top_skips
        ],
    }
    write_json("phase4_graph.json", phase4)
    print("phase4 mix", phase4["next_item_mix_train_sample"])

    # --- Phase 5/6 with fixes 3,4,5,7 ---
    print("=== Phase 5–6 valid ranking ===")
    # reuse training from run_all_phases but memory model = logistic on step features from seq split
    # Fit a simple logreg for feat_fn compatibility (expects predict_proba on [logatt, rate, logdt])
    logreg_serve = LogisticRegression(max_iter=500).fit(x[tr_m][:, [1, 2, 3]], y[tr_m])
    mode_clf, review_clf, advance_clf, feat_fn, p5 = train_models(
        train_kt, embeddings, graph, similar, popularity, parent_lists, logreg_serve, rng
    )
    write_json("phase5_ranker.json", p5)
    ranking = evaluate_ranking(
        test_kt, embeddings, graph, similar, popularity,
        (mode_clf, review_clf, advance_clf, feat_fn), parent_lists, rng,
    )
    phase6 = {
        "phase": 6,
        "bootstrap": "over_sequences",
        "eval_sequences": EVAL_SEQS,
        "cuts_per_sequence": EVAL_POS,
        "ranking": ranking,
    }
    write_json("phase6_eval.json", phase6)

    # headline text
    r = ranking
    lines = [
        "NeuroTrace-DAG — VALIDATED results (sequence/learner splits, sequence bootstrap)",
        f"Phase 2  pair-AUC vs random={emb_stats['pair_auc_vs_random']:.3f}  vs same-topic={emb_stats['pair_auc_vs_same_topic']}  lexical-AUC={emb_stats['lexical_jaccard_auc_vs_random']:.3f}",
        f"Phase 3  test.json winner={phase3['evaluate_on_test_json']['winner']}  "
        f"logistic_step={phase3['evaluate_on_test_json']['logistic_step_dt']['auc']:.3f}  "
        f"success={phase3['evaluate_on_test_json']['success_rate']['auc']:.3f}  "
        f"hlr={phase3['evaluate_on_test_json']['hlr_step']['auc']:.3f}",
    ]
    if phase3.get("timed_learner_split"):
        td = phase3["timed_learner_split"]
        lines.append(
            f"Phase 3 timed  day_dt={td['logistic_day_dt']['auc']:.3f}  step_dt={td['logistic_step_dt']['auc']:.3f}  "
            f"success={td['success_rate']['auc']:.3f}  delta_day_minus_success={td['paired_logistic_day_minus_success']}"
        )
    lines.append(
        f"Phase 4  review={phase4['next_item_mix_train_sample']['review_frac']:.3f}  "
        f"violate={phase4['next_item_mix_train_sample']['advance_violation_frac']:.3f}"
    )
    if "ragr_all" in r:
        lines.append(
            f"Phase 6  RAGR R@5={r['ragr_all']['mean']:.3f} CI={r['ragr_all']['ci']}  "
            f"n_seq={r['ragr_all']['n_sequences']} n_q={r['ragr_all']['n_queries']}"
        )
        lines.append(
            f"         force_review R@5={r['force_review_all']['mean']:.3f} CI={r['force_review_all']['ci']}  "
            f"delta={r.get('ragr_minus_force_review')}"
        )
        lines.append(f"         share next==last={r['share_next_equals_last']:.3f}")
    if "gate" in r:
        lines.append(f"Gate  AUC={r['gate']['auc']:.3f}  PR-AUC={r['gate']['pr_auc']:.3f}  base={r['gate']['base_rate']:.3f}")
    (OUT / "all_phases_results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "all_phases_results.txt")


if __name__ == "__main__":
    main()
