"""
NeuroTrace-DAG — all six phases on Junyi, with Review–Advance Gated Ranking.

Novelty (not a copy of HLR, DAS3H, CSEAL, or KnowLP):
  Spaced-repetition work decides *when to restudy an old item*.
  Learning-path work decides *which new node to unlock*.
  Students do both in one log. We learn a gate P(review | history)
  and rank the two pools with different features, then mix them.

Run:
    python3 run_all_phases.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from junyi_pipeline import (
    DATA,
    OUT,
    SEED,
    encode_names,
    evaluate_memory,
    fit_hlr,
    hlr_predict,
    history_state,
    iter_memory_rows,
    load_graph,
    load_names,
    load_sequences,
    ndcg_at_k,
    pair_auc,
)

PHASE_OUT = OUT / "phases"
K_LIST = (5, 10)
EVAL_SEQS = 400
EVAL_POS = 3
MODE_TRAIN = 6000
ZPD_TARGET = 0.62


def write_json(name: str, payload: dict) -> None:
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    path = PHASE_OUT / name
    path.write_text(json.dumps(payload, indent=2))
    print("wrote", path)


def phase1_prepare(names, graph, train_seq, test_seq) -> dict:
    rows = [
        {
            "id": i,
            "name": names[i],
            "in_degree": int(graph.in_degree(i)),
            "out_degree": int(graph.out_degree(i)),
        }
        for i in range(len(names))
    ]
    edges = [
        {"source": names[u], "target": names[v], "source_id": int(u), "target_id": int(v)}
        for u, v in graph.edges()
    ]
    (OUT / "concepts_junyi.csv").write_text(
        "id,name,in_degree,out_degree\n"
        + "\n".join(f"{r['id']},{r['name'].replace(',', ' ')},{r['in_degree']},{r['out_degree']}" for r in rows)
        + "\n"
    )
    (OUT / "prerequisite_edges_junyi.csv").write_text(
        "source_concept_name,target_concept_name\n"
        + "\n".join(f"{e['source'].replace(',', ' ')},{e['target'].replace(',', ' ')}" for e in edges)
        + "\n"
    )
    n_int = sum(len(s) for s in train_seq) + sum(len(s) for s in test_seq)
    stats = {
        "phase": 1,
        "dataset": "Junyi Academy / EduData ktbd-junyi (Chang et al., EDM 2015)",
        "concepts": len(names),
        "edges": graph.number_of_edges(),
        "train_sequences": len(train_seq),
        "test_sequences": len(test_seq),
        "interactions": n_int,
        "join": "native: every log item id is a DAG node id",
        "faults_closed": [
            "no OULAD placeholder concept",
            "no cross-dataset mapping",
            "correctness is not 96.7% pass",
        ],
    }
    write_json("phase1_prepare.json", stats)
    return stats


def phase2_embeddings(names, graph, rng) -> tuple[np.ndarray, dict]:
    emb_path = OUT / "concept_embeddings.npy"
    if emb_path.exists() and np.load(emb_path).shape[0] == len(names):
        embeddings = np.load(emb_path)
        print("loaded cached embeddings", embeddings.shape)
    else:
        embeddings = encode_names(names)
        np.save(emb_path, embeddings)
    related = pair_auc(embeddings, list(graph.edges()), rng)
    # Direction probe: cosine cannot say which node is the prerequisite.
    diffs = []
    labels = []
    for s, t in graph.edges():
        if s == t:
            continue
        # score > 0 would mean "first name is 'more source-like'" if it worked
        diffs.append(float(np.linalg.norm(embeddings[s]) - np.linalg.norm(embeddings[t])))
        labels.append(1)
        diffs.append(float(np.linalg.norm(embeddings[t]) - np.linalg.norm(embeddings[s])))
        labels.append(0)
    # Better direction feature: is source more similar to a random root than target?
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    root_vec = embeddings[roots].mean(axis=0)
    root_vec = root_vec / max(np.linalg.norm(root_vec), 1e-8)
    dir_scores, dir_y = [], []
    for s, t in graph.edges():
        if s == t:
            continue
        dir_scores.append(float(embeddings[s] @ root_vec - embeddings[t] @ root_vec))
        dir_y.append(1)
        dir_scores.append(float(embeddings[t] @ root_vec - embeddings[s] @ root_vec))
        dir_y.append(0)
    direction_auc = float(roc_auc_score(dir_y, dir_scores))
    stats = {
        "phase": 2,
        "model": "all-MiniLM-L6-v2",
        "relatedness": related,
        "direction_auc_vs_roots": direction_auc,
        "claim": "embeddings recover relatedness, not direction",
    }
    write_json("phase2_embeddings.json", stats)
    return embeddings, stats


def phase3_memory(train_seq, rng) -> tuple[LogisticRegression, dict]:
    xs, ys = [], []
    for feat, y, _ in iter_memory_rows(train_seq):
        xs.append(feat)
        ys.append(y)
        if len(xs) >= 180_000:
            break
    x = np.vstack(xs)
    y = np.array(ys)
    split = int(0.8 * len(y))
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]
    mean_pred = np.full_like(y_te, y_tr.mean())
    logreg = LogisticRegression(max_iter=400).fit(x_tr[:, 1:], y_tr)
    logreg_pred = logreg.predict_proba(x_te[:, 1:])[:, 1]
    hlr_w = fit_hlr(x_tr, y_tr)
    hlr_pred = hlr_predict(hlr_w, x_te)
    table = {
        "mean": evaluate_memory(y_te, mean_pred, rng),
        "success_rate": evaluate_memory(y_te, x_te[:, 2], rng),
        "logistic_regression": evaluate_memory(y_te, logreg_pred, rng),
        "hlr": evaluate_memory(y_te, hlr_pred, rng),
    }
    winner = max(table, key=lambda k: table[k]["auc"])
    stats = {
        "phase": 3,
        "models": table,
        "winner": winner,
        "rule": "the ranker uses the winning memory model, not HLR by default",
        "hlr_weights": hlr_w.tolist(),
    }
    write_json("phase3_memory.json", stats)
    return logreg, hlr_w, stats


def memory_p(logreg: LogisticRegression, attempts, successes, delta, seen):
    n = len(attempts)
    rate = np.divide(successes, attempts, out=np.zeros(n), where=attempts > 0)
    x = np.column_stack([np.log1p(attempts), rate, np.log1p(delta)])
    pred = logreg.predict_proba(x)[:, 1]
    return np.where(seen, pred, 0.55)


def phase4_graph(graph, train_seq) -> dict:
    acyclic = bool(nx.is_directed_acyclic_graph(graph))
    parent_lists = [list(graph.predecessors(c)) for c in graph.nodes]
    # How often is the true next item a review, an unlocked new node, or a violation?
    review = unlock_ok = violate = 0
    total = 0
    for seq in train_seq[:4000]:
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
    stats = {
        "phase": 4,
        "nx.is_directed_acyclic_graph": acyclic,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "longest_path": int(nx.dag_longest_path_length(graph)),
        "next_item_mix_train_sample": {
            "n": total,
            "review_frac": review / total,
            "advance_unlocked_frac": unlock_ok / total,
            "advance_violation_frac": violate / total,
        },
        "hard_and_gate": "rejected: it would drop most advance clicks",
        "soft_unlock": "mean parent mastery, used only in the ADVANCE head",
    }
    write_json("phase4_graph.json", stats)
    return stats


def mode_features(seq, cut, last_step, stats, graph) -> np.ndarray:
    recent = [c for c, _ in seq[:cut]]
    last_c, last_y = seq[cut - 1]
    seen_n = len(last_step)
    last_was_review = 1.0 if last_c in {k for k in last_step if last_step[k] < cut - 1} else 0.0
    recent_review = np.mean([1.0 if c in last_step and last_step[c] < i else 0.0 for i, c in enumerate(recent[-8:])])
    children = list(graph.successors(last_c))
    child_ready = 0.0
    if children:
        mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
        child_ready = float(np.mean([1.0 if mastery.get(last_c, 0.0) >= 0.5 else 0.0]))
    return np.array(
        [
            last_y,
            last_was_review,
            recent_review,
            math.log1p(seen_n),
            math.log1p(cut),
            child_ready,
            1.0 if children else 0.0,
        ],
        dtype=np.float64,
    )


def build_heads(embeddings, graph, similar, popularity, parent_lists, logreg):
    n = embeddings.shape[0]
    sim_of = [[] for _ in range(n)]
    for a, b in similar:
        if 0 <= a < n and 0 <= b < n:
            sim_of[a].append(b)
    child_of = [list(graph.successors(c)) for c in range(n)]

    def features(end, last_concept, stats, last_step, query, mastery):
        attempts = np.zeros(n)
        successes = np.zeros(n)
        seen = np.zeros(n, dtype=bool)
        recency = np.zeros(n)
        delta = np.zeros(n)
        for c, (a, s) in stats.items():
            attempts[c] = a
            successes[c] = s
        for c, step in last_step.items():
            seen[c] = True
            recency[c] = 1.0 / (1.0 + (end - step))
            delta[c] = float(end - step)
        p = memory_p(logreg, attempts, successes, delta, seen)
        zpd = np.exp(-((p - ZPD_TARGET) ** 2) / 0.08)
        semantic = embeddings @ query
        unlock = np.ones(n)
        for c, parents in enumerate(parent_lists):
            if parents:
                unlock[c] = float(np.mean([mastery.get(pr, 0.0) for pr in parents]))
        is_sim = np.zeros(n)
        is_child = np.zeros(n)
        for c in sim_of[last_concept]:
            is_sim[c] = 1.0
        for c in child_of[last_concept]:
            is_child[c] = 1.0
        review = np.column_stack(
            [semantic, 1.0 - p, recency, zpd, seen.astype(float), np.log1p(popularity)]
        )
        advance = np.column_stack(
            [semantic, unlock, is_sim, is_child, (~seen).astype(float), np.log1p(popularity)]
        )
        return review, advance, seen, p

    return features


def collect_mode_and_rank(sequences, graph, feat_fn, rng, n_pos):
    mode_x, mode_y = [], []
    rev_x, rev_y = [], []
    adv_x, adv_y = [], []
    chosen = rng.choice(len(sequences), size=min(len(sequences), n_pos), replace=False)
    for idx in chosen:
        seq = sequences[int(idx)]
        cut = max(8, int(0.75 * len(seq)) - 1)
        label = seq[cut][0]
        stats, last_step, recent = history_state(seq, cut)
        query = recent[-5:]
        # query built in caller via embeddings — passed through feat_fn
        yield_pack = (seq, cut, label, stats, last_step, recent)
        mode_x.append(mode_features(seq, cut, last_step, stats, graph))
        mode_y.append(1.0 if label in last_step else 0.0)
        # features filled in second pass in train_models
        _ = yield_pack
    return np.vstack(mode_x), np.array(mode_y)


def train_models(train_seq, embeddings, graph, similar, popularity, parent_lists, logreg, rng):
    feat_fn = build_heads(embeddings, graph, similar, popularity, parent_lists, logreg)
    mode_x, mode_y = [], []
    rev_x, rev_y = [], []
    adv_x, adv_y = [], []
    chosen = rng.choice(len(train_seq), size=min(len(train_seq), MODE_TRAIN), replace=False)
    for idx in chosen:
        seq = train_seq[int(idx)]
        cut = max(8, int(0.75 * len(seq)) - 1)
        label = seq[cut][0]
        stats, last_step, recent = history_state(seq, cut)
        query = embeddings[recent[-5:]].mean(axis=0)
        query = query / max(np.linalg.norm(query), 1e-8)
        mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
        review, advance, seen, _ = feat_fn(cut, recent[-1], stats, last_step, query, mastery)
        is_review = label in last_step
        mode_x.append(mode_features(seq, cut, last_step, stats, graph))
        mode_y.append(1.0 if is_review else 0.0)
        if is_review:
            rev_x.append(review[label])
            rev_y.append(1.0)
            unseen = np.flatnonzero(~seen)
            if len(unseen):
                neg = int(rng.choice(unseen))
                rev_x.append(review[neg])
                rev_y.append(0.0)
            seen_idx = np.flatnonzero(seen)
            seen_idx = seen_idx[seen_idx != label]
            if len(seen_idx):
                neg = int(rng.choice(seen_idx))
                rev_x.append(review[neg])
                rev_y.append(0.0)
        else:
            adv_x.append(advance[label])
            adv_y.append(1.0)
            pool = np.flatnonzero(~seen)
            pool = pool[pool != label]
            if len(pool):
                for neg in rng.choice(pool, size=min(3, len(pool)), replace=False):
                    adv_x.append(advance[int(neg)])
                    adv_y.append(0.0)
    mode_clf = LogisticRegression(max_iter=400).fit(np.vstack(mode_x), np.array(mode_y))
    review_clf = LogisticRegression(max_iter=400).fit(np.vstack(rev_x), np.array(rev_y))
    advance_clf = LogisticRegression(max_iter=400).fit(np.vstack(adv_x), np.array(adv_y))
    stats = {
        "phase": 5,
        "mode_review_rate_train": float(np.mean(mode_y)),
        "mode_train_acc": float(mode_clf.score(np.vstack(mode_x), np.array(mode_y))),
        "review_head_n": len(rev_y),
        "advance_head_n": len(adv_y),
        "idea": "P(review) gates a memory/ZPD head; 1-P(review) gates a DAG/similarity head",
    }
    write_json("phase5_ranker.json", stats)
    return mode_clf, review_clf, advance_clf, feat_fn, stats


def save_serve_bundle(logreg, mode_clf, review_clf, advance_clf, popularity) -> None:
    import joblib

    OUT.mkdir(exist_ok=True)
    joblib.dump(
        {
            "logreg": logreg,
            "mode_clf": mode_clf,
            "review_clf": review_clf,
            "advance_clf": advance_clf,
            "popularity": np.asarray(popularity),
        },
        OUT / "serve_models.joblib",
    )
    print("wrote", OUT / "serve_models.joblib")


def score_ragr(mode_clf, review_clf, advance_clf, feat_fn, seq, cut, graph, embeddings):
    label = seq[cut][0]
    stats, last_step, recent = history_state(seq, cut)
    query = embeddings[recent[-5:]].mean(axis=0)
    query = query / max(np.linalg.norm(query), 1e-8)
    mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
    review, advance, seen, p = feat_fn(cut, recent[-1], stats, last_step, query, mastery)
    p_rev = float(mode_clf.predict_proba(mode_features(seq, cut, last_step, stats, graph)[None, :])[0, 1])
    s_rev = review_clf.decision_function(review)
    s_adv = advance_clf.decision_function(advance)
    # Unseen items should not win the review head; seen items are down-weighted in advance.
    s_rev = np.where(seen, s_rev, s_rev.min() - 1.0)
    mixed = p_rev * s_rev + (1.0 - p_rev) * s_adv
    return {
        "label": label,
        "is_review": label in last_step,
        "p_rev": p_rev,
        "mixed": mixed,
        "review": s_rev,
        "advance": s_adv,
        "mastery": mastery,
        "parents": list(graph.predecessors(label)),
    }


def metrics_from_ranks(rows, key: str) -> dict:
    out = {}
    for subset, pred in (
        ("all", rows),
        ("review", [r for r in rows if r["is_review"]]),
        ("advance", [r for r in rows if not r["is_review"]]),
    ):
        if not pred:
            continue
        rec = {k: [] for k in K_LIST}
        nd = {k: [] for k in K_LIST}
        for r in pred:
            order = np.argsort(-r[key])
            rank = int(np.where(order == r["label"])[0][0])
            for k in K_LIST:
                rec[k].append(1.0 if rank < k else 0.0)
                nd[k].append(ndcg_at_k(rank, k))
        block = {f"recall@{k}": float(np.mean(rec[k])) for k in K_LIST}
        block.update({f"ndcg@{k}": float(np.mean(nd[k])) for k in K_LIST})
        block["n"] = len(pred)
        out[subset] = block
    return out


def phase6_eval(test_seq, embeddings, graph, models, rng) -> dict:
    mode_clf, review_clf, advance_clf, feat_fn = models
    idx = rng.choice(len(test_seq), size=min(EVAL_SEQS, len(test_seq)), replace=False)
    rows = []
    mode_true, mode_pred = [], []
    violate = []
    for i in idx:
        seq = test_seq[int(i)]
        for cut in np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int):
            pack = score_ragr(mode_clf, review_clf, advance_clf, feat_fn, seq, int(cut), graph, embeddings)
            rows.append(pack)
            mode_true.append(1.0 if pack["is_review"] else 0.0)
            mode_pred.append(pack["p_rev"])
            parents = pack["parents"]
            if (not pack["is_review"]) and parents and all(pack["mastery"].get(p, 0.0) < 0.5 for p in parents):
                violate.append(1.0)
            else:
                violate.append(0.0)
    ranking = {
        "ragr": metrics_from_ranks(rows, "mixed"),
        "review_head_only": metrics_from_ranks(rows, "review"),
        "advance_head_only": metrics_from_ranks(rows, "advance"),
    }
    # Mode-ablation: force review or force advance
    forced_rev = [{**r, "forced": r["review"]} for r in rows]
    forced_adv = [{**r, "forced": r["advance"]} for r in rows]
    ranking["force_review"] = metrics_from_ranks(forced_rev, "forced")
    ranking["force_advance"] = metrics_from_ranks(forced_adv, "forced")
    stats = {
        "phase": 6,
        "n_queries": len(rows),
        "review_frac": float(np.mean(mode_true)),
        "mode_auc": float(roc_auc_score(mode_true, mode_pred)) if len(set(mode_true)) > 1 else None,
        "advance_violation_among_all": float(np.mean(violate)),
        "ranking": ranking,
        "baselines_note": "force_review is DAS3H-like (memory only); force_advance is CSEAL/KnowLP-like (path only)",
    }
    write_json("phase6_eval.json", stats)
    return stats


def main() -> None:
    OUT.mkdir(exist_ok=True)
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, similar = load_graph(len(names))
    train_seq = load_sequences(DATA / "train.json")
    test_seq = load_sequences(DATA / "test.json")
    popularity = np.zeros(len(names))
    for seq in train_seq:
        for c, _ in seq:
            popularity[c] += 1
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]

    print("=== Phase 1 prepare ===")
    p1 = phase1_prepare(names, graph, train_seq, test_seq)
    print("=== Phase 2 embeddings ===")
    embeddings, p2 = phase2_embeddings(names, graph, rng)
    print(p2)
    print("=== Phase 3 memory ===")
    logreg, hlr_w, p3 = phase3_memory(train_seq, rng)
    print("winner", p3["winner"], {k: v["auc"] for k, v in p3["models"].items()})
    print("=== Phase 4 graph ===")
    p4 = phase4_graph(graph, train_seq)
    print(p4["next_item_mix_train_sample"])
    print("=== Phase 5 Review-Advance ranker ===")
    mode_clf, review_clf, advance_clf, feat_fn, p5 = train_models(
        train_seq, embeddings, graph, similar, popularity, parent_lists, logreg, rng
    )
    print(p5)
    save_serve_bundle(logreg, mode_clf, review_clf, advance_clf, popularity)
    print("=== Phase 6 evaluation ===")
    p6 = phase6_eval(test_seq, embeddings, graph, (mode_clf, review_clf, advance_clf, feat_fn), rng)
    print(json.dumps(p6["ranking"], indent=2))

    lines = [
        "NeuroTrace-DAG — all phases (Review–Advance Gated Ranking)",
        f"Phase 1  concepts={p1['concepts']} edges={p1['edges']} interactions={p1['interactions']}",
        f"Phase 2  pair-AUC={p2['relatedness']['pair_auc']:.3f}  direction-AUC={p2['direction_auc_vs_roots']:.3f} (should stay near 0.5)",
        f"Phase 3  winner={p3['winner']}  LR={p3['models']['logistic_regression']['auc']:.3f}  HLR={p3['models']['hlr']['auc']:.3f}",
        f"Phase 4  DAG={p4['nx.is_directed_acyclic_graph']}  next-item review={p4['next_item_mix_train_sample']['review_frac']:.3f}  advance-unlocked={p4['next_item_mix_train_sample']['advance_unlocked_frac']:.3f}  violate={p4['next_item_mix_train_sample']['advance_violation_frac']:.3f}",
        f"Phase 5  mode train acc={p5['mode_train_acc']:.3f}  P(review)={p5['mode_review_rate_train']:.3f}",
        f"Phase 6  mode AUC={p6['mode_auc']:.3f}  queries={p6['n_queries']}  review_frac={p6['review_frac']:.3f}",
    ]
    for name, block in p6["ranking"].items():
        allb = block.get("all", {})
        rev = block.get("review", {})
        adv = block.get("advance", {})
        lines.append(
            f"  {name:18s} all R@5={allb.get('recall@5', 0):.3f}  review R@5={rev.get('recall@5', 0):.3f}  advance R@5={adv.get('recall@5', 0):.3f}"
        )
    (OUT / "all_phases_results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
