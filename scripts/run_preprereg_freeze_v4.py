#!/usr/bin/env python3
"""Pre-prereg freeze upgrade → schema 4.

- Re-select native winners: among configs within 0.005 of best val non-repeat R@5, lowest val CE
- Add first-order Markov (parameter-light) baseline
- Build ~120 cluster tokens (text when available; train-only fallback for XES)
- Run 24-config GRU grid on cluster sequences; record native + cluster winners
- Mark XES question-level as curriculum-order-structured / descriptive
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from run_followup3 import (
    ASSIST_CSV,
    GRUProbe,
    XES_Q,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    popularity_scores,
    score_seq_model,
)
from run_probe_freeze import (
    ATTN_PROBE_ARCHITECTURE,
    GRID,
    GRID_SPEC,
    H_REV_UNIT_POLICY,
    JUNYI_TIMED_MAX_USERS,
    MAX_LEN,
    MIN_LEN,
    MIN_SLICE_QUERIES,
    PHASE,
    REPLICATION_SEED,
    SPLIT,
    eval_r5_on_seqs,
    learner_split,
    sha_list,
    train_seq_model_early_stop,
    write_json,
    xes_question_order_structure,
)

import torch

torch.set_num_threads(1)

SELECTION_TOL = 0.005
TARGET_CLUSTERS = 120
CLUSTER_SEED = REPLICATION_SEED + 17


def select_winner(rows: list[dict]) -> dict:
    """Among configs within TOL of best val non-repeat R@5, pick lowest val CE."""
    scored = []
    for r in rows:
        r5 = r.get("val_nonrepeat_r5")
        if r5 is None and "final_val" in r:
            r5 = r["final_val"]["nonrepeat"]["mean_r5"]
        ce = r.get("best_val_ce_loss")
        if ce is None:
            ce = r.get("val_ce_loss")
        if r5 is None:
            continue
        scored.append((r, float(r5), float(ce) if ce is not None else float("inf")))
    if not scored:
        raise ValueError("no scored configs")
    best_r5 = max(s[1] for s in scored)
    pool = [s for s in scored if s[1] >= best_r5 - SELECTION_TOL]
    pool.sort(key=lambda s: (s[2], -s[1]))
    chosen, r5, ce = pool[0]
    meta = {
        "rule": (
            f"among configs with val_nonrepeat_r5 >= best_r5 - {SELECTION_TOL}, "
            "choose lowest val_ce_loss; ties → higher R@5"
        ),
        "best_r5_in_grid": best_r5,
        "tolerance": SELECTION_TOL,
        "n_in_tolerance_pool": len(pool),
        "chosen_r5": r5,
        "chosen_ce": None if ce == float("inf") else ce,
    }
    out = dict(chosen)
    out["selection_meta"] = meta
    return out


def fit_markov(train_seqs, n_items: int) -> np.ndarray:
    """Return score matrix [n_items, n_items]: log-count transitions + tiny prior."""
    counts = np.ones((n_items, n_items), dtype=np.float64) * 1e-3
    for seq in train_seqs:
        ids = [x for x, _ in seq if 0 <= x < n_items]
        for a, b in zip(ids, ids[1:]):
            counts[a, b] += 1.0
    # row-normalize to probabilities then log
    row = counts.sum(axis=1, keepdims=True)
    probs = counts / row
    return np.log(probs)


def markov_score_fn(trans_log: np.ndarray, pop: np.ndarray):
    def fn(prefix):
        if not prefix:
            return pop.copy()
        last = prefix[-1]
        if 0 <= last < trans_log.shape[0]:
            return trans_log[last].copy()
        return pop.copy()

    return fn


def map_seqs_to_clusters(seqs, item_to_cluster: dict[int, int]):
    out = []
    for seq in seqs:
        mapped = [(item_to_cluster[i], y) for i, y in seq if i in item_to_cluster]
        if mapped:
            out.append(mapped)
    return out


def dense_remap(seqs):
    vocab = {}
    remapped = []
    for seq in seqs:
        new = []
        for it, y in seq:
            if it not in vocab:
                vocab[it] = len(vocab)
            new.append((vocab[it], y))
        remapped.append(new)
    return remapped, len(vocab)


def kmeans_labels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = min(k, max(2, X.shape[0]))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(2048, max(256, X.shape[0])), n_init=10)
    return km.fit_predict(X), {"n_clusters": int(k), "inertia": float(km.inertia_)}


def cluster_junyi_timed(n_items_hint=None):
    """Text clustering: MiniLM concept embeddings (train/text artifact; no eval sequences)."""
    emb = np.load(ROOT / "outputs_junyi" / "concept_embeddings.npy")
    # timed concepts are already in 0..834 space matching embeddings
    n = emb.shape[0]
    X = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    labels, meta = kmeans_labels(X, TARGET_CLUSTERS, CLUSTER_SEED)
    item_to_cluster = {i: int(labels[i]) for i in range(n)}
    return {
        "method": "text_minilm_kmeans",
        "source": "outputs_junyi/concept_embeddings.npy (English exercise names)",
        "n_items_embedded": n,
        "target_n_clusters": TARGET_CLUSTERS,
        **meta,
        "item_to_cluster": item_to_cluster,
        "uses_eval_sequences": False,
    }


def cluster_assistments():
    """Text clustering on skill_name (ASSIST has names). Composite → mean of member skill vectors."""
    raw = pd.read_csv(
        ASSIST_CSV,
        encoding="ISO-8859-1",
        low_memory=False,
        usecols=["order_id", "user_id", "skill_id", "skill_name", "correct"],
    )
    raw = raw.dropna(subset=["user_id", "order_id", "correct"])
    raw["skill_id"] = raw["skill_id"].fillna(-1).astype(int)
    # skill_id → name
    names = (
        raw.loc[raw["skill_id"] >= 0, ["skill_id", "skill_name"]]
        .dropna()
        .drop_duplicates("skill_id")
        .set_index("skill_id")["skill_name"]
        .astype(str)
        .to_dict()
    )
    has_names = len(names) > 0 and sum(1 for v in names.values() if v.strip()) >= 0.9 * len(names)
    g = (
        raw.groupby("order_id", sort=False)
        .agg(
            skills=("skill_id", lambda s: tuple(sorted({int(x) for x in s if int(x) >= 0}))),
        )
        .reset_index()
    )
    g["item"] = g["skills"].map(lambda t: "_".join(map(str, t)) if t else "none")
    vocab = {k: i for i, k in enumerate(sorted(g["item"].unique()))}

    if has_names:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        skill_ids = sorted(names)
        skill_emb = model.encode([names[i] for i in skill_ids], show_progress_bar=False, normalize_embeddings=True)
        skill_vec = {sid: skill_emb[j] for j, sid in enumerate(skill_ids)}
        dim = skill_emb.shape[1]
        item_vecs = []
        item_ids = []
        for tok, idx in vocab.items():
            if tok == "none":
                vec = np.zeros(dim, dtype=np.float32)
            else:
                sids = [int(x) for x in tok.split("_")]
                mats = [skill_vec[s] for s in sids if s in skill_vec]
                vec = np.mean(mats, axis=0) if mats else np.zeros(dim, dtype=np.float32)
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vec = vec / nrm
            item_vecs.append(vec)
            item_ids.append(idx)
        X = np.vstack(item_vecs)
        labels, meta = kmeans_labels(X, TARGET_CLUSTERS, CLUSTER_SEED)
        item_to_cluster = {item_ids[i]: int(labels[i]) for i in range(len(item_ids))}
        return {
            "method": "text_skill_name_minilm_kmeans",
            "assist_has_skill_names": True,
            "n_atomic_named": len(names),
            "n_composite_items": len(vocab),
            "target_n_clusters": TARGET_CLUSTERS,
            **meta,
            "item_to_cluster": item_to_cluster,
            "fallback_used": False,
            "uses_eval_sequences": False,
        }

    # Fallback: train-split-only co-occurrence (should not hit)
    return None  # filled by caller with train-only


def train_only_cooc_clusters(train_seqs, n_items: int, seed: int):
    """SVD of item co-occurrence from TRAIN sequences only → k-means."""
    # symmetric co-occurrence within a window
    co = defaultdict(float)
    for seq in train_seqs:
        ids = [x for x, _ in seq if 0 <= x < n_items]
        for i, a in enumerate(ids):
            for b in ids[i + 1 : i + 6]:
                co[(a, b)] += 1.0
                co[(b, a)] += 1.0
    # build sparse-ish dense matrix via random projection of rows
    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1.0 / np.sqrt(32), size=(n_items, 32))
    # accumulate: for each edge contribute to row
    X = np.zeros((n_items, 32), dtype=np.float64)
    for (a, b), w in co.items():
        X[a] += w * proj[b]
    # items never seen → small noise
    for i in range(n_items):
        if not np.any(X[i]):
            X[i] = rng.normal(0, 1e-3, size=32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    labels, meta = kmeans_labels(X.astype(np.float32), TARGET_CLUSTERS, seed)
    return {i: int(labels[i]) for i in range(n_items)}, {
        "method": "train_only_cooccurrence_random_projection_kmeans",
        "target_n_clusters": TARGET_CLUSTERS,
        **meta,
        "uses_eval_sequences": False,
        "uses_only_train_split": True,
    }


def cluster_xes(train_seqs, n_items: int):
    """No question text in dump → train-only co-occurrence clustering."""
    mapping, meta = train_only_cooc_clusters(train_seqs, n_items, CLUSTER_SEED)
    meta.update(
        {
            "assist_has_skill_names": None,
            "xes_has_question_text": False,
            "fallback_used": True,
            "fallback_reason": "XES3G5M question_level CSVs expose question ids/KC tags only — no question stem text for MiniLM",
            "item_to_cluster": mapping,
        }
    )
    return meta


def normalize_grid_row(r: dict) -> dict:
    """Ensure final_val nesting for selection / winner_report compatibility."""
    if "final_val" in r:
        return r
    def pack(key_r5, key_n=None):
        return {
            "mean_r5": r.get(key_r5),
            "n_queries": r.get(key_n) if key_n else r.get("val_nonrepeat_n_queries"),
            "na_below_min_slice": r.get("val_nonrepeat_na"),
        }
    final = {
        "all": {"mean_r5": r.get("val_all_r5"), "n_queries": None, "na_below_min_slice": None},
        "revisit": {"mean_r5": r.get("val_revisit_r5"), "n_queries": None, "na_below_min_slice": None},
        "advance": {"mean_r5": r.get("val_advance_r5"), "n_queries": None, "na_below_min_slice": None},
        "nonrepeat": {
            "mean_r5": r.get("val_nonrepeat_r5"),
            "n_queries": r.get("val_nonrepeat_n_queries"),
            "na_below_min_slice": r.get("val_nonrepeat_na"),
        },
    }
    out = dict(r)
    out["final_val"] = final
    return out


def winner_from_row(row: dict, *, in_prereg: bool) -> dict:
    fv = row["final_val"]
    return {
        "config": row["config"],
        "config_id": row.get("config_id"),
        "best_epoch": row.get("best_epoch"),
        "val_ce_loss": row.get("best_val_ce_loss"),
        "val_nonrepeat_r5": fv["nonrepeat"]["mean_r5"],
        "val_all_r5": fv["all"]["mean_r5"],
        "val_revisit_r5": fv["revisit"]["mean_r5"],
        "val_advance_r5": fv["advance"]["mean_r5"],
        "val_slice_n_queries": {sl: fv[sl].get("n_queries") for sl in fv},
        "val_slice_na_below_min": {sl: fv[sl].get("na_below_min_slice") for sl in fv},
        "selection_metric": "val_nonrepeat_r5_then_val_ce_within_0.005",
        "selection_meta": row.get("selection_meta"),
        "in_preregistered_h_rev": in_prereg,
        "curve": row.get("curve", []),
        "val": fv,
    }


def run_gru_grid(name: str, train_seqs, val_seqs, n_items: int) -> list[dict]:
    rows = []
    max_r5_seqs = 400 if n_items >= 1000 or len(val_seqs) > 800 else None
    for ci, cfg in enumerate(GRID):
        cfg_id = f"lr{cfg['lr']}_emb{cfg['emb']}_do{cfg['dropout']}_maxep{cfg['max_epochs']}"
        print(f"  [{name} GRU {ci+1}/{len(GRID)}] {cfg_id}", flush=True)
        gru = GRUProbe(n_items, emb=cfg["emb"], hidden=max(64, cfg["emb"] * 2))
        gru, pack = train_seq_model_early_stop(
            gru, train_seqs, val_seqs, n_items, cfg=cfg, seed=REPLICATION_SEED
        )
        def score_fn(prefix, model=gru):
            return score_seq_model(model, prefix, n_items)

        pack["final_val"] = eval_r5_on_seqs(
            score_fn, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 9
        )
        rows.append({"config": cfg, "config_id": cfg_id, **pack})
    return rows


def freeze_unit_baselines(train_seqs, val_seqs, n_items: int):
    pop = popularity_scores(train_seqs, n_items)
    max_r5_seqs = 400 if n_items >= 1000 or len(val_seqs) > 800 else None

    def pop_fn(prefix):
        return pop.copy()

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

    trans = fit_markov(train_seqs, n_items)
    mk = markov_score_fn(trans, pop)
    baselines = {
        "popularity": {
            "parameter_free": True,
            "parameter_light": False,
            "val": eval_r5_on_seqs(pop_fn, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 3),
            "r5_val_max_seqs": max_r5_seqs,
        },
        "recency": {
            "parameter_free": True,
            "parameter_light": False,
            "val": eval_r5_on_seqs(recency_fn, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 4),
            "r5_val_max_seqs": max_r5_seqs,
        },
        "markov_order1": {
            "parameter_free": False,
            "parameter_light": True,
            "budget_note": (
                "First-order item→item successor counts on train only; no embedding/LR grid. "
                "Listed as parameter-light in the H-rev budget statement (not equal-grid)."
            ),
            "val": eval_r5_on_seqs(mk, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 5),
            "r5_val_max_seqs": max_r5_seqs,
        },
    }
    return baselines, pop


def main():
    t0 = time.time()
    prev_path = PHASE / "probe_freeze.json"
    prev = json.loads(prev_path.read_text())
    assert prev.get("n_grid_configs") == 24

    # --- load native sequences ---
    print("loading sequences…", flush=True)
    j_seqs, j_uids = build_learner_sequences_junyi_timed(
        max_users=JUNYI_TIMED_MAX_USERS, min_len=MIN_LEN, max_len=MAX_LEN
    )
    n_j = max(max(x for x, _ in s) for s in j_seqs) + 1
    a_seqs, a_uids, n_a = build_learner_sequences_assist(min_len=MIN_LEN, max_len=MAX_LEN)
    x_seqs, x_uids, n_x = build_learner_sequences_xes(min_len=MIN_LEN, max_len=MAX_LEN)

    datasets_native = {
        "junyi_timed": (j_seqs, j_uids, n_j, "exercise"),
        "assistments": (a_seqs, a_uids, n_a, "composite_skill_token"),
        "xes3g5m": (x_seqs, x_uids, n_x, "question"),
    }

    # --- clustering specs ---
    print("clustering…", flush=True)
    cluster_specs = {}
    # junyi
    cluster_specs["junyi_timed"] = cluster_junyi_timed()
    # assist
    assist_c = cluster_assistments()
    if assist_c is None:
        tr_i, _, _ = learner_split(a_uids, REPLICATION_SEED)
        train_a = [a_seqs[i] for i in tr_i]
        mapping, meta = train_only_cooc_clusters(train_a, n_a, CLUSTER_SEED)
        # sensitivity: shuffle labels null
        rng = np.random.default_rng(CLUSTER_SEED)
        shuffled = {i: int(x) for i, x in enumerate(rng.permutation([mapping[j] for j in range(n_a)]))}
        assist_c = {
            **meta,
            "assist_has_skill_names": False,
            "fallback_used": True,
            "fallback_reason": "skill_name unavailable",
            "item_to_cluster": mapping,
            "sensitivity_shuffled_cluster_labels": {
                "note": "Same k; labels permuted — expected to destroy structure",
                "item_to_cluster_shuffled_sha": hashlib.sha256(
                    json.dumps(sorted(shuffled.items())).encode()
                ).hexdigest()[:16],
            },
        }
    else:
        # sensitivity note even when text works: report n_named
        assist_c["sensitivity_note"] = (
            "Primary clustering uses skill_name MiniLM; no train-only fallback needed. "
            "Composite tokens = mean of member skill embeddings."
        )
    cluster_specs["assistments"] = assist_c

    # xes: train-only on train split
    tr_i, _, _ = learner_split(x_uids, REPLICATION_SEED)
    train_x = [x_seqs[i] for i in tr_i]
    cluster_specs["xes3g5m"] = cluster_xes(train_x, n_x)

    # persist cluster maps lightly (without huge dumps duplicated)
    cluster_dir = PHASE / "cluster_maps"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for ds, spec in cluster_specs.items():
        map_path = cluster_dir / f"{ds}_item_to_cluster.json"
        map_path.write_text(json.dumps(spec["item_to_cluster"]) + "\n")
        slim = {k: v for k, v in spec.items() if k != "item_to_cluster"}
        slim["item_to_cluster_path"] = str(map_path.relative_to(ROOT))
        slim["n_mapped_items"] = len(spec["item_to_cluster"])
        cluster_specs[ds] = {**slim, "item_to_cluster": spec["item_to_cluster"]}

    h_rev_policy = dict(H_REV_UNIT_POLICY)
    h_rev_policy["primary"] = {
        **H_REV_UNIT_POLICY["primary"],
        "status": "computed_for_freeze_schema_4",
        "target_n_clusters": TARGET_CLUSTERS,
        "clustering_seed": CLUSTER_SEED,
    }
    h_rev_policy["preregistered_h_rev_models"] = [
        "popularity",
        "recency",
        "markov_order1",
        "gru_probe",
    ]
    h_rev_policy["minimum_models_for_reversal_claim"] = {
        "required": ["popularity", "recency", "markov_order1", "gru_probe"],
        "optional_exploratory": ["attn_probe", "ragr_lite"],
        "rule": (
            "A cross-slice rank reversal is claimable only when the full required set is evaluated "
            "on the same learner-disjoint split and fixed sample indices; attn is not required."
        ),
    }
    h_rev_policy["budget_statement"] = {
        "equal_grid_trainable": ["gru_probe", "attn_probe (exploratory only)"],
        "parameter_free": ["popularity", "recency"],
        "parameter_light": ["markov_order1"],
        "note": "Markov is not given the 24-config grid; it fits transition counts on train only.",
    }

    out = {
        "schema_version": 4,
        "replication_seed": REPLICATION_SEED,
        "split_rule": SPLIT,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "min_slice_queries": MIN_SLICE_QUERIES,
        "grid_spec": GRID_SPEC,
        "n_grid_configs": len(GRID),
        "grid": GRID,
        "selection_metric": "val_nonrepeat_r5_then_lowest_ce_within_0.005",
        "selection_tolerance_r5": SELECTION_TOL,
        "h_rev_unit_policy": h_rev_policy,
        "attn_probe_architecture": ATTN_PROBE_ARCHITECTURE,
        "clustering": {
            ds: {k: v for k, v in spec.items() if k != "item_to_cluster"} for ds, spec in cluster_specs.items()
        },
        "prior_peeking_disclosure": prev.get("prior_peeking_disclosure"),
        "deviations_doc": "docs/DEVIATIONS.md",
        "assistments_filter_funnel": prev.get("assistments_filter_funnel"),
        "datasets": {},
    }

    partial = PHASE / "probe_freeze.partial.json"
    if partial.exists():
        try:
            prev_partial = json.loads(partial.read_text())
            if prev_partial.get("schema_version") == 4:
                out["datasets"] = prev_partial.get("datasets", {})
                print("resume datasets", list(out["datasets"]), flush=True)
        except Exception:
            pass

    for ds_name, (seqs, uids, n_items, unit) in datasets_native.items():
        if ds_name in out["datasets"] and out["datasets"][ds_name].get("cluster_unit", {}).get("winners"):
            print("skip complete", ds_name, flush=True)
            continue
        print(f"\n=== upgrade {ds_name} ===", flush=True)
        tr_i, va_i, te_i = learner_split(uids, REPLICATION_SEED)
        train_seqs = [seqs[i] for i in tr_i]
        val_seqs = [seqs[i] for i in va_i]
        test_uids = [uids[i] for i in te_i]

        # Native baselines + Markov
        baselines, pop = freeze_unit_baselines(train_seqs, val_seqs, n_items)

        # Re-select from previous grid results (GRU + attn)
        prev_ds = prev["datasets"][ds_name]
        native_winners = {}
        native_grid = {}
        for m, rows in prev_ds["grid_results"].items():
            norm = [normalize_grid_row(r) for r in rows]
            native_grid[m] = [
                {
                    "config": r["config"],
                    "config_id": r["config_id"],
                    "best_epoch": r.get("best_epoch"),
                    "best_val_ce_loss": r.get("best_val_ce_loss"),
                    "curve": r.get("curve", []),
                    "val_nonrepeat_r5": r["final_val"]["nonrepeat"]["mean_r5"],
                    "val_all_r5": r["final_val"]["all"]["mean_r5"],
                    "val_revisit_r5": r["final_val"]["revisit"]["mean_r5"],
                    "val_advance_r5": r["final_val"]["advance"]["mean_r5"],
                    "val_nonrepeat_n_queries": r["final_val"]["nonrepeat"].get("n_queries"),
                    "val_nonrepeat_na": r["final_val"]["nonrepeat"].get("na_below_min_slice"),
                    "final_val": r["final_val"],
                }
                for r in norm
            ]
            # re-select with CE tie-break (skip ragr_lite CE-less: keep max R@5)
            if m == "ragr_lite":
                best = max(norm, key=lambda r: (r["final_val"]["nonrepeat"]["mean_r5"] is not None, r["final_val"]["nonrepeat"]["mean_r5"] or -1))
                best["selection_meta"] = {"rule": "max val_nonrepeat_r5 (no CE for ragr_lite)"}
            else:
                best = select_winner(norm)
            native_winners[m] = winner_from_row(best, in_prereg=(m == "gru_probe"))

        # query counts from baselines for reporting
        val_query_counts = {
            "baselines": {b: baselines[b]["val"] for b in baselines},
            "note": "Each slice reports n_queries; < min_slice_queries → N/A",
        }

        # Cluster sequences
        imap = cluster_specs[ds_name]["item_to_cluster"]
        # keys may be str if loaded — ensure int
        imap = {int(k): int(v) for k, v in imap.items()}
        c_train = map_seqs_to_clusters(train_seqs, imap)
        c_val = map_seqs_to_clusters(val_seqs, imap)
        c_train, n_c = dense_remap(c_train)
        # val remap must use same vocab — rebuild via full map then dense on train vocab only
        # Fix: dense_remap on train+val together keeping alignment
        # redo properly:
        c_train_raw = map_seqs_to_clusters(train_seqs, imap)
        c_val_raw = map_seqs_to_clusters(val_seqs, imap)
        vocab = {}
        def remap(seq):
            out = []
            for it, y in seq:
                if it not in vocab:
                    vocab[it] = len(vocab)
                out.append((vocab[it], y))
            return out
        c_train = [remap(s) for s in c_train_raw]
        c_val = [remap(s) for s in c_val_raw]
        n_c = len(vocab)

        print(f"  cluster n_items={n_c} n_train={len(c_train)} n_val={len(c_val)}", flush=True)
        c_baselines, _ = freeze_unit_baselines(c_train, c_val, n_c)
        c_grid = run_gru_grid(f"{ds_name}/cluster", c_train, c_val, n_c)
        c_norm = [normalize_grid_row(r) for r in c_grid]
        c_best = select_winner(c_norm)
        cluster_winner = winner_from_row(c_best, in_prereg=True)

        payload = {
            "dataset": ds_name,
            "ranking_unit_native": unit,
            "ranking_unit": unit,
            "n_items": n_items,
            "n_sequences": len(seqs),
            "min_slice_queries": MIN_SLICE_QUERIES,
            "split": {
                "seed": REPLICATION_SEED,
                "fractions": SPLIT,
                "n_train": int(len(tr_i)),
                "n_val": int(len(va_i)),
                "n_test": int(len(te_i)),
                "train_uid_hash": sha_list(sorted(int(uids[i]) for i in tr_i)),
                "val_uid_hash": sha_list(sorted(int(uids[i]) for i in va_i)),
                "test_uid_hash": sha_list(sorted(int(u) for u in test_uids)),
                "test_touched": False,
            },
            "parameter_free_baselines": {
                "popularity": baselines["popularity"],
                "recency": baselines["recency"],
            },
            "parameter_light_baselines": {"markov_order1": baselines["markov_order1"]},
            "equal_tuning_budget": {
                "applies_to": ["gru_probe", "attn_probe"],
                "grid_spec": GRID_SPEC,
                "n_configs": len(GRID),
                "validation_metric_for_selection": (
                    f"val non-repeat R@5; among configs within {SELECTION_TOL} of best, lowest val CE"
                ),
                "early_stopping": "val_ce_loss with patience",
                "parameter_light_not_in_equal_grid": ["markov_order1"],
            },
            "grid_results": native_grid,
            "winners": native_winners,
            "winner_metrics_note": (
                "Winners report val_nonrepeat_r5 and val_ce_loss; selection uses R@5 then CE within 0.005."
            ),
            "val_query_counts": val_query_counts,
            "stop_rule": {
                "rule": "if best GRU nonrepeat val R@5 <= popularity nonrepeat val R@5, stop tuning",
                "popularity_nonrepeat_r5": baselines["popularity"]["val"]["nonrepeat"]["mean_r5"],
                "markov_nonrepeat_r5": baselines["markov_order1"]["val"]["nonrepeat"]["mean_r5"],
                "best_gru_nonrepeat_r5": native_winners["gru_probe"]["val_nonrepeat_r5"],
                "stop_tuning": (
                    native_winners["gru_probe"]["val_nonrepeat_r5"] or -1
                )
                <= (baselines["popularity"]["val"]["nonrepeat"]["mean_r5"] or 0),
            },
            "native_unit": {
                "ranking_unit": unit,
                "winners": native_winners,
                "baselines": {
                    "popularity": baselines["popularity"]["val"],
                    "recency": baselines["recency"]["val"],
                    "markov_order1": baselines["markov_order1"]["val"],
                },
            },
            "cluster_unit": {
                "ranking_unit": f"matched_concept_clusters~{TARGET_CLUSTERS}",
                "n_clusters_used": n_c,
                "clustering_method": cluster_specs[ds_name].get("method"),
                "baselines": {
                    "popularity": c_baselines["popularity"]["val"],
                    "recency": c_baselines["recency"]["val"],
                    "markov_order1": c_baselines["markov_order1"]["val"],
                },
                "grid_results": {
                    "gru_probe": [
                        {
                            "config": r["config"],
                            "config_id": r["config_id"],
                            "best_epoch": r.get("best_epoch"),
                            "best_val_ce_loss": r.get("best_val_ce_loss"),
                            "val_nonrepeat_r5": r["final_val"]["nonrepeat"]["mean_r5"],
                            "val_nonrepeat_n_queries": r["final_val"]["nonrepeat"].get("n_queries"),
                            "val_all_r5": r["final_val"]["all"]["mean_r5"],
                            "val_revisit_r5": r["final_val"]["revisit"]["mean_r5"],
                            "val_advance_r5": r["final_val"]["advance"]["mean_r5"],
                        }
                        for r in c_norm
                    ]
                },
                "winners": {"gru_probe": cluster_winner},
                "role": "primary H-rev unit",
            },
            "display_names": {
                "gru_probe": "GRU-based next-item probe",
                "attn_probe": "attention-based next-item probe (exploratory; excluded from H-rev)",
                "ragr_lite": "RAGR-lite (Junyi budget-matched)",
                "popularity": "popularity (parameter-free)",
                "recency": "recency (parameter-free)",
                "markov_order1": "first-order Markov successor (parameter-light)",
            },
        }
        if ds_name == "junyi_timed":
            payload["learner_selection"] = prev_ds.get("learner_selection")
        if ds_name == "xes3g5m":
            qo = xes_question_order_structure()
            payload["question_order_structure"] = qo
            payload["curriculum_order_structured"] = True
            payload["native_unit_role"] = "descriptive_only"
            payload["curriculum_order_note"] = (
                f"Question-level H-rev is curriculum-order-structured: empirical mode successor share≈"
                f"{qo['share_next_is_empirical_mode_successor_of_prev']:.3f}; "
                f"global-order successor≈{qo['share_next_is_global_order_successor']:.3f}. "
                "Native-unit contrasts are descriptive; primary confirmatory unit is cluster-matched."
            )

        out["datasets"][ds_name] = payload
        partial.write_text(json.dumps(out, indent=2) + "\n")
        print("checkpointed", ds_name, flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    write_json("probe_freeze.json", out)
    if partial.exists():
        partial.unlink()
    print("done schema4", out["elapsed_sec"], "s", flush=True)


if __name__ == "__main__":
    main()
