#!/usr/bin/env python3
"""Paired sequence-bootstrap CIs: GRU winner vs Markov order-1, per dataset/unit.

Retrains the frozen winner GRU config (same seed / early-stop) and scores the
same val sequences / cuts as Markov. Also runs an XES-native longer-epoch
diagnostic (max_epochs=30, patience=5) to test undertraining.

Does not write a preregistration.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Set before numpy/torch so MiniBatchKMeans and the GRU do not double-load OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_followup3 import (
    GRUProbe,
    action_slice,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    popularity_scores,
    score_seq_model,
)
from run_preprereg_freeze_v4 import (
    fit_markov,
    map_seqs_to_clusters,
    markov_score_fn,
)
from run_probe_freeze import (
    EVAL_POS,
    JUNYI_TIMED_MAX_USERS,
    K,
    MAX_LEN,
    MIN_LEN,
    MIN_SLICE_QUERIES,
    PHASE,
    REPLICATION_SEED,
    learner_split,
    recall_at,
    train_seq_model_early_stop,
)

torch.set_num_threads(1)
N_BOOT = 400


def per_seq_r5(score_fn, seqs, n_items, max_seqs=None, seed=0):
    """Return per-sequence mean R@5 for all / nonrepeat (aligned sequence index)."""
    if max_seqs is not None and len(seqs) > max_seqs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(seqs), size=max_seqs, replace=False)
        seqs = [seqs[int(i)] for i in idx]
    all_s, nr_s = [], []
    n_q_all = n_q_nr = 0
    for seq in seqs:
        if len(seq) < 10:
            continue
        cuts = np.linspace(8, len(seq) - 1, num=min(EVAL_POS, max(1, len(seq) - 9)), dtype=int)
        hits_all, hits_nr = [], []
        for cut in cuts:
            cut = int(cut)
            label = seq[cut][0]
            prefix = [x for x, _ in seq[:cut]]
            last = prefix[-1]
            hist = set(prefix)
            sl = action_slice(label, hist, last)
            scores = score_fn(prefix)
            hit = recall_at(scores, label, K)
            hits_all.append(hit)
            n_q_all += 1
            if sl in ("revisit", "advance"):
                hits_nr.append(hit)
                n_q_nr += 1
        if hits_all:
            all_s.append(float(np.mean(hits_all)))
        if hits_nr:
            nr_s.append(float(np.mean(hits_nr)))
        else:
            nr_s.append(np.nan)
    return {
        "all": np.asarray(all_s, dtype=np.float64),
        "nonrepeat": np.asarray(nr_s, dtype=np.float64),
        "n_queries_all": n_q_all,
        "n_queries_nonrepeat": n_q_nr,
        "n_sequences": len(all_s),
        "max_seqs": max_seqs,
    }


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=0) -> dict:
    """a − b on sequences that have a finite value in both."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return {"n_sequences": int(len(a)), "mean_delta": None, "ci": None}
    delta = a - b
    rng = np.random.default_rng(seed)
    boots = []
    n = len(delta)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(delta[idx].mean()))
    arr = np.asarray(boots)
    return {
        "n_sequences": int(n),
        "mean_gru": float(a.mean()),
        "mean_markov": float(b.mean()),
        "mean_delta_gru_minus_markov": float(delta.mean()),
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "ci_excludes_zero": bool(np.quantile(arr, 0.025) > 0 or np.quantile(arr, 0.975) < 0),
        "gru_beats_markov": bool(delta.mean() > 0 and np.quantile(arr, 0.025) > 0),
        "markov_beats_gru": bool(delta.mean() < 0 and np.quantile(arr, 0.975) < 0),
        "n_boot": n_boot,
        "unit": "sequence",
    }


def remap_pair(train_raw, val_raw):
    vocab = {}

    def remap(seq):
        out = []
        for it, y in seq:
            if it not in vocab:
                vocab[it] = len(vocab)
            out.append((vocab[it], y))
        return out

    return [remap(s) for s in train_raw], [remap(s) for s in val_raw], len(vocab)


def train_gru(n_items, train_seqs, val_seqs, cfg, seed):
    gru = GRUProbe(n_items, emb=cfg["emb"], hidden=max(64, cfg["emb"] * 2))
    gru, pack = train_seq_model_early_stop(
        gru, train_seqs, val_seqs, n_items, cfg=cfg, seed=seed
    )
    return gru, pack


def score_pair(gru, mk, pop, val_seqs, n_items, max_seqs, seed):
    def gru_fn(prefix, model=gru):
        return score_seq_model(model, prefix, n_items)

    g = per_seq_r5(gru_fn, val_seqs, n_items, max_seqs=max_seqs, seed=seed)
    m = per_seq_r5(mk, val_seqs, n_items, max_seqs=max_seqs, seed=seed)
    return {
        "val_query_counts": {
            "all": g["n_queries_all"],
            "nonrepeat": g["n_queries_nonrepeat"],
            "n_sequences": g["n_sequences"],
            "min_slice_queries": MIN_SLICE_QUERIES,
            "nonrepeat_na_below_min": g["n_queries_nonrepeat"] < MIN_SLICE_QUERIES,
        },
        "all": paired_ci(g["all"], m["all"], seed=seed + 3),
        "nonrepeat": paired_ci(g["nonrepeat"], m["nonrepeat"], seed=seed + 5),
    }


def main() -> None:
    t0 = time.time()
    freeze = json.loads((PHASE / "probe_freeze.json").read_text())
    print("loading sequences…", flush=True)
    j_seqs, j_uids = build_learner_sequences_junyi_timed(
        max_users=JUNYI_TIMED_MAX_USERS, min_len=MIN_LEN, max_len=MAX_LEN
    )
    n_j = max(max(x for x, _ in s) for s in j_seqs) + 1
    a_seqs, a_uids, n_a = build_learner_sequences_assist(min_len=MIN_LEN, max_len=MAX_LEN)
    x_seqs, x_uids, n_x = build_learner_sequences_xes(min_len=MIN_LEN, max_len=MAX_LEN)

    native = {
        "junyi_timed": (j_seqs, j_uids, n_j),
        "assistments": (a_seqs, a_uids, n_a),
        "xes3g5m": (x_seqs, x_uids, n_x),
    }
    # Freeze primary maps (same assignment as probe_freeze.json). Do not refit
    # MiniLM/KMeans here: a second OpenMP load after torch segfaults (exit 139),
    # and a refit would not be the frozen unit.
    print("loading freeze cluster maps…", flush=True)
    map_dir = PHASE / "cluster_maps"
    maps = {}
    for ds_name, fname in (
        ("junyi_timed", "junyi_timed_item_to_cluster.json"),
        ("assistments", "assistments_item_to_cluster.json"),
        ("xes3g5m", "xes3g5m_item_to_cluster.json"),
    ):
        raw = json.loads((map_dir / fname).read_text())
        maps[ds_name] = {int(k): int(v) for k, v in raw.items()}

    results = {}
    xes_long = None
    for ds_name, (seqs, uids, n_items) in native.items():
        print(f"\n=== {ds_name} ===", flush=True)
        tr_i, va_i, _ = learner_split(uids, REPLICATION_SEED)
        train_seqs = [seqs[i] for i in tr_i]
        val_seqs = [seqs[i] for i in va_i]
        pop = popularity_scores(train_seqs, n_items)
        trans = fit_markov(train_seqs, n_items)
        mk = markov_score_fn(trans, pop)
        max_r5 = 400 if n_items >= 1000 or len(val_seqs) > 800 else None

        ds_fr = freeze["datasets"][ds_name]
        gru_cfg = ds_fr["native_unit"]["winners"]["gru_probe"]["config"]
        print(f"  native GRU {gru_cfg} n_items={n_items}", flush=True)
        gru, pack = train_gru(n_items, train_seqs, val_seqs, gru_cfg, REPLICATION_SEED)
        native_pair = score_pair(gru, mk, pop, val_seqs, n_items, max_r5, REPLICATION_SEED + 9)
        native_pair["gru_train"] = {
            "best_epoch": pack.get("best_epoch"),
            "best_val_ce_loss": pack.get("best_val_ce_loss"),
            "max_epochs": gru_cfg.get("max_epochs"),
            "early_stopping_patience": gru_cfg.get("early_stopping_patience"),
            "n_items": n_items,
            "dropout_in_config_but_unused_by_GRUProbe": gru_cfg.get("dropout"),
        }

        imap = maps[ds_name]
        c_train, c_val, n_c = remap_pair(
            map_seqs_to_clusters(train_seqs, imap),
            map_seqs_to_clusters(val_seqs, imap),
        )
        c_pop = popularity_scores(c_train, n_c)
        c_mk = markov_score_fn(fit_markov(c_train, n_c), c_pop)
        c_cfg = ds_fr["cluster_unit"]["winners"]["gru_probe"]["config"]
        c_max = 400 if n_c >= 1000 or len(c_val) > 800 else None
        print(f"  cluster GRU {c_cfg} n_clusters={n_c}", flush=True)
        c_gru, c_pack = train_gru(n_c, c_train, c_val, c_cfg, REPLICATION_SEED)
        cluster_pair = score_pair(c_gru, c_mk, c_pop, c_val, n_c, c_max, REPLICATION_SEED + 9)
        cluster_pair["gru_train"] = {
            "best_epoch": c_pack.get("best_epoch"),
            "best_val_ce_loss": c_pack.get("best_val_ce_loss"),
            "max_epochs": c_cfg.get("max_epochs"),
            "n_clusters_used": n_c,
        }
        cluster_pair["n_clusters_used"] = n_c

        results[ds_name] = {
            "n_items_native": n_items,
            "native": native_pair,
            "cluster": cluster_pair,
        }

        if ds_name == "xes3g5m":
            long_cfg = dict(gru_cfg)
            long_cfg["max_epochs"] = 30
            long_cfg["early_stopping_patience"] = 5
            print(f"  XES native longer diagnostic {long_cfg}", flush=True)
            long_gru, long_pack = train_gru(n_items, train_seqs, val_seqs, long_cfg, REPLICATION_SEED)
            long_pair = score_pair(long_gru, mk, pop, val_seqs, n_items, max_r5, REPLICATION_SEED + 9)
            xes_long = {
                "config": long_cfg,
                "best_epoch": long_pack.get("best_epoch"),
                "best_val_ce_loss": long_pack.get("best_val_ce_loss"),
                "stopped_early": long_pack.get("stopped_early"),
                "curve": long_pack.get("curve"),
                "paired_vs_markov": long_pair,
                "interpretation_if_still_loses": (
                    "If GRU still loses to Markov after 30 epochs / patience 5, "
                    "undertraining at max_epochs=10 is not a sufficient explanation; "
                    "the 7,439-way catalog is curriculum-order-structured and a 1-step "
                    "table is the right null."
                ),
            }

    h_rev_note = {
        "what_losing_to_markov_means": (
            "H-rev asks whether a probe's ranking vs baselines reverses when the action "
            "unit changes (native vs matched clusters). Markov order-1 is the "
            "parameter-light 1-step successor table. If GRU loses to Markov at a unit, "
            "that unit's GRU 'win' vs popularity/recency is not a learned-structure win: "
            "a count table already captures the regularity (sticky or curriculum-order). "
            "A confirmatory H-rev claim at that unit cannot treat GRU as beating the "
            "required baseline set. Native XES (7,439 questions, mode successor ≈0.35, "
            "GRU 0.614 vs Markov 0.739 at freeze) is therefore descriptive. The "
            "informative contrast is the cluster unit, where GRU can still beat Markov."
        ),
        "xes_native_undertraining_checks": {
            "n_classes": 7439,
            "freeze_max_epochs": 10,
            "freeze_patience": 2,
            "freeze_best_epoch": freeze["datasets"]["xes3g5m"]["native_unit"]["winners"]["gru_probe"].get(
                "best_epoch"
            ),
            "GRUProbe_ignores_dropout_hyperparam": True,
            "val_r5_subsample": 400,
        },
    }

    out = {
        "schema": "gru_markov_paired_ci_v1",
        "replication_seed": REPLICATION_SEED,
        "n_boot": N_BOOT,
        "slice": "validation sequences; same max_seqs cap as freeze",
        "datasets": results,
        "xes_native_longer_epoch_diagnostic": xes_long,
        "h_rev_implication": h_rev_note,
        "elapsed_sec": time.time() - t0,
    }
    dest = PHASE / "gru_markov_paired_ci.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest, "elapsed", out["elapsed_sec"], flush=True)


if __name__ == "__main__":
    main()
