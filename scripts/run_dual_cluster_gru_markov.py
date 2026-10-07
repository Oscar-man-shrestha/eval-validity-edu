#!/usr/bin/env python3
"""GRU−Markov on BOTH cluster maps (text + co-occurrence) for Junyi and ASSISTments.

Uses freeze train/val (seed 20261004), frozen cluster-unit winner hyperparameters,
and fixed best_epoch epochs (no early stopping). Writes dual_cluster_gru_markov.json.

XES has one method only — not scored here (see prereg: amended one-method rule).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from junyi_pipeline import OUT
from run_followup3 import (
    GRUProbe,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    score_seq_model,
)
from run_preprereg_freeze_v4 import fit_markov, map_seqs_to_clusters, markov_score_fn
from run_probe_freeze import REPLICATION_SEED, learner_split
from scripts.gru_markov_paired_ci import paired_ci, per_seq_r5, remap_pair
from scripts.recompute_ci_verdicts import ci_verdict

torch.set_num_threads(1)
PHASE = OUT / "phases"
T_R5 = 0.01
N_BOOT = 400


def train_fixed_epochs(n_items, train_seqs, cfg, best_epoch, seed):
    """Train exactly best_epoch epochs — no early stopping (replication protocol)."""
    model = GRUProbe(n_items, emb=cfg["emb"], hidden=max(64, cfg["emb"] * 2))
    model.to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    # windows
    pairs = []
    max_len = 50
    for seq in train_seqs:
        ids = [x + 1 for x, _ in seq]
        for t in range(1, len(ids)):
            pairs.append((ids[max(0, t - max_len) : t], seq[t][0]))
    rng.shuffle(pairs)
    pairs = pairs[:80_000]
    model.train()
    for _ep in range(int(best_epoch)):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), 64):
            bp = pairs[i : i + 64]
            if not bp:
                continue
            max_t = max(len(c) for c, _ in bp)
            x = torch.zeros(len(bp), max_t, dtype=torch.long)
            y = torch.tensor([lab for _, lab in bp], dtype=torch.long)
            for bi, (ctx, _) in enumerate(bp):
                x[bi, -len(ctx) :] = torch.tensor(ctx, dtype=torch.long)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model


def load_map(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text())
    return {int(k): int(v) for k, v in raw.items()}


def run_one(name, seqs, uids, map_path, cfg, best_epoch, dual_ari):
    item_to_c = load_map(map_path)
    tr_i, va_i, te_i = learner_split(uids, REPLICATION_SEED)
    assert set(int(uids[i]) for i in va_i).isdisjoint(set(int(uids[i]) for i in te_i))
    train_fit_raw = [seqs[i] for i in tr_i]
    val_raw = [seqs[i] for i in va_i]
    train_fit_c = map_seqs_to_clusters(train_fit_raw, item_to_c)
    val_c = map_seqs_to_clusters(val_raw, item_to_c)
    tr_fit, va, n_items = remap_pair(train_fit_c, val_c)
    pop = np.zeros(n_items)
    for seq in tr_fit:
        for it, _ in seq:
            if 0 <= it < n_items:
                pop[it] += 1.0
    mk_mat = fit_markov(tr_fit, n_items)
    mk_fn = markov_score_fn(mk_mat, pop)
    gru = train_fixed_epochs(n_items, tr_fit, cfg, best_epoch, seed=REPLICATION_SEED)
    max_seqs = 400 if n_items >= 100 or len(va) > 800 else None

    def gru_fn(prefix, model=gru):
        return score_seq_model(model, prefix, n_items)

    g = per_seq_r5(gru_fn, va, n_items, max_seqs=max_seqs, seed=REPLICATION_SEED)
    m = per_seq_r5(mk_fn, va, n_items, max_seqs=max_seqs, seed=REPLICATION_SEED)
    nr = paired_ci(g["nonrepeat"], m["nonrepeat"], n_boot=N_BOOT, seed=REPLICATION_SEED + 7)
    verdict = ci_verdict(nr.get("ci"), T_R5)["verdict"]
    return {
        "map_path": str(map_path.relative_to(ROOT)),
        "n_items_clustered": n_items,
        "best_epoch_fixed": int(best_epoch),
        "config": cfg,
        "ari_vs_other_method": dual_ari,
        "nonrepeat": {**nr, "verdict": verdict, "t": T_R5},
        "val_query_nonrepeat": g["n_queries_nonrepeat"],
        "test_learners_used_in_fit_or_selection": False,
        "scored_split": "freeze_val_only",
        "n_test_held_out": len(te_i),
    }


def main():
    t0 = time.time()
    fr = json.loads((PHASE / "probe_freeze.json").read_text())
    dual = json.loads((PHASE / "dual_clustering.json").read_text())
    out = {
        "schema": "dual_cluster_gru_markov_v1",
        "seed": REPLICATION_SEED,
        "protocol": "fixed best_epoch; train on freeze train; score freeze val; no test touch",
        "datasets": {},
    }

    builders = {
        "junyi_timed": lambda: build_learner_sequences_junyi_timed(
            max_users=6000, min_len=12, max_len=200
        )[:2],
        "assistments": lambda: build_learner_sequences_assist()[:2],
    }
    map_keys = {
        "junyi_timed": ("text", "cooccurrence"),
        "assistments": ("text", "cooccurrence"),
    }

    for ds, builder in builders.items():
        print("===", ds, flush=True)
        seqs, uids = builder()
        cw = fr["datasets"][ds]["cluster_unit"]["winners"]["gru_probe"]
        cfg = dict(cw["config"])
        best_ep = int(cw["best_epoch"])
        dinfo = dual["datasets"][ds]
        ari = (dinfo.get("method_agreement") or dinfo.get("agreement") or {}).get("ari")
        results = {}
        for method in map_keys[ds]:
            mpath = ROOT / dinfo[method]["map_path"]
            print(f"  {method} {mpath.name} ep={best_ep}", flush=True)
            results[method] = run_one(ds, seqs, uids, mpath, cfg, best_ep, ari)
        vt = results["text"]["nonrepeat"]["verdict"]
        vc = results["cooccurrence"]["nonrepeat"]["verdict"]
        agree = vt == vc and vt.startswith("supported_")
        cluster_registered = {
            "text_verdict": vt,
            "cooccurrence_verdict": vc,
            "same_supported_verdict": agree,
            "registered_cluster_verdict": (
                vt if agree else "inconclusive"
            ),
            "rule": (
                "cluster-unit supported_* only if text and co-occurrence give the same supported_* verdict; "
                "otherwise inconclusive"
            ),
            "ari": ari,
        }
        out["datasets"][ds] = {
            "by_method": results,
            "cluster_unit_agreement": cluster_registered,
        }
        print("  agreement", cluster_registered, flush=True)

    out["xes3g5m"] = {
        "cluster_methods": 1,
        "amendment": (
            "XES has no text clustering locally. Under the amended one-method rule, "
            "the co-occurrence map is the registered cluster unit; two-method agreement is N/A; "
            "ARI vs text is N/A. XES cluster-unit GRU−Markov remains in the confirmatory family of 6 "
            "under this amendment (see docs/PREREGISTRATION.md)."
        ),
        "ari": None,
    }
    out["elapsed_sec"] = round(time.time() - t0, 1)
    path = PHASE / "dual_cluster_gru_markov.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, "in", out["elapsed_sec"], "s", flush=True)


if __name__ == "__main__":
    main()
