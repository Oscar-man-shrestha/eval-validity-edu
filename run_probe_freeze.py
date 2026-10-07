#!/usr/bin/env python3
"""STEP 2 (checkpoint redo): equal-budget freeze on VALIDATION only.

Prints full grid definition, per-epoch validation curves, ASSISTments skill
construction, and never evaluates on test (hashes test uids only).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_followup3 import (
    AttnProbe,
    GRUProbe,
    action_slice,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    popularity_scores,
    score_seq_model,
)

ROOT = Path(__file__).resolve().parent
PHASE = ROOT / "outputs_junyi" / "phases"
PHASE.mkdir(parents=True, exist_ok=True)
ASSIST_CSV = ROOT / "data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv"

REPLICATION_SEED = 20261004
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_LEN, MAX_LEN = 12, 200
MAX_WINDOWS = 20_000
EVAL_POS = 5
K = 5
MIN_SLICE_QUERIES = 500  # below this, slice test is N/A at replication
JUNYI_TIMED_MAX_USERS = 6000

# Equal tuning budget for ALL trainable models (same Cartesian product)
# Checkpoint-2 extension: +emb 256, +lr 0.002 (once), then accept winners.
GRID_SPEC = {
    "lr": [5e-4, 1e-3, 2e-3],
    "emb": [32, 64, 128, 256],
    "dropout": [0.1, 0.3],
    "max_epochs": 10,
    "early_stopping_patience": 2,
    "early_stopping_metric": "val_ce_loss",
    "epoch_candidates_for_logging": [2, 5, 10],
    "selection_metric": "val_nonrepeat_r5_at_early_stopped_checkpoint",
    "n_configs": 24,
    "note": (
        "Cartesian lr×emb×dropout = 24 configs; identical for every trainable model "
        "(GRUProbe, AttnProbe) per dataset. RAGR-lite is not part of this equal-budget "
        "trainable grid. GRUProbe has no dropout module (dropout grid is a no-op for GRU). "
        "max_epochs=10, patience=2 (see docs/DEVIATIONS.md)."
    ),
    "extension_from_prior_freeze": "Added emb=256 and lr=0.002 once after emb=128 edge winners",
}

ATTN_PROBE_ARCHITECTURE = {
    "display_name": "attention-based next-item probe",
    "class": "AttnProbe",
    "not_a_published_baseline": True,
    "library_validation": "none — not checked against SASRec / RecBole / authors' code",
    "preregistered_h_rev_family": "excluded",
    "exclusion_reason": (
        "Unvalidated local probe; do not treat as SASRec. Documented for exploratory comparison only."
    ),
    "item_embedding": "nn.Embedding(n_items+1, emb, padding_idx=0)",
    "positional_encoding": "learned nn.Embedding(max_len, emb); added to token embeddings",
    "max_len": 50,
    "encoder": "torch.nn.TransformerEncoder",
    "n_layers": 1,
    "layer_type": "TransformerEncoderLayer",
    "n_heads": 2,
    "dim_feedforward": 64,
    "dropout": "from grid",
    "causal_mask": "triu boolean mask (diagonal=1) passed to TransformerEncoder",
    "prediction_head": "Linear(emb, n_items) on last position; embeddings not tied",
    "loss": "cross_entropy on next-item id",
}

H_REV_UNIT_POLICY = {
    "primary": {
        "unit": "matched_concept_clusters",
        "target_n_clusters": 120,
        "clustering": "preregistered; train/text-only (never eval sequences); two methods with agreement rule",
        "status": "not_yet_computed — freeze tunes on native units; cluster eval at replication",
    },
    "secondary": {
        "unit": "native",
        "junyi_timed": "exercise",
        "assistments": "composite_skill_token",
        "xes3g5m": "question",
    },
    "cross_dataset_native_contrasts": "descriptive_only_if_not_cluster_matched",
    "freeze_tuning_unit": "native",
    "preregistered_h_rev_models": ["popularity", "recency", "gru_probe"],
    "excluded_from_preregistered_h_rev": ["attn_probe", "ragr_lite"],
    "excluded_notes": {
        "attn_probe": "unvalidated architecture — exploratory only",
        "ragr_lite": "N/A on ASSIST/XES; not full DAG RAGR",
    },
}


def expand_grid(spec: dict) -> list[dict]:
    keys = ["lr", "emb", "dropout"]
    configs = []
    for vals in itertools.product(*[spec[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        cfg["max_epochs"] = spec["max_epochs"]
        cfg["early_stopping_patience"] = spec["early_stopping_patience"]
        configs.append(cfg)
    return configs


GRID = expand_grid(GRID_SPEC)
DEVICE = torch.device("cpu")


def write_json(name: str, obj: dict) -> None:
    path = PHASE / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    print("wrote", path, flush=True)


def sha_list(xs) -> str:
    payload = json.dumps([int(x) for x in xs], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def learner_split(uids: list[int], seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uids))
    n = len(order)
    n_train = int(SPLIT["train"] * n)
    n_val = int(SPLIT["val"] * n)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def build_windows(seqs, max_len=50, seed=0, limit=MAX_WINDOWS):
    pairs = []
    for seq in seqs:
        ids = [x + 1 for x, _ in seq]
        for t in range(1, len(ids)):
            pairs.append((ids[max(0, t - max_len) : t], seq[t][0]))
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    return pairs[:limit]


def batch_ce(model, pairs, batch=256) -> float:
    if not pairs:
        return float("nan")
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            bp = pairs[i : i + batch]
            max_t = max(len(c) for c, _ in bp)
            x = torch.zeros(len(bp), max_t, dtype=torch.long, device=DEVICE)
            y = torch.tensor([lab for _, lab in bp], dtype=torch.long, device=DEVICE)
            for bi, (ctx, _) in enumerate(bp):
                x[bi, -len(ctx) :] = torch.tensor(ctx, dtype=torch.long)
            total += float(loss_fn(model(x), y).item())
            n += len(bp)
    return total / max(n, 1)


def train_seq_model_early_stop(model, train_seqs, val_seqs, n_items, *, cfg, seed=0):
    """Train with early stopping on val CE; return model + curve."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    train_pairs = build_windows(train_seqs, seed=seed)
    val_pairs = build_windows(val_seqs, seed=seed + 7, limit=min(10_000, MAX_WINDOWS))
    curve = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = cfg["early_stopping_patience"]
    max_ep = cfg["max_epochs"]

    for ep in range(1, max_ep + 1):
        model.train()
        rng = np.random.default_rng(seed + ep)
        order = np.arange(len(train_pairs))
        rng.shuffle(order)
        for i in range(0, len(order), 64):
            idx = order[i : i + 64]
            bp = [train_pairs[j] for j in idx]
            if not bp:
                continue
            max_t = max(len(c) for c, _ in bp)
            x = torch.zeros(len(bp), max_t, dtype=torch.long, device=DEVICE)
            y = torch.tensor([lab for _, lab in bp], dtype=torch.long, device=DEVICE)
            for bi, (ctx, _) in enumerate(bp):
                x[bi, -len(ctx) :] = torch.tensor(ctx, dtype=torch.long)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        val_ce = batch_ce(model, val_pairs)
        # Curve logs val CE every epoch (selection uses early-stopped CE).
        # R@5 is computed once after training (too expensive on large catalogs mid-loop).
        point = {"epoch": ep, "val_ce_loss": val_ce}
        curve.append(point)
        print(f"      ep {ep}: val_ce={val_ce:.4f}", flush=True)

        if val_ce < best_val - 1e-4:
            best_val = val_ce
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # R@5 once at best checkpoint; subsample val for large catalogs (documented)
    max_r5_seqs = 400 if n_items >= 1000 or len(val_seqs) > 800 else None
    final_r5 = eval_r5_on_seqs(
        lambda p, m=model: score_seq_model(m, p, n_items),
        val_seqs,
        n_items,
        max_seqs=max_r5_seqs,
        seed=seed + 99,
    )
    # attach CE at logged candidate epochs for the curve summary
    for pt in curve:
        if pt["epoch"] in GRID_SPEC["epoch_candidates_for_logging"]:
            pt["r5_note"] = "R@5 computed only at early-stopped checkpoint, not mid-training"
    return model, {
        "curve": curve,
        "best_epoch": best_epoch,
        "best_val_ce_loss": best_val,
        "final_val": final_r5,
        "r5_val_max_seqs": max_r5_seqs,
        "stopped_early": best_epoch < max_ep and len(curve) < max_ep,
    }


def recall_at(scores: np.ndarray, label: int, k: int) -> float:
    if label < 0 or label >= len(scores):
        return 0.0
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return float(label in set(top.tolist()))


def eval_r5_on_seqs(score_fn, seqs, n_items, slices=("all", "revisit", "advance", "nonrepeat"), max_seqs=None, seed=0):
    if max_seqs is not None and len(seqs) > max_seqs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(seqs), size=max_seqs, replace=False)
        seqs = [seqs[i] for i in idx]
    buckets = {sl: [] for sl in slices}
    n_queries = {sl: 0 for sl in slices}
    for seq in seqs:
        if len(seq) < 10:
            continue
        cuts = np.linspace(8, len(seq) - 1, num=min(EVAL_POS, max(1, len(seq) - 9)), dtype=int)
        local = {sl: [] for sl in slices}
        for cut in cuts:
            cut = int(cut)
            label = seq[cut][0]
            prefix = [x for x, _ in seq[:cut]]
            last = prefix[-1]
            hist = set(prefix)
            sl = action_slice(label, hist, last)
            scores = score_fn(prefix)
            hit = recall_at(scores, label, K)
            local["all"].append(hit)
            n_queries["all"] += 1
            if sl in ("revisit", "advance"):
                local[sl].append(hit)
                local["nonrepeat"].append(hit)
                n_queries[sl] += 1
                n_queries["nonrepeat"] += 1
        for sl in slices:
            if local[sl]:
                buckets[sl].append(float(np.mean(local[sl])))
    out = {}
    for sl in slices:
        arr = buckets[sl]
        out[sl] = {
            "mean_r5": float(np.mean(arr)) if arr else None,
            "n_sequences": len(arr),
            "n_queries": n_queries[sl],
            "na_below_min_slice": n_queries[sl] < MIN_SLICE_QUERIES,
            "min_slice_queries": MIN_SLICE_QUERIES,
        }
    return out


def share_next_equals_last(seqs_labels: list[list]) -> dict:
    hit = tot = 0
    for labels in seqs_labels:
        last = None
        for lab in labels:
            if last is not None:
                tot += 1
                if lab == last:
                    hit += 1
            last = lab
    return {"share": (hit / tot) if tot else None, "n": tot, "hits": hit}


def assist_skill_diagnostics() -> dict:
    raw = pd.read_csv(
        ASSIST_CSV,
        encoding="ISO-8859-1",
        low_memory=False,
        usecols=["order_id", "user_id", "skill_id", "correct"],
    )
    steps = [{"step": "raw_csv", "n_learners": int(raw["user_id"].nunique()), "n_rows": int(len(raw))}]
    raw = raw.dropna(subset=["user_id", "order_id", "correct"])
    steps.append(
        {"step": "drop_na_user_order_correct", "n_learners": int(raw["user_id"].nunique()), "n_rows": int(len(raw))}
    )
    raw["skill_id"] = raw["skill_id"].fillna(-1).astype(int)
    n_atomic = int(raw.loc[raw["skill_id"] >= 0, "skill_id"].nunique())
    g = (
        raw.groupby("order_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            correct=("correct", "first"),
            skills=("skill_id", lambda s: tuple(sorted({int(x) for x in s if int(x) >= 0}))),
        )
        .reset_index()
    )
    g["composite"] = g["skills"].map(lambda t: "_".join(map(str, t)) if t else "none")
    g["primary"] = g["skills"].map(lambda t: str(t[0]) if t else "none")
    steps.append(
        {"step": "collapse_order_id_skills", "n_learners": int(g["user_id"].nunique()), "n_rows": int(len(g))}
    )
    skill_build = {
        "rule": (
            "groupby order_id; skills = sorted unique skill_id (>=0); "
            "composite item id = '_'.join(skills); primary = first of sorted skills"
        ),
        "n_atomic_skill_ids_ge0": n_atomic,
        "n_composite_items_all_rows": int(g["composite"].nunique()),
        "n_primary_skills_all_rows": int(g["primary"].nunique()),
        "n_multi_skill_rows": int((g["skills"].map(len) > 1).sum()),
        "n_single_skill_rows": int((g["skills"].map(len) == 1).sum()),
        "n_empty_skill_rows": int((g["skills"].map(len) == 0).sum()),
        "why_n_items_150_vs_123_atomic": (
            f"Freeze ranking unit uses COMPOSITE skill tokens "
            f"({int(g['composite'].nunique())} distinct after collapse), not the "
            f"{n_atomic} atomic skill_ids. Composites arise from multi-skill order_ids."
        ),
    }
    g = g.sort_values(["user_id", "order_id"])
    lens = g.groupby("user_id").size()
    keep_min = set(lens[lens >= MIN_LEN].index.astype(int))
    g_min = g[g["user_id"].isin(keep_min)]
    steps.append(
        {
            "step": f"keep_learners_len_ge_{MIN_LEN}",
            "n_learners": int(g_min["user_id"].nunique()),
            "n_rows": int(len(g_min)),
        }
    )
    g_trunc = g_min.groupby("user_id", sort=False, group_keys=False).head(MAX_LEN)
    steps.append(
        {
            "step": f"truncate_each_learner_to_{MAX_LEN}",
            "n_learners": int(g_trunc["user_id"].nunique()),
            "n_rows": int(len(g_trunc)),
            "n_composite_items": int(g_trunc["composite"].nunique()),
            "n_primary_skills": int(g_trunc["primary"].nunique()),
        }
    )
    # sequences for repetition both ways
    comp_seqs, prim_seqs = [], []
    for _, grp in g_trunc.groupby("user_id", sort=False):
        comp_seqs.append(list(grp["composite"]))
        prim_seqs.append(list(grp["primary"]))
    return {
        "steps": steps,
        "skill_construction": skill_build,
        "repetition_share": {
            "composite_skill_token": share_next_equals_last(comp_seqs),
            "primary_atomic_skill": share_next_equals_last(prim_seqs),
        },
        "note": "4217→2920 is len>=12 (+truncate 200), not skill collapse.",
        "sequence_builder_n_learners": int(g_trunc["user_id"].nunique()),
    }


def junyi_timed_selection_note(n_returned: int) -> dict:
    return {
        "source": "data/junyi_raw/timed_interactions.npz (capped 8M-row extract)",
        "selection_rule": (
            f"Stream users in file order; keep a user when their contiguous block length >= {MIN_LEN}; "
            f"take the first {JUNYI_TIMED_MAX_USERS} such users; truncate each sequence to {MAX_LEN}."
        ),
        "n_requested_cap": JUNYI_TIMED_MAX_USERS,
        "n_returned": n_returned,
        "not_random_sample": True,
        "implication": "First-in-file users only; not a uniform sample of all 204k users in the extract.",
    }


def xes_question_order_structure() -> dict:
    """Share of transitions where next Q is successor in dataset question ordering (raw ids)."""
    from collections import Counter, defaultdict

    from run_followup3 import XES_Q, _parse_int_list

    frames = [
        pd.read_csv(XES_Q / n)
        for n in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    best: dict[int, list[int]] = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        if len(qs) < MIN_LEN:
            continue
        if uid not in best or len(qs) > len(best[uid]):
            best[uid] = qs[:MAX_LEN]
    all_q = sorted({q for seq in best.values() for q in seq})
    rank = {q: i for i, q in enumerate(all_q)}
    n = same = global_succ = raw_plus1 = 0
    big: dict[int, Counter] = defaultdict(Counter)
    for seq in best.values():
        for a, b in zip(seq, seq[1:]):
            n += 1
            if b == a:
                same += 1
            if rank[b] == rank[a] + 1:
                global_succ += 1
            if b == a + 1:
                raw_plus1 += 1
            big[a][b] += 1
    mode_hit = tot = 0
    for ctr in big.values():
        mode = ctr.most_common(1)[0][0]
        for b, c in ctr.items():
            tot += c
            if b == mode:
                mode_hit += c
    return {
        "ranking_unit": "question",
        "n_sequences": len(best),
        "n_unique_questions": len(all_q),
        "n_transitions": n,
        "share_next_equals_last": same / n if n else None,
        "share_next_is_global_order_successor": global_succ / n if n else None,
        "share_next_is_raw_id_plus_1": raw_plus1 / n if n else None,
        "share_next_is_empirical_mode_successor_of_prev": mode_hit / tot if tot else None,
        "interpretation": (
            "Non-trivial fraction of steps advance to the next question in the dataset's "
            "sorted question-id order (~same as raw id+1). Combined with very low continue "
            "rate, sequence probes can hit high non-repeat R@5 by learning quiz/curriculum "
            "order rather than spaced revisit structure."
        ),
        "min_slice_queries": MIN_SLICE_QUERIES,
    }


def freeze_dataset(name: str, seqs, uids, n_items, *, allow_ragr: bool, ranking_unit: str) -> dict:
    print(f"\n=== freeze {name} n_seq={len(seqs)} n_items={n_items} unit={ranking_unit} ===", flush=True)
    print(f"  GRID n_configs={len(GRID)} spec={GRID_SPEC}", flush=True)
    tr_i, va_i, te_i = learner_split(uids, REPLICATION_SEED)
    train_seqs = [seqs[i] for i in tr_i]
    val_seqs = [seqs[i] for i in va_i]
    test_uids = [uids[i] for i in te_i]
    test_index_hash = sha_list(sorted(int(u) for u in test_uids))

    pop = popularity_scores(train_seqs, n_items)

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

    max_r5_seqs = 400 if n_items >= 1000 or len(val_seqs) > 800 else None
    baselines = {
        "popularity": {
            "parameter_free": True,
            "val": eval_r5_on_seqs(pop_fn, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 3),
            "r5_val_max_seqs": max_r5_seqs,
        },
        "recency": {
            "parameter_free": True,
            "val": eval_r5_on_seqs(recency_fn, val_seqs, n_items, max_seqs=max_r5_seqs, seed=REPLICATION_SEED + 4),
            "r5_val_max_seqs": max_r5_seqs,
        },
    }
    pop_nonrep = baselines["popularity"]["val"]["nonrepeat"]["mean_r5"]

    trainable = ["gru_probe", "attn_probe"]
    if allow_ragr:
        trainable.append("ragr_lite")

    grid_results = {m: [] for m in trainable}
    winners = {}

    for ci, cfg in enumerate(GRID):
        cfg_id = f"lr{cfg['lr']}_emb{cfg['emb']}_do{cfg['dropout']}_maxep{cfg['max_epochs']}"
        print(f"  [{ci+1}/{len(GRID)}] {cfg_id}", flush=True)

        print("    gru_probe…", flush=True)
        gru = GRUProbe(n_items, emb=cfg["emb"], hidden=max(64, cfg["emb"] * 2))
        gru, gru_pack = train_seq_model_early_stop(
            gru, train_seqs, val_seqs, n_items, cfg=cfg, seed=REPLICATION_SEED
        )
        grid_results["gru_probe"].append({"config": cfg, "config_id": cfg_id, **gru_pack})

        print("    attn_probe…", flush=True)
        attn = AttnProbe(n_items, emb=cfg["emb"], dropout=cfg["dropout"])
        attn, attn_pack = train_seq_model_early_stop(
            attn, train_seqs, val_seqs, n_items, cfg=cfg, seed=REPLICATION_SEED + 1
        )
        grid_results["attn_probe"].append({"config": cfg, "config_id": cfg_id, **attn_pack})

        if allow_ragr:
            # Budget-matched evaluations (same n_configs); not epoch-trained
            def ragr_fn(prefix, pop=pop):
                scores = pop.copy()
                for rank, item in enumerate(reversed(list(dict.fromkeys(prefix))[:5])):
                    if 0 <= item < n_items:
                        scores[item] = scores[item] + (10 - rank)
                return scores

            ragr_val = eval_r5_on_seqs(ragr_fn, val_seqs, n_items)
            grid_results["ragr_lite"].append(
                {
                    "config": cfg,
                    "config_id": cfg_id,
                    "curve": [],
                    "best_epoch": None,
                    "best_val_ce_loss": None,
                    "final_val": ragr_val,
                    "note": "budget-matched lite; not DAG-full RAGR; no CE training",
                }
            )

    for m, rows in grid_results.items():
        best = max(
            rows,
            key=lambda r: (
                r["final_val"]["nonrepeat"]["mean_r5"] is not None,
                r["final_val"]["nonrepeat"]["mean_r5"] or -1,
            ),
        )
        winners[m] = best

    best_gru_nonrep = winners["gru_probe"]["final_val"]["nonrepeat"]["mean_r5"]
    best_attn_nonrep = winners["attn_probe"]["final_val"]["nonrepeat"]["mean_r5"]
    # Stop rule for preregistered family uses GRU only (attn excluded from H-rev)
    stop_tuning = bool(pop_nonrep is not None and (best_gru_nonrep or -1) <= pop_nonrep)

    # edge detection
    edge_flags = []
    for m, w in winners.items():
        if m == "ragr_lite":
            continue
        cfg = w["config"]
        if cfg["emb"] == max(GRID_SPEC["emb"]):
            edge_flags.append(f"{m}:emb_max")
        if cfg["emb"] == min(GRID_SPEC["emb"]):
            edge_flags.append(f"{m}:emb_min")
        if cfg["lr"] == max(GRID_SPEC["lr"]):
            edge_flags.append(f"{m}:lr_max")
        if cfg["lr"] == min(GRID_SPEC["lr"]):
            edge_flags.append(f"{m}:lr_min")
        if cfg["dropout"] == max(GRID_SPEC["dropout"]):
            edge_flags.append(f"{m}:dropout_max")
        if cfg["dropout"] == min(GRID_SPEC["dropout"]):
            edge_flags.append(f"{m}:dropout_min")
        if w.get("best_epoch") == GRID_SPEC["max_epochs"]:
            edge_flags.append(f"{m}:hit_max_epochs")

    def winner_report(m: str) -> dict:
        w = winners[m]
        fv = w["final_val"]
        return {
            "config": w["config"],
            "config_id": w["config_id"],
            "best_epoch": w.get("best_epoch"),
            "val_ce_loss": w.get("best_val_ce_loss"),
            "val_nonrepeat_r5": fv["nonrepeat"]["mean_r5"],
            "val_all_r5": fv["all"]["mean_r5"],
            "val_revisit_r5": fv["revisit"]["mean_r5"],
            "val_advance_r5": fv["advance"]["mean_r5"],
            "val_slice_n_queries": {
                sl: fv[sl]["n_queries"] for sl in ("all", "revisit", "advance", "nonrepeat")
            },
            "val_slice_na_below_min": {
                sl: fv[sl]["na_below_min_slice"] for sl in ("all", "revisit", "advance", "nonrepeat")
            },
            "selection_metric": "val_nonrepeat_r5",
            "curve": w.get("curve", []),
            "in_preregistered_h_rev": m in ("gru_probe",),
            "val": fv,  # full nested metrics for tests / PDF
        }

    return {
        "dataset": name,
        "ranking_unit_native": ranking_unit,
        "ranking_unit": ranking_unit,
        "hypothesis_unit_notes": {
            "H-rep_native_descriptive": ranking_unit,
            "H-rev_primary": "matched_concept_clusters~120 (pending)",
            "H-rev_secondary_native": ranking_unit,
            "H-graph": "N/A" if name != "junyi_timed" else "Junyi exercise/DAG",
            "H-forget": "N/A" if name == "assistments" else ("timed day gap" if name == "junyi_timed" else "timestamps pending validation"),
        },
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
            "test_uid_hash": test_index_hash,
            "test_touched": False,
            "note": "test indices hashed only; not evaluated in freeze",
        },
        "parameter_free_baselines": baselines,
        "equal_tuning_budget": {
            "applies_to": trainable,
            "grid_spec": GRID_SPEC,
            "n_configs": len(GRID),
            "grid": GRID,
            "max_windows": MAX_WINDOWS,
            "K": K,
            "validation_metric_for_selection": "mean_R@5 on nonrepeat (revisit+advance) validation sequences at early-stopped checkpoint",
            "early_stopping": "val_ce_loss with patience",
        },
        "grid_results": {
            m: [
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
                    "val_nonrepeat_n_queries": r["final_val"]["nonrepeat"]["n_queries"],
                    "val_nonrepeat_na": r["final_val"]["nonrepeat"]["na_below_min_slice"],
                }
                for r in rows
            ]
            for m, rows in grid_results.items()
        },
        "winners": {m: winner_report(m) for m in winners},
        "winner_metrics_note": (
            "Each winner reports val_nonrepeat_r5 and val_ce_loss; "
            "model selection used val_nonrepeat_r5 (not CE)."
        ),
        "winner_on_grid_edge": edge_flags,
        "stop_rule": {
            "rule": (
                "if best GRU probe nonrepeat val R@5 <= popularity nonrepeat val R@5, stop tuning "
                "(attn excluded from preregistered H-rev family)"
            ),
            "popularity_nonrepeat_r5": pop_nonrep,
            "best_gru_nonrepeat_r5": best_gru_nonrep,
            "best_attn_nonrepeat_r5_exploratory": best_attn_nonrep,
            "best_probe_nonrepeat_r5": best_gru_nonrep,  # alias for older tests
            "stop_tuning": stop_tuning,
        },
        "ragr_applicable": allow_ragr,
        "attn_probe_architecture": ATTN_PROBE_ARCHITECTURE,
        "display_names": {
            "gru_probe": "GRU-based next-item probe",
            "attn_probe": "attention-based next-item probe (exploratory; excluded from H-rev)",
            "ragr_lite": "RAGR-lite (Junyi budget-matched)",
            "popularity": "popularity (parameter-free)",
            "recency": "recency (parameter-free)",
        },
    }


def main():
    t0 = time.time()
    assert len(GRID) > 1, "grid must have more than one point"
    print("GRID_SPEC", GRID_SPEC, "n_configs", len(GRID), flush=True)
    partial_path = PHASE / "probe_freeze.partial.json"
    assert len(GRID) == 24, len(GRID)
    out = {
        "schema_version": 3,
        "replication_seed": REPLICATION_SEED,
        "split_rule": SPLIT,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "min_slice_queries": MIN_SLICE_QUERIES,
        "grid_spec": GRID_SPEC,
        "n_grid_configs": len(GRID),
        "grid": GRID,
        "selection_metric": "val_nonrepeat_r5_at_early_stopped_checkpoint",
        "h_rev_unit_policy": H_REV_UNIT_POLICY,
        "attn_probe_architecture": ATTN_PROBE_ARCHITECTURE,
        "prior_peeking_disclosure": (
            "Follow-up 3 inspected these datasets before preregistration; freeze uses a new seed/split"
        ),
        "deviations_doc": "docs/DEVIATIONS.md",
        "assistments_filter_funnel": assist_skill_diagnostics(),
        "datasets": {},
    }
    if partial_path.exists():
        prev = json.loads(partial_path.read_text())
        if prev.get("replication_seed") == REPLICATION_SEED and prev.get("n_grid_configs") == len(GRID):
            out["datasets"] = prev.get("datasets", {})
            print("resuming from partial:", list(out["datasets"]), flush=True)

    if "junyi_timed" not in out["datasets"]:
        j_seqs, j_uids = build_learner_sequences_junyi_timed(
            max_users=JUNYI_TIMED_MAX_USERS, min_len=MIN_LEN, max_len=MAX_LEN
        )
        n_j = max(max(x for x, _ in s) for s in j_seqs) + 1
        junyi = freeze_dataset("junyi_timed", j_seqs, j_uids, n_j, allow_ragr=True, ranking_unit="exercise")
        junyi["learner_selection"] = junyi_timed_selection_note(len(j_seqs))
        out["datasets"]["junyi_timed"] = junyi
        partial_path.write_text(json.dumps(out, indent=2) + "\n")
        print("checkpointed junyi_timed", flush=True)

    if "assistments" not in out["datasets"]:
        a_seqs, a_uids, n_a = build_learner_sequences_assist(min_len=MIN_LEN, max_len=MAX_LEN)
        out["datasets"]["assistments"] = freeze_dataset(
            "assistments", a_seqs, a_uids, n_a, allow_ragr=False, ranking_unit="composite_skill_token"
        )
        partial_path.write_text(json.dumps(out, indent=2) + "\n")
        print("checkpointed assistments", flush=True)

    if "xes3g5m" not in out["datasets"]:
        x_seqs, x_uids, n_x = build_learner_sequences_xes(min_len=MIN_LEN, max_len=MAX_LEN)
        xes = freeze_dataset(
            "xes3g5m", x_seqs, x_uids, n_x, allow_ragr=False, ranking_unit="question"
        )
        xes["question_order_structure"] = xes_question_order_structure()
        out["datasets"]["xes3g5m"] = xes
        partial_path.write_text(json.dumps(out, indent=2) + "\n")
        print("checkpointed xes3g5m", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    write_json("probe_freeze.json", out)
    if partial_path.exists():
        partial_path.unlink()
    print("done freeze", out["elapsed_sec"], "s", "n_grid", len(GRID), flush=True)


if __name__ == "__main__":
    main()
