#!/usr/bin/env python3
"""Freeze-partition confirmatory replication runner (prereg-v1).

Train on freeze train+val with frozen configs + fixed best_epoch; three GRU seeds.
Score freeze test only with --touch-test (once; lock file). See docs/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch
import torch.nn as nn

from junyi_pipeline import OUT, SEED
from run_followup3 import (
    GRUProbe,
    action_slice,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    popularity_scores,
    score_seq_model,
)
from run_probe_freeze import (
    MAX_WINDOWS,
    REPLICATION_SEED,
    eval_r5_on_seqs,
    learner_split,
    sha_list,
    train_seq_model_early_stop,
)
from run_valid_eval import recall_at
from gru_markov_paired_ci import remap_pair

PHASE = OUT / "phases"
FREEZE_SEED = 20261004
MIN_SLICE_QUERIES = 500
MIN_PRIMARY_NONREPEAT_QUERIES = 500
K = 5
N_BOOT = 10_000
M_PLANNED = 6
GRU_TRAIN_SEEDS = (20261101, 20261102, 20261103)
REQUIRED = ["popularity", "recency", "markov_order1", "gru_probe"]
T_R5 = 0.01
MAX_LEN_CTX = 50
BATCH = 64
# Match freeze: GRUProbe(n_items, emb=emb, hidden=max(64, emb * 2)); dropout unused
CATALOG_N_ITEMS = {"junyi_timed": 835, "assistments": 150, "xes3g5m": 7439}
TOUCH_LOCK = PHASE / "touch_test.lock.json"
DEVIATIONS = ROOT / "docs" / "DEVIATIONS.md"
OVERRIDE_MARKER = "TOUCH_TEST_OVERRIDE:"


def load_freeze() -> dict:
    return json.loads((PHASE / "probe_freeze.json").read_text())


def stable_seed(*parts: str) -> int:
    """Deterministic seed from string parts via zlib.crc32 (not builtin hashing)."""
    return zlib.crc32("::".join(parts).encode("utf-8")) & 0xFFFFFFFF


def bonferroni_level(m: int) -> float | None:
    if m < 1:
        return None
    return 1.0 - 0.05 / m


def family_size_after_na(n_na_cells: int, m_planned: int = M_PLANNED) -> dict:
    m = m_planned - int(n_na_cells)
    return {
        "m_planned": m_planned,
        "n_na_cells": int(n_na_cells),
        "m": m,
        "ci_level": bonferroni_level(m),
        "rule": (
            "primary cell N/A if non-repeat queries < 500; m := m_planned − n_na; "
            "CI level = 1 − 0.05/m (undefined if m < 1)"
        ),
    }


def ci_quantile_bounds(m: int) -> tuple[float, float]:
    alpha = 0.05 / m
    return alpha / 2.0, 1.0 - alpha / 2.0


def ci_verdict(ci, t: float = T_R5) -> str:
    if not ci or ci[0] is None or (isinstance(ci[0], float) and np.isnan(ci[0])):
        return "unavailable"
    lo, hi = float(ci[0]), float(ci[1])
    if lo > t:
        return "supported_A"
    if hi < -t:
        return "supported_B"
    if lo > -t and hi < t:
        return "negligible"
    return "inconclusive"


def combine_two_method_verdicts(v_text: str, v_cooc: str) -> dict:
    """Registered cluster-unit combination (Junyi / ASSISTments)."""
    supported = {"supported_A", "supported_B"}
    if v_text in supported and v_text == v_cooc:
        reg = v_text
    else:
        # supported_* + negligible, disagreement, inconclusive, N/A → inconclusive
        reg = "inconclusive"
    return {
        "text_verdict": v_text,
        "cooccurrence_verdict": v_cooc,
        "registered_cluster_verdict": reg,
        "rule": (
            "same supported_* on both → that verdict; "
            "supported_* with negligible, disagreement, or inconclusive → inconclusive"
        ),
    }


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    common = [m for m in order_a if m in order_b]
    if len(common) < 2:
        return None
    ra = {m: i for i, m in enumerate(order_a)}
    rb = {m: i for i, m in enumerate(order_b)}
    xs = [ra[m] for m in common]
    ys = [rb[m] for m in common]
    conc = disc = 0
    for i, j in combinations(range(len(common)), 2):
        dx, dy = xs[i] - xs[j], ys[i] - ys[j]
        if dx == 0 or dy == 0:
            continue
        if dx * dy > 0:
            conc += 1
        else:
            disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom else None


def fit_markov(train_seqs, n_items):
    counts = np.ones((n_items, n_items), dtype=np.float64) * 1e-3
    for seq in train_seqs:
        ids = [x for x, _ in seq if 0 <= x < n_items]
        for a, b in zip(ids, ids[1:]):
            counts[a, b] += 1.0
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def parse_winner_cfg(winner_cfg: dict | None) -> tuple[int, float, int, float | None]:
    winner_cfg = winner_cfg or {}
    cfg = winner_cfg.get("config") if isinstance(winner_cfg.get("config"), dict) else winner_cfg
    if not isinstance(cfg, dict):
        cfg = {}
    emb = int(cfg.get("emb") or winner_cfg.get("emb") or 64)
    lr = float(cfg.get("lr", 1e-3))
    best_ep = int(winner_cfg.get("best_epoch") or 3)
    dropout = cfg.get("dropout")
    return emb, lr, best_ep, (float(dropout) if dropout is not None else None)


def train_gru_fixed(n_items, train_seqs, *, emb: int, lr: float, best_ep: int, seed: int):
    """Faithful to freeze GRUProbe + windowing; fixed epoch count (no early stop on test).

    Matches freeze: hidden=max(64, emb*2), max_len=50, batch=64, pair cap=MAX_WINDOWS (20k).
    GRUProbe has no dropout module — winner dropout is recorded but ignored (freeze grid no-op for GRU).
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    hidden = max(64, emb * 2)
    gru = GRUProbe(n_items, emb=emb, hidden=hidden)
    opt = torch.optim.Adam(gru.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs = []
    for seq in train_seqs:
        ids = [x + 1 for x, _ in seq]
        for t in range(1, len(ids)):
            pairs.append((ids[max(0, t - MAX_LEN_CTX) : t], seq[t][0]))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pairs)
    pairs = pairs[:MAX_WINDOWS]
    gru.train()
    for ep in range(int(best_ep)):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), BATCH):
            bp = pairs[i : i + BATCH]
            if not bp:
                continue
            max_t = max(len(c) for c, _ in bp)
            x = torch.zeros(len(bp), max_t, dtype=torch.long)
            y = torch.tensor([lab for _, lab in bp], dtype=torch.long)
            for bi, (ctx, _) in enumerate(bp):
                x[bi, -len(ctx) :] = torch.tensor(ctx, dtype=torch.long)
            opt.zero_grad()
            loss_fn(gru(x), y).backward()
            opt.step()
    gru.eval()
    return gru


def verify_and_split(dataset: str, uids: list[int], freeze: dict):
    fr = freeze["datasets"][dataset]["split"]
    assert fr["seed"] == FREEZE_SEED
    tr_i, va_i, te_i = learner_split(uids, FREEZE_SEED)
    train_uids = [int(uids[i]) for i in tr_i]
    val_uids = [int(uids[i]) for i in va_i]
    test_uids = [int(uids[i]) for i in te_i]
    assert set(val_uids).isdisjoint(set(test_uids))
    assert set(train_uids).isdisjoint(set(test_uids))
    assert set(train_uids).isdisjoint(set(val_uids))
    h_test, h_val, h_train = sha_list(sorted(test_uids)), sha_list(sorted(val_uids)), sha_list(sorted(train_uids))
    assert h_test == fr["test_uid_hash"], f"{dataset} test hash mismatch"
    assert h_val == fr["val_uid_hash"], f"{dataset} val hash mismatch"
    assert h_train == fr["train_uid_hash"], f"{dataset} train hash mismatch"
    assert fr.get("test_touched") is False, f"{dataset} freeze test already touched"
    return tr_i, va_i, te_i, {
        "n_train": len(tr_i),
        "n_val": len(va_i),
        "n_test": len(te_i),
        "test_uid_hash": h_test,
        "val_uid_hash": h_val,
        "train_uid_hash": h_train,
        "hash_match_freeze": True,
        "val_test_overlap": 0,
    }


def recency_scores(prefix, n_items):
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


def paired_bootstrap_ci(
    delta_per_seq: np.ndarray,
    m: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = FREEZE_SEED,
) -> dict:
    """Sequence-level bootstrap; delta_per_seq may contain NaN (dropped)."""
    delta = np.asarray(delta_per_seq, dtype=float)
    mask = np.isfinite(delta)
    delta = delta[mask]
    n = int(delta.size)
    if n < 20 or m < 1:
        return {
            "n_sequences": n,
            "ci": None,
            "ci_95": None,
            "mean_delta": None,
            "verdict": "unavailable" if m >= 1 else "N/A_family_empty",
            "n_boot": n_boot,
            "resampling": "sequence_level",
            "bonferroni_m": m,
            "bonferroni_level": bonferroni_level(m),
        }
    rng = np.random.default_rng(int(seed))
    dboot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        dboot[b] = float(delta[idx].mean())
    q_lo, q_hi = ci_quantile_bounds(m)
    ci_bonf = [float(np.quantile(dboot, q_lo)), float(np.quantile(dboot, q_hi))]
    ci_95 = [float(np.quantile(dboot, 0.025)), float(np.quantile(dboot, 0.975))]
    return {
        "n_sequences": n,
        "n_boot": n_boot,
        "resampling": "sequence_level",
        "mean_delta": float(delta.mean()),
        "ci": ci_bonf,
        "ci_95": ci_95,
        "bonferroni_m": m,
        "bonferroni_level": bonferroni_level(m),
        "verdict": ci_verdict(ci_bonf, T_R5),
        "t": T_R5,
        "bootstrap_seed": int(seed),
    }


def nanmean_finite(arr: np.ndarray) -> float | None:
    a = np.asarray(arr, dtype=float)
    m = np.isfinite(a)
    if not m.any():
        return None
    return float(a[m].mean())


def score_unit_multisseed(
    train_seqs,
    test_seqs,
    n_items,
    *,
    winner_cfg: dict,
    unit_name: str,
):
    """Per-sequence R@5 arrays indexed by sequence (NaN where undefined)."""
    n_seq = len(test_seqs)
    pop = popularity_scores(train_seqs, n_items)
    trans = fit_markov(train_seqs, n_items)
    emb, lr, best_ep, dropout = parse_winner_cfg(winner_cfg)

    grus = [
        train_gru_fixed(n_items, train_seqs, emb=emb, lr=lr, best_ep=best_ep, seed=s)
        for s in GRU_TRAIN_SEEDS
    ]

    slices = ["all", "revisit", "advance", "nonrepeat"]
    # method -> slice -> length-n_seq array
    per_seq = {m: {sl: np.full(n_seq, np.nan, dtype=float) for sl in slices} for m in REQUIRED}
    per_seq_gru_seed = {
        s: {sl: np.full(n_seq, np.nan, dtype=float) for sl in slices} for s in GRU_TRAIN_SEEDS
    }
    query_counts = {sl: 0 for sl in slices}
    query_rows = []
    cut_lists = [None] * n_seq  # for secondary alignment checks

    for seq_i, seq in enumerate(test_seqs):
        if len(seq) < 10:
            continue
        cuts = np.linspace(8, len(seq) - 1, num=min(5, len(seq) - 9), dtype=int)
        cut_lists[seq_i] = [int(c) for c in cuts]
        # accumulators: list of hits per slice (recency: 0 on advance for all/nonrepeat)
        local = {m: {sl: [] for sl in slices} for m in REQUIRED}
        local_seed = {s: {sl: [] for sl in slices} for s in GRU_TRAIN_SEEDS}

        for cut in cuts:
            cut = int(cut)
            label = seq[cut][0]
            prefix = [x for x, _ in seq[:cut]]
            last = prefix[-1]
            hist = set(prefix)
            sl = action_slice(label, hist, last)
            query_counts["all"] += 1
            if sl in ("revisit", "advance"):
                query_counts[sl] += 1
                query_counts["nonrepeat"] += 1

            seed_hits = [
                recall_at(score_seq_model(gru, prefix, n_items), label, K) for gru in grus
            ]
            gru_hit = float(np.mean(seed_hits))
            pop_h = recall_at(pop.copy(), label, K)
            mk_h = recall_at(trans[last].copy() if prefix else pop.copy(), label, K)
            # Recency: real hit on revisit/continue; 0 on advance for all/nonrepeat; N/A on advance slice
            if sl == "advance":
                rec_for_all = 0.0
                rec_advance = None  # N/A for advance-slice display
            else:
                rec_for_all = recall_at(recency_scores(prefix, n_items), label, K)
                rec_advance = rec_for_all

            hits = {
                "popularity": pop_h,
                "recency": rec_for_all if sl != "advance" else 0.0,
                "markov_order1": mk_h,
                "gru_probe": gru_hit,
            }
            query_rows.append(
                {
                    "seq_i": seq_i,
                    "cut": cut,
                    "slice": sl,
                    "nonrepeat": sl in ("revisit", "advance"),
                    "hits": {
                        "popularity": pop_h,
                        "recency": rec_for_all if sl != "advance" else 0.0,
                        "recency_advance_slice": rec_advance,  # None ⇒ N/A
                        "markov_order1": mk_h,
                        "gru_probe": gru_hit,
                    },
                    "gru_hits_by_seed": {str(s): float(h) for s, h in zip(GRU_TRAIN_SEEDS, seed_hits)},
                }
            )

            for m in REQUIRED:
                h = hits[m]
                local[m]["all"].append(h)
                if sl == "revisit":
                    local[m]["revisit"].append(h)
                    local[m]["nonrepeat"].append(h)
                elif sl == "advance":
                    if m == "recency":
                        # advance slice: N/A (do not append); all/nonrepeat already got 0 above
                        local[m]["nonrepeat"].append(0.0)
                    else:
                        local[m]["advance"].append(h)
                        local[m]["nonrepeat"].append(h)

            for s, h in zip(GRU_TRAIN_SEEDS, seed_hits):
                local_seed[s]["all"].append(h)
                if sl in ("revisit", "advance"):
                    local_seed[s][sl].append(h)
                    local_seed[s]["nonrepeat"].append(h)

        for m in REQUIRED:
            for sln in slices:
                if local[m][sln]:
                    per_seq[m][sln][seq_i] = float(np.mean(local[m][sln]))
                # else leave NaN (e.g. recency on advance-only sequences for advance slice)
        for s in GRU_TRAIN_SEEDS:
            for sln in slices:
                if local_seed[s][sln]:
                    per_seq_gru_seed[s][sln][seq_i] = float(np.mean(local_seed[s][sln]))

    seed_means = {}
    for s in GRU_TRAIN_SEEDS:
        seed_means[str(s)] = nanmean_finite(per_seq_gru_seed[s]["nonrepeat"])
    vals = [v for v in seed_means.values() if v is not None]
    between_seed = {
        "gru_nonrepeat_r5_by_seed": seed_means,
        "mean_across_seeds": float(np.mean(vals)) if vals else None,
        "std_across_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "range_across_seeds": float(max(vals) - min(vals)) if vals else None,
        "markov": "deterministic_single_fit",
        "gru_seeds": list(GRU_TRAIN_SEEDS),
    }

    rankings = {}
    for sl in slices:
        rows = []
        for m in REQUIRED:
            if m == "recency" and sl == "advance":
                rows.append(
                    {
                        "method": m,
                        "r5": None,
                        "r5_display": "N/A",
                        "note": "recency N/A for advance-slice display/ordering; counts as 0 in all/nonrepeat",
                    }
                )
                continue
            r5 = nanmean_finite(per_seq[m][sl])
            rows.append(
                {
                    "method": m,
                    "r5": r5,
                    "n_sequences_defined": int(np.isfinite(per_seq[m][sl]).sum()),
                }
            )
        orderable = [r for r in rows if r.get("r5") is not None]
        orderable.sort(key=lambda r: r["r5"], reverse=True)
        rankings[sl] = {
            "rows": rows,
            "order": [r["method"] for r in orderable],
            "n_queries": query_counts[sl],
            "min_slice_queries": MIN_SLICE_QUERIES,
            "slice_ok": query_counts[sl] >= MIN_SLICE_QUERIES,
            "primary_cell_ok": query_counts.get("nonrepeat", 0) >= MIN_PRIMARY_NONREPEAT_QUERIES
            if sl == "nonrepeat"
            else True,
        }

    # Flip probs / CIs: pair by sequence index (NaN-aware)
    flip = {}
    for sl in slices:
        if not rankings[sl]["slice_ok"]:
            flip[sl] = {"status": "N/A_below_min_queries", "n_queries": query_counts[sl]}
            continue
        flip[sl] = {}
        methods_sl = [m for m in REQUIRED if not (m == "recency" and sl == "advance")]
        for a, b in combinations(methods_sl, 2):
            aa = per_seq[a][sl]
            bb = per_seq[b][sl]
            both = np.isfinite(aa) & np.isfinite(bb)
            n = int(both.sum())
            if n < 20:
                flip[sl][f"{a}_vs_{b}"] = {"n": n, "flip_prob": None}
                continue
            av, bv = aa[both], bb[both]
            point_a_better = bool(av.mean() > bv.mean())
            rng = np.random.default_rng(stable_seed("flip", unit_name, sl, a, b))
            boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                boots.append(1.0 if av[idx].mean() > bv[idx].mean() else 0.0)
            flip_prob = float(np.mean(np.asarray(boots) != point_a_better))
            # provisional CI at m_planned; eligibility pipeline recomputes primary GM
            pack = paired_bootstrap_ci(
                av - bv,
                M_PLANNED,
                seed=stable_seed("ci", unit_name, sl, a, b),
            )
            flip[sl][f"{a}_vs_{b}"] = {
                "n_sequences": n,
                "n_boot": N_BOOT,
                "resampling": "sequence_level",
                "paired_by": "sequence_index_nan_aware",
                "point_a_better": point_a_better,
                "flip_probability": flip_prob,
                "mean_delta_a_minus_b": float((av - bv).mean()),
                "ci_95": pack["ci_95"],
                "ci_bonferroni_m_planned": pack["ci"],
                "verdict_m_planned": pack["verdict"],
            }

    kendall = {
        sl: {
            "kendall_tau_vs_all": kendall_tau(rankings["all"]["order"], rankings[sl]["order"]),
            "all_order": rankings["all"]["order"],
            "slice_order": rankings[sl]["order"],
        }
        for sl in ("revisit", "advance", "nonrepeat")
    }

    # Store arrays as lists with null for NaN (JSON)
    def arr_to_json(a: np.ndarray):
        return [None if not np.isfinite(x) else float(x) for x in a]

    return {
        "unit": unit_name,
        "scored": True,
        "n_sequences": n_seq,
        "n_items_catalog": n_items,
        "gru_train_seeds": list(GRU_TRAIN_SEEDS),
        "gru_aggregation": "per_sequence R@5 = mean over three seeds; bootstrap on that mean",
        "markov": "deterministic_single_fit",
        "dropout_from_winner_config": dropout,
        "dropout_note": (
            "GRUProbe has no dropout module; freeze dropout grid is a no-op for GRU. "
            "Value recorded for audit only."
        ),
        "train_faithfulness": {
            "hidden": f"max(64, emb*2) → {max(64, emb * 2)}",
            "max_windows": MAX_WINDOWS,
            "batch": BATCH,
            "max_len_ctx": MAX_LEN_CTX,
            "pair_cap_matches_freeze": True,
        },
        "winner_cfg": {"emb": emb, "lr": lr, "best_epoch": best_ep, "dropout_ignored": dropout},
        "between_seed_spread": between_seed,
        "query_counts": query_counts,
        "rankings_by_slice_r5": rankings,
        "kendall_tau_by_slice": kendall,
        "bootstrap_flip_probability": flip,
        "per_sequence_r5": {m: {sl: arr_to_json(per_seq[m][sl]) for sl in slices} for m in REQUIRED},
        "per_sequence_r5_arrays": per_seq,  # kept in-memory for claim assembly; stripped before write
        "cut_lists": cut_lists,
        "query_rows": query_rows,
        "recency_rule": (
            "recency counts as 0 on advance queries for all and nonrepeat; "
            "N/A only for display/ordering on the advance slice"
        ),
        "recall_at_tie_rule": "lower item index wins among equal scores (stable lexsort)",
    }


def load_cluster_map(rel_path: str) -> dict[int, int]:
    raw = json.loads((ROOT / rel_path).read_text())
    return {int(k): int(v) for k, v in raw.items()}


def map_seqs_preserve_length(seqs, item_to_cluster: dict[int, int]):
    """Map every position; fail if any item missing (length must match native)."""
    out = []
    for seq in seqs:
        mapped = []
        for i, y in seq:
            if i not in item_to_cluster:
                raise RuntimeError(f"item {i} missing from cluster map — cannot preserve length")
            mapped.append((item_to_cluster[i], y))
        out.append(mapped)
    return out


def assert_native_cluster_alignment(native_pack: dict, cluster_pack: dict, label: str):
    n_n = native_pack["n_sequences"]
    n_c = cluster_pack["n_sequences"]
    if n_n != n_c:
        raise RuntimeError(f"{label}: sequence count mismatch native={n_n} cluster={n_c}")
    cuts_n, cuts_c = native_pack["cut_lists"], cluster_pack["cut_lists"]
    for i in range(n_n):
        if cuts_n[i] is None and cuts_c[i] is None:
            continue
        if cuts_n[i] != cuts_c[i]:
            raise RuntimeError(
                f"{label}: cut list mismatch at seq {i}: native={cuts_n[i]} cluster={cuts_c[i]}"
            )
    # lengths of underlying sequences implied by cut max
    return True


def secondary_common_nonrepeat(native_pack: dict, cluster_pack: dict, m: int) -> dict:
    assert_native_cluster_alignment(native_pack, cluster_pack, "secondary_common_nonrepeat")
    c_map = {(q["seq_i"], q["cut"]): q for q in cluster_pack["query_rows"]}
    by_seq_n, by_seq_c = {}, {}
    n_common = 0
    for q in native_pack["query_rows"]:
        if not q["nonrepeat"]:
            continue
        cq = c_map.get((q["seq_i"], q["cut"]))
        if cq is None or not cq["nonrepeat"]:
            continue
        n_common += 1
        si = q["seq_i"]
        by_seq_n.setdefault(si, []).append(q["hits"]["gru_probe"] - q["hits"]["markov_order1"])
        by_seq_c.setdefault(si, []).append(cq["hits"]["gru_probe"] - cq["hits"]["markov_order1"])

    def to_aligned(d):
        arr = np.full(native_pack["n_sequences"], np.nan, dtype=float)
        for k, vals in d.items():
            arr[k] = float(np.mean(vals))
        return arr

    dn, dc = to_aligned(by_seq_n), to_aligned(by_seq_c)
    native_ci = paired_bootstrap_ci(dn, m, seed=stable_seed("sec", "native"))
    cluster_ci = paired_bootstrap_ci(dc, m, seed=stable_seed("sec", "cluster"))
    eligible = n_common >= MIN_PRIMARY_NONREPEAT_QUERIES
    if not eligible:
        native_ci = {**native_ci, "verdict": "N/A", "status": "below_min_common_queries"}
        cluster_ci = {**cluster_ci, "verdict": "N/A", "status": "below_min_common_queries"}
    return {
        "analysis": "secondary",
        "definition": (
            "queries non-repeat at both native and freeze-primary cluster units; "
            "aligned by (seq_i, cut); equal lengths/cuts asserted"
        ),
        "n_queries_common": n_common,
        "eligible": eligible,
        "native_gru_minus_markov": native_ci,
        "cluster_gru_minus_markov": cluster_ci,
        "reporting_rule": (
            "Report primary and secondary. A flip present only in the primary slices "
            "is reported as a primary-only flip."
        ),
    }


def winner_cfg_for(freeze: dict, dataset: str, *, unit: str = "native") -> dict:
    ds = freeze["datasets"][dataset]
    if unit == "cluster":
        winners = (ds.get("cluster_unit") or {}).get("winners") or {}
    else:
        winners = ds.get("winners") or {}
    for key in ("gru_probe", "gru", "native"):
        if key in winners and isinstance(winners[key], dict):
            return winners[key]
    return {}


def gm_delta_array(pack: dict) -> np.ndarray:
    g = np.asarray(pack["per_sequence_r5_arrays"]["gru_probe"]["nonrepeat"], dtype=float)
    m = np.asarray(pack["per_sequence_r5_arrays"]["markov_order1"]["nonrepeat"], dtype=float)
    return g - m


def assemble_primary_claims(dataset_packs: dict) -> dict:
    """eligibility → m → Bonferroni CIs → dataset/project claims."""
    # Provisional query counts / raw deltas per cell
    cell_specs = []
    for ds in ("junyi_timed", "assistments", "xes3g5m"):
        pack = dataset_packs[ds]
        native = pack["native"]
        n_q_n = native["query_counts"]["nonrepeat"]
        cell_specs.append(
            {
                "dataset": ds,
                "unit": "native",
                "n_queries_nonrepeat": n_q_n,
                "eligible": n_q_n >= MIN_PRIMARY_NONREPEAT_QUERIES,
                "delta": gm_delta_array(native),
                "method_detail": None,
            }
        )
        if ds == "xes3g5m":
            cl = pack["cluster_cooccurrence"]
            n_q_c = cl["query_counts"]["nonrepeat"]
            cell_specs.append(
                {
                    "dataset": ds,
                    "unit": "cluster",
                    "n_queries_nonrepeat": n_q_c,
                    "eligible": n_q_c >= MIN_PRIMARY_NONREPEAT_QUERIES,
                    "delta": gm_delta_array(cl),
                    "method_detail": {"methods": ["cooccurrence"], "amendment": True},
                }
            )
        else:
            ct = pack["cluster_text"]
            cc = pack["cluster_cooccurrence"]
            n_q_c = min(ct["query_counts"]["nonrepeat"], cc["query_counts"]["nonrepeat"])
            cell_specs.append(
                {
                    "dataset": ds,
                    "unit": "cluster",
                    "n_queries_nonrepeat": n_q_c,
                    "eligible": (
                        ct["query_counts"]["nonrepeat"] >= MIN_PRIMARY_NONREPEAT_QUERIES
                        and cc["query_counts"]["nonrepeat"] >= MIN_PRIMARY_NONREPEAT_QUERIES
                    ),
                    "delta_text": gm_delta_array(ct),
                    "delta_cooc": gm_delta_array(cc),
                    "method_detail": {
                        "methods": ["text", "cooccurrence"],
                        "config_source": "primary_cluster_winner_no_retune_on_cooc",
                    },
                }
            )

    n_na = sum(1 for c in cell_specs if not c["eligible"])
    fam = family_size_after_na(n_na)
    m = fam["m"]

    cells_out = []
    verdict_by = {}
    for c in cell_specs:
        key = (c["dataset"], c["unit"])
        if not c["eligible"] or m < 1:
            cell = {
                **{k: c[k] for k in ("dataset", "unit", "n_queries_nonrepeat", "eligible")},
                "verdict": "N/A",
                "status": "below_min_nonrepeat_queries" if not c["eligible"] else "family_empty",
                "ci": None,
                "bonferroni_m": m,
                "bonferroni_level": fam["ci_level"],
                "method_detail": c.get("method_detail"),
            }
            cells_out.append(cell)
            verdict_by[key] = "N/A"
            continue

        if c["unit"] == "native" or c["dataset"] == "xes3g5m":
            pack_ci = paired_bootstrap_ci(
                c["delta"],
                m,
                seed=stable_seed("primary", c["dataset"], c["unit"]),
            )
            cell = {
                "dataset": c["dataset"],
                "unit": c["unit"],
                "n_queries_nonrepeat": c["n_queries_nonrepeat"],
                "eligible": True,
                "mean_delta": pack_ci["mean_delta"],
                "ci": pack_ci["ci"],
                "ci_95": pack_ci["ci_95"],
                "verdict": pack_ci["verdict"],
                "bonferroni_m": m,
                "bonferroni_level": fam["ci_level"],
                "method_detail": c.get("method_detail"),
            }
            verdict_by[key] = pack_ci["verdict"]
        else:
            ci_t = paired_bootstrap_ci(
                c["delta_text"], m, seed=stable_seed("primary", c["dataset"], "cluster_text")
            )
            ci_c = paired_bootstrap_ci(
                c["delta_cooc"], m, seed=stable_seed("primary", c["dataset"], "cluster_cooc")
            )
            comb = combine_two_method_verdicts(ci_t["verdict"], ci_c["verdict"])
            cell = {
                "dataset": c["dataset"],
                "unit": "cluster",
                "n_queries_nonrepeat": c["n_queries_nonrepeat"],
                "eligible": True,
                "text": ci_t,
                "cooccurrence": ci_c,
                "combination": comb,
                "verdict": comb["registered_cluster_verdict"],
                "bonferroni_m": m,
                "bonferroni_level": fam["ci_level"],
                "method_detail": c.get("method_detail"),
            }
            verdict_by[key] = comb["registered_cluster_verdict"]
        cells_out.append(cell)

    dataset_claims = {}
    for ds in ("junyi_timed", "assistments", "xes3g5m"):
        vn, vc = verdict_by[(ds, "native")], verdict_by[(ds, "cluster")]
        supported = {"supported_A", "supported_B"}
        satisfies = vn in supported and vc in supported and vn != vc
        dataset_claims[ds] = {
            "native_verdict": vn,
            "cluster_verdict": vc,
            "satisfies_primary_claim": satisfies,
            "rule": "both units supported_* with opposite signs",
        }

    satisfying = [ds for ds, d in dataset_claims.items() if d["satisfies_primary_claim"]]
    project = {
        "satisfies_project_claim": len(satisfying) >= 1,
        "datasets_satisfying": satisfying,
        "datasets_that_can": ["junyi_timed", "assistments", "xes3g5m"],
        "rule": "at least one dataset satisfies the primary claim",
    }

    return {
        "eligibility_and_family": fam,
        "cells": cells_out,
        "dataset_claims": dataset_claims,
        "project_claim": project,
        "pipeline": (
            "eligibility (>=500 non-repeat queries/cell) → m = 6 − n_NA → "
            "Bonferroni level → CIs at that level → dataset claims → project claim"
        ),
    }


def check_touch_lock():
    if not TOUCH_LOCK.exists():
        return
    text = DEVIATIONS.read_text() if DEVIATIONS.exists() else ""
    if OVERRIDE_MARKER not in text:
        raise SystemExit(
            f"Refusing second --touch-test: {TOUCH_LOCK} exists. "
            f"To override, add a line containing '{OVERRIDE_MARKER}' with justification "
            f"to docs/DEVIATIONS.md, then re-run."
        )
    print("WARNING: touch-test override recorded in DEVIATIONS.md", flush=True)


def write_touch_lock():
    TOUCH_LOCK.write_text(
        json.dumps(
            {
                "touched": True,
                "note": "Freeze test scored once after prereg-v1. Second run requires DEVIATIONS override.",
            },
            indent=2,
        )
        + "\n"
    )


def strip_arrays(pack: dict) -> dict:
    out = {k: v for k, v in pack.items() if k not in ("per_sequence_r5_arrays", "query_rows", "cut_lists")}
    return out


def score_dataset_full(train_seqs, test_seqs, n_items, *, freeze: dict, dataset: str):
    native_cfg = winner_cfg_for(freeze, dataset, unit="native")
    cluster_cfg = winner_cfg_for(freeze, dataset, unit="cluster")
    # Registered: co-occurrence map uses primary cluster winner config (no retune)
    print(f"  native GRU ×{len(GRU_TRAIN_SEEDS)}…", flush=True)
    native = score_unit_multisseed(
        train_seqs, test_seqs, n_items, winner_cfg=native_cfg, unit_name="native"
    )

    dual = json.loads((PHASE / "dual_clustering.json").read_text())
    packs = {"native": native}

    if dataset == "xes3g5m":
        path = freeze["clustering"][dataset]["item_to_cluster_path"]
        item_to_c = load_cluster_map(path)
        tr_c = map_seqs_preserve_length(train_seqs, item_to_c)
        te_c = map_seqs_preserve_length(test_seqs, item_to_c)
        for i, (a, b) in enumerate(zip(test_seqs, te_c)):
            if len(a) != len(b):
                raise RuntimeError(f"{dataset}: length mismatch at test seq {i}")
        tr_r, te_r, n_c = remap_pair(tr_c, te_c)
        print(f"  cluster cooc GRU ×{len(GRU_TRAIN_SEEDS)} (n={n_c})…", flush=True)
        packs["cluster_cooccurrence"] = score_unit_multisseed(
            tr_r, te_r, n_c, winner_cfg=cluster_cfg, unit_name="cluster_cooccurrence"
        )
        assert_native_cluster_alignment(native, packs["cluster_cooccurrence"], dataset)
    else:
        dinfo = dual["datasets"][dataset]
        for method, key in (("text", "cluster_text"), ("cooccurrence", "cluster_cooccurrence")):
            path = dinfo[method]["map_path"]
            item_to_c = load_cluster_map(path)
            tr_c = map_seqs_preserve_length(train_seqs, item_to_c)
            te_c = map_seqs_preserve_length(test_seqs, item_to_c)
            for i, (a, b) in enumerate(zip(test_seqs, te_c)):
                if len(a) != len(b):
                    raise RuntimeError(f"{dataset}/{method}: length mismatch at test seq {i}")
            tr_r, te_r, n_c = remap_pair(tr_c, te_c)
            print(
                f"  cluster {method} GRU ×{len(GRU_TRAIN_SEEDS)} (n={n_c}); "
                f"config=primary_cluster_winner…",
                flush=True,
            )
            packs[key] = score_unit_multisseed(
                tr_r, te_r, n_c, winner_cfg=cluster_cfg, unit_name=f"cluster_{method}"
            )
            assert_native_cluster_alignment(native, packs[key], f"{dataset}/{method}")

    # Secondary uses freeze primary cluster map (= text for Junyi/ASSIST, cooc for XES)
    primary_cluster = packs.get("cluster_text") or packs["cluster_cooccurrence"]
    # secondary m filled later after family size known — placeholder with m_planned
    secondary = secondary_common_nonrepeat(native, primary_cluster, m=M_PLANNED)

    return packs, secondary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch-test", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["junyi_timed", "assistments", "xes3g5m"])
    args = ap.parse_args()

    freeze = load_freeze()
    out = {
        "schema": "freeze_partition_replication_v3",
        "freeze_seed": FREEZE_SEED,
        "gru_train_seeds": list(GRU_TRAIN_SEEDS),
        "catalog_n_items": CATALOG_N_ITEMS,
        "max_windows": MAX_WINDOWS,
        "gru_hidden_rule": "max(64, emb*2) — matches freeze",
        "dropout_policy": "GRUProbe ignores dropout; freeze dropout grid is a no-op for GRU",
        "recall_at_tie_rule": "among equal scores, lower item index wins (stable lexsort)",
        "recency_rule": (
            "0 on advance for all/nonrepeat; N/A only for advance-slice display/ordering"
        ),
        "n_boot_confirmatory": N_BOOT,
        "m_planned": M_PLANNED,
        "touch_test": args.touch_test,
        "junyi_pipeline_SEED_not_used": SEED,
        "datasets": {},
    }

    if args.touch_test:
        check_touch_lock()

    def build_junyi():
        seqs, uids = build_learner_sequences_junyi_timed(max_users=6000, min_len=12, max_len=200)
        return seqs, uids, CATALOG_N_ITEMS["junyi_timed"]

    def build_assist():
        seqs, uids, n_from_data = build_learner_sequences_assist()
        n_cat = CATALOG_N_ITEMS["assistments"]
        assert max(max(x for x, _ in s) for s in seqs) + 1 <= n_cat
        assert n_from_data == n_cat
        return seqs, uids, n_cat

    def build_xes():
        seqs, uids, _n = build_learner_sequences_xes()
        return seqs, uids, CATALOG_N_ITEMS["xes3g5m"]

    builders = {
        "junyi_timed": build_junyi,
        "assistments": build_assist,
        "xes3g5m": build_xes,
    }

    scored_packs = {}
    secondaries = {}

    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        seqs, uids, n_items = builders[ds]()
        # catalog size for output dim
        max_id = max(max(x for x, _ in s) for s in seqs)
        assert max_id < n_items, f"{ds}: max item id {max_id} >= catalog {n_items}"
        tr_i, va_i, te_i, split_meta = verify_and_split(ds, uids, freeze)
        train = [seqs[i] for i in list(tr_i) + list(va_i)]
        test = [seqs[i] for i in te_i]
        print("  hashes OK; n_test", split_meta["n_test"], "touch", args.touch_test, "n_items", n_items, flush=True)

        if not args.touch_test:
            out["datasets"][ds] = {
                "split": split_meta,
                "n_items_catalog": n_items,
                "result": {
                    "scored": False,
                    "reason": "pass --touch-test only after prereg-v1 tag",
                    "gru_train_seeds": list(GRU_TRAIN_SEEDS),
                },
            }
            continue

        packs, secondary = score_dataset_full(
            train, test, n_items, freeze=freeze, dataset=ds
        )
        scored_packs[ds] = packs
        secondaries[ds] = secondary

        # sidecar per-seq (JSON-safe)
        side = {
            "native": packs["native"]["per_sequence_r5"],
            "cluster_text": (packs.get("cluster_text") or {}).get("per_sequence_r5"),
            "cluster_cooccurrence": packs["cluster_cooccurrence"]["per_sequence_r5"],
        }
        side_path = PHASE / f"replication_per_seq_r5_{ds}.json"
        side_path.write_text(json.dumps(side) + "\n")
        out["datasets"][ds] = {
            "split": split_meta,
            "n_items_catalog": n_items,
            "native_winner_cfg": winner_cfg_for(freeze, ds, unit="native"),
            "cluster_winner_cfg": winner_cfg_for(freeze, ds, unit="cluster"),
            "cooc_uses_primary_cluster_config": True,
            "units": {k: strip_arrays(v) for k, v in packs.items()},
            "per_sequence_r5_path": str(side_path.relative_to(ROOT)),
        }
        print("  wrote", side_path, flush=True)

    if args.touch_test and scored_packs:
        claims = assemble_primary_claims(scored_packs)
        m = claims["eligibility_and_family"]["m"]
        # recompute secondary at actual m
        for ds in scored_packs:
            native = scored_packs[ds]["native"]
            primary_c = scored_packs[ds].get("cluster_text") or scored_packs[ds]["cluster_cooccurrence"]
            secondaries[ds] = secondary_common_nonrepeat(native, primary_c, m=m if m >= 1 else 1)
            out["datasets"][ds]["secondary_common_nonrepeat"] = secondaries[ds]
            # primary-only flip note
            dc = claims["dataset_claims"][ds]
            sec = secondaries[ds]
            sn = sec["native_gru_minus_markov"].get("verdict")
            sc = sec["cluster_gru_minus_markov"].get("verdict")
            primary_flip = dc["satisfies_primary_claim"]
            sec_flip = (
                sn in ("supported_A", "supported_B")
                and sc in ("supported_A", "supported_B")
                and sn != sc
            )
            out["datasets"][ds]["flip_reporting"] = {
                "primary_flip": primary_flip,
                "secondary_flip": sec_flip,
                "primary_only_flip": bool(primary_flip and not sec_flip),
                "note": "A flip present only in the primary slices is reported as primary-only",
            }
        out["primary_claims"] = claims
        write_touch_lock()
        print("wrote touch lock", TOUCH_LOCK, flush=True)

    path = PHASE / "freeze_partition_replication.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)
    if not args.touch_test:
        print(
            "VERIFY-ONLY complete. After prereg-v1 + OSF: "
            "PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test",
            flush=True,
        )


if __name__ == "__main__":
    main()
