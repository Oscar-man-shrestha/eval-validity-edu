"""Follow-up 3: forgetting redo, XES KC methods, concept table, exact-dup chance,
rank-reversal study. Numbers → JSON only.

Run: python3 run_followup3.py
Then: python3 scripts/render_external_review.py
      python3 scripts/render_team_report.py
      python3 scripts/build_dataflow_report.py
      python3 -m pytest tests/test_leakage_guards.py -q
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import build_heads, write_json
from run_claude_fixes import N_BOOT, seq_hash
from run_valid_eval import EVAL_POS, EVAL_SEQS, TIMED_NPZ, recall_at, score_features

PHASE = OUT / "phases"
US_PER_DAY = 1_000_000 * 86_400
ROOT = Path(__file__).resolve().parent
ASSIST_CSV = ROOT / "data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv"
EXERCISE_CSV = ROOT / "data/junyi_raw/junyi/junyi_Exercise_table.csv"
XES_Q = ROOT / "data/xes3g5m/XES3G5M/question_level"
DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


def mark_forgetting_unresolved() -> None:
    """Until the redo finishes, do not call the old result a null."""
    path = PHASE / "followup2_forgetting.json"
    if path.exists():
        old = json.loads(path.read_text())
        old["claim"] = "unresolved"
        old["claim_superseded_by"] = "followup3_forgetting.json"
        old["note"] = (
            "Follow-up 3: previous day-vs-success result must not be cited as a null "
            "until gap>=1-only redo completes."
        )
        write_json("followup2_forgetting.json", old)
    summary_path = PHASE / "review_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary.setdefault("claims", {})["forgetting"] = {
            "status": "unresolved",
            "pending": "followup3_forgetting.json",
        }
        write_json("review_summary.json", summary)


# ---------------------------------------------------------------------------
# 1. Forgetting redo
# ---------------------------------------------------------------------------

def collect_gap_ge1_rows():
    z = np.load(TIMED_NPZ)
    u, c, t, y = z["user_id"], z["concept"], z["t_us"], z["correct"]
    day = t.astype(np.int64) // US_PER_DAY
    by_learner: dict[int, list] = defaultdict(list)
    i = 0
    n = len(u)
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        uid = int(u[i])
        last_day: dict[int, int] = {}
        stats: dict[int, list[int]] = {}
        for idx in range(i, j):
            concept = int(c[idx])
            correct = int(y[idx])
            d = int(day[idx])
            if concept in last_day:
                gap = d - last_day[concept]
                if gap >= 1:
                    a, s = stats[concept]
                    rate = s / a if a else 0.5
                    by_learner[uid].append(
                        {
                            "gap": float(gap),
                            "log1p_gap": float(np.log1p(gap)),
                            "success_rate": float(rate),
                            "y": float(correct),
                            "gap_bin": (
                                0 if gap <= 3 else 1 if gap <= 14 else 2  # 1-3 / 4-14 / 15+
                            ),
                        }
                    )
            a, s = stats.get(concept, [0, 0])
            stats[concept] = [a + 1, s + correct]
            last_day[concept] = d
        i = j
    return by_learner


def paired_learner_auc_delta(packs_y, packs_p0, packs_p1, rng, n_boot=N_BOOT) -> dict:
    y_all = np.concatenate(packs_y)
    p0_all = np.concatenate(packs_p0)
    p1_all = np.concatenate(packs_p1)
    point = None
    if len(np.unique(y_all)) >= 2:
        point = float(roc_auc_score(y_all, p1_all) - roc_auc_score(y_all, p0_all))
    L = len(packs_y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, L, L)
        yb = np.concatenate([packs_y[i] for i in idx])
        if len(np.unique(yb)) < 2:
            continue
        p0b = np.concatenate([packs_p0[i] for i in idx])
        p1b = np.concatenate([packs_p1[i] for i in idx])
        boots.append(float(roc_auc_score(yb, p1b) - roc_auc_score(yb, p0b)))
    if not boots:
        return {"mean": point, "ci": [None, None], "n_learners": L}
    arr = np.asarray(boots)
    return {
        "mean": point,
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_learners": L,
        "n_rows": int(len(y_all)),
        "n_boot": n_boot,
        "n_boot_kept": len(boots),
        "ci_excludes_zero": bool(np.quantile(arr, 0.025) > 0 or np.quantile(arr, 0.975) < 0),
        "statistic": "pooled_auc_delta_bootstrap_over_learners",
    }


def run_forgetting_redo() -> dict:
    print("1 forgetting redo (gap>=1 only)…", flush=True)
    by_learner = collect_gap_ge1_rows()
    learners = np.array(sorted(by_learner.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(learners)
    n_te = max(1, int(0.2 * len(learners)))
    te_u = set(learners[:n_te].tolist())
    tr_u = set(learners[n_te:].tolist())

    def stack(uids):
        rows = []
        for uid in uids:
            rows.extend(by_learner[uid])
        if not rows:
            return None
        return {
            "success": np.array([r["success_rate"] for r in rows]),
            "log_gap": np.array([r["log1p_gap"] for r in rows]),
            "gap_bin": np.array([r["gap_bin"] for r in rows]),
            "y": np.array([r["y"] for r in rows]),
            "gap": np.array([r["gap"] for r in rows]),
            "uid": np.array([uid for uid in uids for _ in by_learner[uid]]),
        }

    tr = stack(tr_u)
    # per-learner test packs
    te_rows = {uid: by_learner[uid] for uid in te_u if by_learner[uid]}
    n_tr = len(tr["y"])
    n_te_rows = sum(len(v) for v in te_rows.values())

    # fit three models on TRAIN gap>=1 only
    # A: success_rate only (fitted logistic)
    m_succ = LogisticRegression(max_iter=500).fit(tr["success"].reshape(-1, 1), tr["y"])
    # B: success + log1p(gap)
    x_log = np.column_stack([tr["success"], tr["log_gap"]])
    m_log = LogisticRegression(max_iter=500).fit(x_log, tr["y"])
    # C: success + gap bin dummies
    bins_oh = np.eye(3)[tr["gap_bin"]]
    x_bin = np.column_stack([tr["success"], bins_oh])
    m_bin = LogisticRegression(max_iter=500).fit(x_bin, tr["y"])

    def predict_packs(model, feat_fn):
        ys, ps = [], []
        for uid, rows in te_rows.items():
            y = np.array([r["y"] for r in rows])
            x = feat_fn(rows)
            p = model.predict_proba(x)[:, 1]
            ys.append(y)
            ps.append(p)
        return ys, ps

    def feat_succ(rows):
        return np.array([r["success_rate"] for r in rows]).reshape(-1, 1)

    def feat_log(rows):
        return np.column_stack(
            [
                [r["success_rate"] for r in rows],
                [r["log1p_gap"] for r in rows],
            ]
        )

    def feat_bin(rows):
        gb = np.array([r["gap_bin"] for r in rows])
        return np.column_stack([[r["success_rate"] for r in rows], np.eye(3)[gb]])

    y_ps, p_succ = predict_packs(m_succ, feat_succ)
    _, p_log = predict_packs(m_log, feat_log)
    _, p_bin = predict_packs(m_bin, feat_bin)

    # AUCs
    y_all = np.concatenate(y_ps)
    auc = {}
    for name, p in [("success_only", p_succ), ("success_plus_log1p_gap", p_log), ("success_plus_gap_bins", p_bin)]:
        p_all = np.concatenate(p)
        auc[name] = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) >= 2 else None

    deltas = {
        "log1p_gap_minus_success": paired_learner_auc_delta(y_ps, p_succ, p_log, rng),
        "gap_bins_minus_success": paired_learner_auc_delta(y_ps, p_succ, p_bin, rng),
        "gap_bins_minus_log1p_gap": paired_learner_auc_delta(y_ps, p_log, p_bin, rng),
    }

    # accuracy by gap bin within 5 success-rate strata (on test rows)
    all_rows = [r for rows in te_rows.values() for r in rows]
    succ = np.array([r["success_rate"] for r in all_rows])
    # 5 quantile strata
    try:
        qs = np.quantile(succ, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ensure unique edges
        qs = np.unique(qs)
        if len(qs) < 3:
            strata_ids = np.zeros(len(succ), dtype=int)
            n_strata = 1
        else:
            strata_ids = np.digitize(succ, qs[1:-1], right=True)
            n_strata = int(strata_ids.max()) + 1
    except Exception:
        strata_ids = np.zeros(len(succ), dtype=int)
        n_strata = 1
    gap_names = {0: "1_3_days", 1: "4_14_days", 2: "15p_days"}
    strata_table = []
    for s in range(n_strata):
        for gb, gname in gap_names.items():
            mask = (strata_ids == s) & (np.array([r["gap_bin"] for r in all_rows]) == gb)
            if mask.sum() == 0:
                strata_table.append(
                    {"success_stratum": int(s), "gap_bin": gname, "n": 0, "accuracy": None}
                )
                continue
            yy = np.array([r["y"] for r in all_rows])[mask]
            strata_table.append(
                {
                    "success_stratum": int(s),
                    "gap_bin": gname,
                    "n": int(mask.sum()),
                    "accuracy": float(yy.mean()),
                    "mean_success_rate": float(succ[mask].mean()),
                }
            )

    # status from log1p vs success
    d = deltas["log1p_gap_minus_success"]
    if d["mean"] is not None and d.get("ci_excludes_zero") and d["mean"] > 0:
        status = "log_gap_helps"
    elif d["mean"] is not None and d.get("ci_excludes_zero") and d["mean"] < 0:
        status = "log_gap_hurts_vs_success"
    else:
        status = "unresolved"

    out = {
        "protocol": {
            "fit_and_test_rows": "gap >= 1 calendar day only",
            "split": "learner-level 80/20",
            "seed": SEED,
            "gap_feature": "log1p(gap_days)",
            "models": [
                "logistic(success_rate)",
                "logistic(success_rate, log1p_gap)",
                "logistic(success_rate, gap_bin_dummies)",
            ],
        },
        "n_train_learners": len(tr_u),
        "n_test_learners": len(te_rows),
        "n_train_rows": int(n_tr),
        "n_test_rows": int(n_te_rows),
        "auc": auc,
        "paired_deltas": deltas,
        "accuracy_by_gap_bin_within_success_strata": strata_table,
        "n_success_strata": n_strata,
        "status": status,
        "status_rule": (
            "log_gap_helps if paired CI of (AUC_log - AUC_success) entirely above 0; "
            "log_gap_hurts_vs_success if entirely below 0; else unresolved"
        ),
        "supersedes": "followup2_forgetting.json (must not cite old result as null)",
    }
    write_json("followup3_forgetting.json", out)
    print("forgetting status", status, "auc", auc, "delta_log", d, flush=True)
    return out


# ---------------------------------------------------------------------------
# 2–3. XES KC methods + concept-level table
# ---------------------------------------------------------------------------

def load_topic_map() -> dict[int, str]:
    slug_to_idx = {}
    for line in (DATA / "vertex_id2idx").read_text().splitlines():
        if not line.strip():
            continue
        slug, idx = line.rsplit(",", 1)
        slug_to_idx[slug.strip()] = int(idx)
    ex = pd.read_csv(EXERCISE_CSV)
    out = {}
    for _, row in ex.iterrows():
        slug = str(row["name"]).strip()
        if slug in slug_to_idx:
            out[slug_to_idx[slug]] = str(row["topic"]) if pd.notna(row["topic"]) else "unknown"
    return out


def share_and_mix(seq_labels: list[list]) -> dict:
    hit = tot = 0
    mix = {"continue": 0, "revisit": 0, "advance": 0}
    n_mix = 0
    for labels in seq_labels:
        seen = set()
        last = None
        for lab in labels:
            if last is not None:
                tot += 1
                if lab == last:
                    hit += 1
                    mix["continue"] += 1
                elif lab in seen:
                    mix["revisit"] += 1
                else:
                    mix["advance"] += 1
                n_mix += 1
            seen.add(lab)
            last = lab
    return {
        "share_next_equals_last": {"share": hit / max(tot, 1), "n": tot, "hits": hit},
        "three_way_mix": {
            k: {"count": v, "frac": v / max(n_mix, 1)} for k, v in mix.items()
        }
        | {"n": n_mix},
        "n_sequences": len(seq_labels),
    }


def _parse_int_list(s: str) -> list[int]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part and part != "-1":
            out.append(int(part))
    return out


def _parse_concepts(s: str) -> list[list[int]]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part or part == "-1":
            continue
        out.append([int(x) for x in part.split("_") if x not in ("", "-1")])
    return out


def run_xes_kc_methods() -> dict:
    print("2 XES KC repetition methods…", flush=True)
    # load parquet if available for is_repeat; else reconstruct from quelevel
    frames = []
    for name in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv"):
        frames.append(pd.read_csv(XES_Q / name))
    df = pd.concat(frames, ignore_index=True)

    # Method A: question → primary KC (first KC), then next==last on that KC stream
    method_a_seqs = []
    # Method B: expand KCs but drop repeats within same question (keep only first KC of each question)
    method_b_seqs = []

    best = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        cons = _parse_concepts(row["concepts"])
        if len(qs) < 12:
            continue
        if uid not in best or len(qs) > best[uid][0]:
            best[uid] = (len(qs), qs, cons)

    for uid, (_, qs, cons) in best.items():
        primary = []
        first_kc_only = []
        for i, q in enumerate(qs):
            kcs = cons[i] if i < len(cons) else []
            if not kcs:
                continue
            primary.append(kcs[0])
            # method B: one KC token per question (drop is_repeat-style extras)
            first_kc_only.append(kcs[0])
        if len(primary) >= 12:
            method_a_seqs.append(primary)
            method_b_seqs.append(first_kc_only)

    # Also method B-alt from HF parquet is_repeat if present
    hf = ROOT / "data/xes3g5m/hf/train.parquet"
    method_b_is_repeat = None
    if hf.exists():
        try:
            pq = pd.read_parquet(hf)
            seqs = []
            for _, row in pq.iterrows():
                concepts = np.asarray(row["concepts"])
                masks = np.asarray(row["selectmasks"])
                repeats = np.asarray(row["is_repeat"])
                keep = (masks == 1) & (repeats == 0)
                labs = [int(x) for x in concepts[keep]]
                if len(labs) >= 12:
                    seqs.append(labs[:200])
            method_b_is_repeat = share_and_mix(seqs)
            method_b_is_repeat["method"] = "drop_is_repeat_eq_1_rows_from_kc_expanded_windows"
            method_b_is_repeat["n_sequences"] = len(seqs)
        except Exception as e:
            method_b_is_repeat = {"error": str(e)}

    out = {
        "method_a_question_then_primary_kc": {
            "method": "map each question to its first KC, then share(next==last) on KC ids",
            **share_and_mix(method_a_seqs),
        },
        "method_b_drop_multi_kc_extras": {
            "method": "one KC token per question (equivalent to dropping is_repeat extras when expanding)",
            **share_and_mix(method_b_seqs),
        },
        "method_b_drop_is_repeat_parquet": method_b_is_repeat,
        "primary_reported": "both method_a and method_b; prefer method_a for concept-level table",
    }
    write_json("followup3_xes_kc_repetition.json", out)
    print(
        "xes A",
        out["method_a_question_then_primary_kc"]["share_next_equals_last"]["share"],
        "B",
        out["method_b_drop_multi_kc_extras"]["share_next_equals_last"]["share"],
        flush=True,
    )
    return out


def assist_skill_sequences() -> list[list]:
    raw = pd.read_csv(
        ASSIST_CSV,
        encoding="ISO-8859-1",
        low_memory=False,
        usecols=["order_id", "user_id", "skill_id"],
    )
    raw = raw.dropna(subset=["user_id", "order_id"])
    raw["skill_id"] = raw["skill_id"].fillna(-1).astype(int)
    g = (
        raw.groupby("order_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            skills=("skill_id", lambda s: tuple(sorted({int(x) for x in s if int(x) >= 0}))),
        )
        .reset_index()
    )
    g["skill_key"] = g["skills"].map(lambda t: "_".join(map(str, t)) if t else "none")
    g = g.sort_values(["user_id", "order_id"])
    seqs = []
    for _, grp in g.groupby("user_id", sort=False):
        labs = grp["skill_key"].tolist()
        if len(labs) >= 12:
            seqs.append(labs[:200])
    return seqs


def run_concept_table(xes_kc: dict) -> dict:
    print("3 concept-level repetition table…", flush=True)
    topic_of = load_topic_map()
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")
    junyi_topic = [[topic_of.get(c, "unknown") for c, _ in s] for s in train + test]
    assist = assist_skill_sequences()
    xes_a = None
    # rebuild seqs from xes result stats already computed — reload method A seqs via share file
    # use method A primary KC from JSON n_sequences; recompute seqs quickly
    frames = [
        pd.read_csv(XES_Q / n)
        for n in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    best = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        cons = _parse_concepts(row["concepts"])
        if len(qs) < 12:
            continue
        if uid not in best or len(qs) > best[uid][0]:
            primary = []
            for i in range(len(qs)):
                kcs = cons[i] if i < len(cons) else []
                if kcs:
                    primary.append(kcs[0])
            if len(primary) >= 12:
                best[uid] = (len(qs), primary)
    xes_seqs = [v[1] for v in best.values()]

    table = {
        "level": "concept",
        "note": "exercise/question-level rows removed from cross-dataset comparison",
        "rows": [
            {"dataset_unit": "Junyi topic", **share_and_mix(junyi_topic)},
            {"dataset_unit": "ASSISTments skill", **share_and_mix(assist)},
            {
                "dataset_unit": "XES3G5M KC (method A: question→primary KC)",
                **share_and_mix(xes_seqs),
                "method_ref": "followup3_xes_kc_repetition.method_a_question_then_primary_kc",
            },
            {
                "dataset_unit": "XES3G5M KC (method B: drop multi-KC extras)",
                **xes_kc["method_b_drop_multi_kc_extras"],
            },
        ],
    }
    # also store method B is_repeat if available
    if xes_kc.get("method_b_drop_is_repeat_parquet") and "share_next_equals_last" in (
        xes_kc.get("method_b_drop_is_repeat_parquet") or {}
    ):
        table["rows"].append(
            {
                "dataset_unit": "XES3G5M KC (method B parquet is_repeat=0)",
                **xes_kc["method_b_drop_is_repeat_parquet"],
            }
        )
    write_json("followup3_concept_repetition.json", table)
    write_json("followup2_cross_dataset.json", {"columns": ["concept-level only"], "rows": table["rows"], "replaced_by": "followup3_concept_repetition.json"})
    print("concept table shares", [r["share_next_equals_last"]["share"] for r in table["rows"][:3]], flush=True)
    return table


# ---------------------------------------------------------------------------
# 4. Exact-duplicate chance baseline
# ---------------------------------------------------------------------------

def run_exact_dup_chance() -> dict:
    print("4 exact-duplicate chance…", flush=True)
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")
    rng = np.random.default_rng(SEED)
    tr_hashes = [seq_hash(s) for s in train]
    te_hashes = {seq_hash(s) for s in test}
    tr_set = set(tr_hashes)
    real = te_hashes & tr_set

    n_pseudo = min(6290, len(train))
    idx = rng.choice(len(train), size=n_pseudo, replace=False)
    pseudo = {tr_hashes[i] for i in idx}
    remain = {tr_hashes[i] for i in range(len(train)) if i not in set(idx.tolist())}
    chance = pseudo & remain

    out = {
        "real_train_test_exact": {
            "n_overlap_hashes": len(real),
            "n_test_sequences": len(test),
            "n_test_unique_hashes": len(te_hashes),
        },
        "chance_baseline_pseudo_test": {
            "n_pseudo_test": n_pseudo,
            "n_remaining_train": len(remain),
            "n_overlap_hashes": len(chance),
            "seed": SEED,
            "rule": "sample 6290 train sequences; exact-hash collide against remaining train",
        },
        "comparison": {
            "real_overlap": len(real),
            "chance_overlap": len(chance),
            "real_minus_chance": len(real) - len(chance),
        },
    }
    write_json("followup3_exact_dup_chance.json", out)
    print(out["comparison"], flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. Rank-reversal study
# ---------------------------------------------------------------------------

def mrr_at(scores: np.ndarray, label: int) -> float:
    order = np.argsort(-scores)
    hits = np.where(order == label)[0]
    return 0.0 if len(hits) == 0 else 1.0 / float(hits[0] + 1)


def seq_boot_mean(values_per_seq, rng, n_boot=N_BOOT):
    means = np.array([np.mean(v) for v in values_per_seq if v], dtype=float)
    if len(means) == 0:
        return {"mean": None, "ci": [None, None], "n_sequences": 0}
    boots = [float(means[rng.integers(0, len(means), len(means))].mean()) for _ in range(n_boot)]
    arr = np.asarray(boots)
    return {
        "mean": float(means.mean()),
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_sequences": int(len(means)),
        "n_boot": n_boot,
    }


def paired_delta(a_per_seq, b_per_seq, rng, n_boot=N_BOOT):
    d = []
    for a, b in zip(a_per_seq, b_per_seq):
        if a and b:
            d.append(float(np.mean(a) - np.mean(b)))
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return {"mean": None, "ci": [None, None]}
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)]
    arr = np.asarray(boots)
    return {
        "mean": float(d.mean()),
        "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "n_sequences": int(len(d)),
        "ci_excludes_zero": bool(np.quantile(arr, 0.025) > 0 or np.quantile(arr, 0.975) < 0),
    }


def action_slice(label, history_set, last):
    if label == last:
        return "continue"
    if label in history_set:
        return "revisit"
    return "advance"


class GRUProbe(nn.Module):
    """Local GRU-based next-item probe — not a published GRU4Rec reproduction."""

    def __init__(self, n_items, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_items)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)
        return self.out(h.squeeze(0))


class AttnProbe(nn.Module):
    """Local attention-based next-item probe — not a published SASRec reproduction."""

    def __init__(self, n_items, emb=32, n_heads=2, max_len=50, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb)
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=n_heads, dim_feedforward=64, batch_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(emb, n_items)
        self.max_len = max_len

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.emb(x) + self.pos(pos)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask=mask)
        return self.out(h[:, -1, :])


# Back-compat aliases (do not use in new code / docs)
GRU4Rec = GRUProbe
SASRec = AttnProbe


def train_seq_model(model, train_seqs, n_items, epochs=3, batch=64, lr=1e-3, max_len=50):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # build windows
    pairs = []
    for seq in train_seqs:
        ids = [x + 1 for x, _ in seq]  # 1-based for padding
        for t in range(1, len(ids)):
            ctx = ids[max(0, t - max_len) : t]
            pairs.append((ctx, seq[t][0]))  # label is 0-based item id
    rng = np.random.default_rng(SEED)
    rng.shuffle(pairs)
    pairs = pairs[:80_000]
    model.train()
    for _ in range(epochs):
        for i in range(0, len(pairs), batch):
            batch_pairs = pairs[i : i + batch]
            if not batch_pairs:
                continue
            max_t = max(len(c) for c, _ in batch_pairs)
            x = torch.zeros(len(batch_pairs), max_t, dtype=torch.long, device=DEVICE)
            y = torch.tensor([lab for _, lab in batch_pairs], dtype=torch.long, device=DEVICE)
            for bi, (ctx, _) in enumerate(batch_pairs):
                x[bi, -len(ctx) :] = torch.tensor(ctx, dtype=torch.long)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def score_seq_model(model, seq_prefix_items, n_items, max_len=50):
    ids = [x + 1 for x in seq_prefix_items[-max_len:]]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits = model(x).squeeze(0).cpu().numpy()
    if len(logits) < n_items:
        pad = np.full(n_items - len(logits), -1e9)
        logits = np.concatenate([logits, pad])
    return logits[:n_items]


def build_learner_sequences_junyi_timed(max_users=6000, min_len=12, max_len=200):
    z = np.load(TIMED_NPZ)
    u, c, y = z["user_id"], z["concept"], z["correct"]
    seqs, uids = [], []
    i = 0
    n = len(u)
    while i < n and len(seqs) < max_users:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        if j - i >= min_len:
            seq = [(int(c[k]), int(y[k])) for k in range(i, min(j, i + max_len))]
            if len(seq) >= min_len:
                seqs.append(seq)
                uids.append(int(u[i]))
        i = j
    return seqs, uids


def build_learner_sequences_assist(min_len=12, max_len=200):
    raw = pd.read_csv(
        ASSIST_CSV,
        encoding="ISO-8859-1",
        low_memory=False,
        usecols=["order_id", "user_id", "skill_id", "correct"],
    )
    raw = raw.dropna(subset=["user_id", "order_id", "correct"])
    raw["skill_id"] = raw["skill_id"].fillna(-1).astype(int)
    g = (
        raw.groupby("order_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            correct=("correct", "first"),
            skills=("skill_id", lambda s: tuple(sorted({int(x) for x in s if int(x) >= 0}))),
        )
        .reset_index()
    )
    g["item"] = g["skills"].map(lambda t: "_".join(map(str, t)) if t else "none")
    g = g.sort_values(["user_id", "order_id"])
    # map items to ids
    vocab = {k: i for i, k in enumerate(sorted(g["item"].unique()))}
    seqs, uids = [], []
    for uid, grp in g.groupby("user_id", sort=False):
        seq = [(vocab[it], int(c)) for it, c in zip(grp["item"], grp["correct"])]
        if len(seq) >= min_len:
            seqs.append(seq[:max_len])
            uids.append(int(uid))
    return seqs, uids, len(vocab)


def build_learner_sequences_xes(min_len=12, max_len=200):
    frames = [
        pd.read_csv(XES_Q / n)
        for n in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    best = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        rs = _parse_int_list(row["responses"]) if "responses" in row else [0] * len(qs)
        if len(qs) < min_len:
            continue
        if uid not in best or len(qs) > len(best[uid]):
            seq = [(q, int(rs[i]) if i < len(rs) else 0) for i, q in enumerate(qs[:max_len])]
            best[uid] = seq
    # remap question ids to dense
    all_q = sorted({q for seq in best.values() for q, _ in seq})
    vocab = {q: i for i, q in enumerate(all_q)}
    seqs, uids = [], []
    for uid, seq in best.items():
        seqs.append([(vocab[q], y) for q, y in seq])
        uids.append(uid)
    return seqs, uids, len(vocab)


def popularity_scores(train_seqs, n_items):
    cnt = np.zeros(n_items, dtype=float)
    for seq in train_seqs:
        for it, _ in seq:
            if 0 <= it < n_items:
                cnt[it] += 1
    return cnt


def evaluate_methods_on_split(name, train_seqs, test_seqs, n_items, sample_idx, rng, ragr_pack=None):
    """Evaluate popularity, recency, GRU/attention probes, RAGR on fixed test sample indices."""
    print(f"  rank-eval {name} n_items={n_items} n_test_pool={len(test_seqs)}…", flush=True)
    pop = popularity_scores(train_seqs, n_items)
    # train neural probes (not published GRU4Rec/SASRec)
    gru = train_seq_model(GRUProbe(n_items), train_seqs, n_items, epochs=2)
    sas = train_seq_model(AttnProbe(n_items), train_seqs, n_items, epochs=2)

    methods = ["popularity", "recency", "gru_probe", "attn_probe", "ragr"]
    metrics = ["r5", "r1", "mrr"]
    slices = ["all", "revisit", "advance"]
    buckets = {m: {sl: {met: [] for met in metrics} for sl in slices} for m in methods}

    idx = sample_idx
    for si in idx:
        seq = test_seqs[int(si)]
        cuts = np.linspace(8, len(seq) - 1, num=min(EVAL_POS, max(1, len(seq) - 9)), dtype=int)
        local = {m: {sl: {met: [] for met in metrics} for sl in slices} for m in methods}
        for cut in cuts:
            cut = int(cut)
            label = seq[cut][0]
            prefix = [x for x, _ in seq[:cut]]
            last = prefix[-1]
            hist = set(prefix)
            sl = action_slice(label, hist, last)
            # popularity
            scores = {
                "popularity": pop.copy(),
                "recency": np.full(n_items, -1e9),
                "gru_probe": score_seq_model(gru, prefix, n_items),
                "attn_probe": score_seq_model(sas, prefix, n_items),
            }
            # recency: most recent distinct
            distinct = []
            for x in reversed(prefix):
                if x not in distinct:
                    distinct.append(x)
                if len(distinct) >= 5:
                    break
            for rank, item in enumerate(distinct):
                if 0 <= item < n_items:
                    scores["recency"][item] = 5 - rank
            # RAGR
            if ragr_pack is not None:
                scores["ragr"] = ragr_pack["score"](seq, cut)
            else:
                # portable RAGR-lite: mix recency (review) and popularity (advance)
                # gate: higher review if last wrong
                last_y = seq[cut - 1][1]
                p_rev = 0.9 if last_y == 0 else 0.7
                rev = scores["recency"].copy()
                adv = pop.copy()
                # mask: review only seen
                for i in range(n_items):
                    if i not in hist:
                        rev[i] = rev.min() - 1 if np.isfinite(rev.min()) else -1e9
                    else:
                        adv[i] = adv.min() - 1
                # normalize roughly
                def nz(a):
                    a = a.astype(float)
                    if np.allclose(a.max(), a.min()):
                        return np.zeros_like(a)
                    return (a - a.mean()) / (a.std() + 1e-6)

                scores["ragr"] = p_rev * nz(rev) + (1 - p_rev) * nz(adv)

            for m in methods:
                sc = scores[m]
                if label < 0 or label >= n_items:
                    continue
                vals = {
                    "r5": recall_at(sc, label, 5),
                    "r1": recall_at(sc, label, 1),
                    "mrr": mrr_at(sc, label),
                }
                for met, v in vals.items():
                    local[m]["all"][met].append(v)
                    if sl in ("revisit", "advance"):
                        local[m][sl][met].append(v)

        for m in methods:
            for sl in slices:
                for met in metrics:
                    if local[m][sl][met]:
                        buckets[m][sl][met].append(local[m][sl][met])

    # summarize
    summary = {}
    for m in methods:
        summary[m] = {}
        for sl in slices:
            summary[m][sl] = {met: seq_boot_mean(buckets[m][sl][met], rng) for met in metrics}

    # rankings per slice by R@5
    rankings = {}
    reversals = []
    for sl in slices:
        order = sorted(
            methods,
            key=lambda m: (-1e9 if summary[m][sl]["r5"]["mean"] is None else -summary[m][sl]["r5"]["mean"], m),
        )
        rankings[sl] = [{"method": m, "r5": summary[m][sl]["r5"]["mean"]} for m in order]
    # flag rank reversal: order on all vs revisit or advance differs
    all_order = [x["method"] for x in rankings["all"]]
    for sl in ("revisit", "advance"):
        sl_order = [x["method"] for x in rankings[sl]]
        if sl_order != all_order and all(summary[m][sl]["r5"]["mean"] is not None for m in methods):
            reversals.append({"slice": sl, "all_order": all_order, "slice_order": sl_order})

    # paired RAGR - others on all R@5
    paired = {}
    if buckets["ragr"]["all"]["r5"]:
        for m in methods:
            if m == "ragr":
                continue
            paired[f"ragr_minus_{m}_r5"] = paired_delta(
                buckets["ragr"]["all"]["r5"], buckets[m]["all"]["r5"], rng
            )

    return {
        "metrics": summary,
        "rankings_by_slice_r5": rankings,
        "rank_reversals": reversals,
        "paired_ragr_minus_baselines_all_r5": paired,
    }


def make_ragr_junyi():
    names = load_names()
    graph, similar = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]
    bundle = joblib.load(OUT / "serve_models.joblib")
    feat_fn = build_heads(
        embeddings, graph, similar, np.asarray(bundle["popularity"]), parent_lists, bundle["logreg"]
    )

    def score(seq, cut):
        pack = score_features(
            feat_fn, bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"], seq, cut, graph, embeddings
        )
        return pack["mixed"]

    return {"score": score, "n_items": len(names)}


def run_rank_reversal() -> dict:
    print("5 rank-reversal study…", flush=True)
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "eval_cuts": EVAL_POS, "datasets": {}}

    # --- Junyi timed learner-disjoint ---
    seqs, uids = build_learner_sequences_junyi_timed()
    uids = np.array(uids)
    order = np.arange(len(seqs))
    rng.shuffle(order)
    n_te = max(1, int(0.2 * len(order)))
    te_i = order[:n_te]
    tr_i = order[n_te:]
    train_seqs = [seqs[i] for i in tr_i]
    test_seqs = [seqs[i] for i in te_i]
    n_items = max(max(c for c, _ in s) for s in seqs) + 1
    n_eval = min(EVAL_SEQS, len(test_seqs))
    sample_idx = rng.choice(len(test_seqs), size=n_eval, replace=False)
    ragr = make_ragr_junyi()
    # ensure n_items matches
    n_items = max(n_items, ragr["n_items"])
    junyi_res = evaluate_methods_on_split(
        "junyi_timed", train_seqs, test_seqs, n_items, sample_idx, rng, ragr_pack=ragr
    )
    results["datasets"]["junyi_timed"] = {
        "split": {
            "unit": "learner",
            "n_train": len(train_seqs),
            "n_test": len(test_seqs),
            "n_eval_sequences": int(n_eval),
            "sample_indices": sample_idx.tolist(),
            "sample_indices_sha256": hashlib.sha256(sample_idx.astype(np.int64).tobytes()).hexdigest(),
            "train_uids": [int(uids[i]) for i in tr_i[:20]],
            "test_uids_head": [int(uids[i]) for i in te_i[:20]],
        },
        "n_items": n_items,
        **junyi_res,
    }

    # --- ASSISTments ---
    rng2 = np.random.default_rng(SEED)
    a_seqs, a_uids, a_n = build_learner_sequences_assist()
    a_uids = np.array(a_uids)
    order = np.arange(len(a_seqs))
    rng2.shuffle(order)
    n_te = max(1, int(0.2 * len(order)))
    te_i, tr_i = order[:n_te], order[n_te:]
    train_seqs = [a_seqs[i] for i in tr_i]
    test_seqs = [a_seqs[i] for i in te_i]
    n_eval = min(EVAL_SEQS, len(test_seqs))
    sample_idx = rng2.choice(len(test_seqs), size=n_eval, replace=False)
    assist_res = evaluate_methods_on_split(
        "assistments", train_seqs, test_seqs, a_n, sample_idx, rng2, ragr_pack=None
    )
    results["datasets"]["assistments"] = {
        "split": {
            "unit": "learner",
            "n_train": len(train_seqs),
            "n_test": len(test_seqs),
            "n_eval_sequences": int(n_eval),
            "sample_indices": sample_idx.tolist(),
            "sample_indices_sha256": hashlib.sha256(sample_idx.astype(np.int64).tobytes()).hexdigest(),
        },
        "n_items": a_n,
        "ragr_note": "portable RAGR-lite (no expert DAG)",
        **assist_res,
    }

    # --- XES3G5M ---
    rng3 = np.random.default_rng(SEED)
    x_seqs, x_uids, x_n = build_learner_sequences_xes()
    order = np.arange(len(x_seqs))
    rng3.shuffle(order)
    n_te = max(1, int(0.2 * len(order)))
    te_i, tr_i = order[:n_te], order[n_te:]
    train_seqs = [x_seqs[i] for i in tr_i]
    test_seqs = [x_seqs[i] for i in te_i]
    n_eval = min(400, len(test_seqs))  # XES slower
    sample_idx = rng3.choice(len(test_seqs), size=n_eval, replace=False)
    xes_res = evaluate_methods_on_split(
        "xes3g5m", train_seqs, test_seqs, x_n, sample_idx, rng3, ragr_pack=None
    )
    results["datasets"]["xes3g5m"] = {
        "split": {
            "unit": "uid",
            "n_train": len(train_seqs),
            "n_test": len(test_seqs),
            "n_eval_sequences": int(n_eval),
            "sample_indices": sample_idx.tolist(),
            "sample_indices_sha256": hashlib.sha256(sample_idx.astype(np.int64).tobytes()).hexdigest(),
        },
        "n_items": x_n,
        "ragr_note": "portable RAGR-lite (no expert DAG)",
        **xes_res,
    }

    write_json("followup3_rank_reversal.json", results)
    for ds, payload in results["datasets"].items():
        print(ds, "all ranking", payload["rankings_by_slice_r5"]["all"], "reversals", payload["rank_reversals"], flush=True)
    return results


# ---------------------------------------------------------------------------
# summary + main
# ---------------------------------------------------------------------------

def write_summary(forget, xes_kc, concept, chance, ranks) -> None:
    summary = json.loads((PHASE / "review_summary.json").read_text()) if (PHASE / "review_summary.json").exists() else {"claims": {}}
    summary.setdefault("claims", {})
    summary["claims"]["forgetting"] = {
        "status": forget["status"],
        "protocol": forget["protocol"],
        "paired_deltas": forget["paired_deltas"],
        "auc": forget["auc"],
        "source": "followup3_forgetting.json",
    }
    summary["claims"]["concept_repetition"] = {
        "rows": [
            {
                "dataset_unit": r["dataset_unit"],
                "share": r["share_next_equals_last"]["share"],
                "n": r["share_next_equals_last"]["n"],
            }
            for r in concept["rows"]
        ]
    }
    summary["claims"]["exact_dup_chance"] = chance["comparison"]
    summary["claims"]["rank_reversal"] = {
        ds: {
            "all_order": [x["method"] for x in payload["rankings_by_slice_r5"]["all"]],
            "reversals": payload["rank_reversals"],
        }
        for ds, payload in ranks["datasets"].items()
    }
    write_json("review_summary.json", summary)
    write_json(
        "followup3_summary.json",
        {
            "forgetting_status": forget["status"],
            "concept_repetition": concept,
            "exact_dup_chance": chance["comparison"],
            "rank_reversal_flags": summary["claims"]["rank_reversal"],
        },
    )


def main() -> None:
    print("start follow-up 3", flush=True)
    mark_forgetting_unresolved()
    forget = run_forgetting_redo()
    xes_kc = run_xes_kc_methods()
    concept = run_concept_table(xes_kc)
    chance = run_exact_dup_chance()
    ranks = run_rank_reversal()
    write_summary(forget, xes_kc, concept, chance, ranks)

    lines_path = OUT / "all_phases_results.txt"
    lines = lines_path.read_text().splitlines() if lines_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("F3 ")]
    rev_flags = {ds: bool(ranks["datasets"][ds]["rank_reversals"]) for ds in ranks["datasets"]}
    lines += [
        f"F3 forgetting status={forget['status']} "
        f"auc_succ={forget['auc']['success_only']} "
        f"auc_log={forget['auc']['success_plus_log1p_gap']} "
        f"delta={forget['paired_deltas']['log1p_gap_minus_success']}",
        f"F3 concept shares={[r['share_next_equals_last']['share'] for r in concept['rows'][:3]]}",
        f"F3 exact dup real={chance['comparison']['real_overlap']} chance={chance['comparison']['chance_overlap']}",
        f"F3 rank reversals={rev_flags}",
    ]
    lines_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]), flush=True)
    print("done follow-up 3", flush=True)


if __name__ == "__main__":
    main()
