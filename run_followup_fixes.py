"""Follow-up fixes (ordered): claim-4 threshold, timed diagnosis, KTBD-filtered
timed share, duplicates, enriched ranking. Writes JSON only; docs from JSON.

Run: python3 run_followup_fixes.py
Then: python3 scripts/render_external_review.py && python3 scripts/build_dataflow_report.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import build_heads, write_json
from run_claude_fixes import (
    N_BOOT,
    action_label,
    is_bounce,
    load_similarity_weighted,
    mrr_at,
    paired_seq_delta,
    pool_stratum,
    scores_to_probs,
    seq_bootstrap_mean,
    seq_hash,
    prefix12_hash,
)
from run_valid_eval import (
    EVAL_POS,
    EVAL_SEQS,
    TIMED_NPZ,
    build_timed_sequences,
    eval_memory_table,
    memory_rows_by_sequence,
    recall_at,
    score_features,
    split_by_id,
)

US_PER_DAY = 1_000_000 * 86_400
PRACTICAL_DELTA_AUC = 0.005
BUNDLE = OUT / "serve_models.joblib"
PHASE = OUT / "phases"


# ---------------------------------------------------------------------------
# 1. Claim 4 practical threshold
# ---------------------------------------------------------------------------

def relabel_claim4() -> dict:
    ready = json.loads((PHASE / "review_readiness.json").read_text())
    primary = ready["dag_helps_correctness"].get("primary_delta_auc") or {}
    mean = primary.get("mean")
    ci_hi0 = bool(primary.get("ci_above_zero"))
    if mean is not None and mean >= PRACTICAL_DELTA_AUC and ci_hi0:
        status = "supported"
    elif mean is not None and ci_hi0:
        status = "detectable_but_negligible"
    else:
        status = "unresolved"
    ready["dag_helps_correctness"] = {
        "status": status,
        "practical_threshold_delta_auc": PRACTICAL_DELTA_AUC,
        "rule": (
            f"supported if mean ΔAUC >= {PRACTICAL_DELTA_AUC} and CI entirely above 0; "
            "detectable_but_negligible if CI above 0 but mean < threshold; else unresolved"
        ),
        "primary_variant": ready["dag_helps_correctness"].get(
            "primary_variant", "min_all_parents_observed / logistic"
        ),
        "primary_delta_auc": primary,
    }
    write_json("review_readiness.json", ready)

    summary = json.loads((PHASE / "review_summary.json").read_text())
    summary["claims"]["3_dag_helps_correctness"] = {
        "status": status,
        "practical_threshold_delta_auc": PRACTICAL_DELTA_AUC,
        "primary": ready["dag_helps_correctness"],
    }
    summary["claims"]["4_confounding"] = {
        "status": (
            "inconclusive_pending_test"
            if status != "supported"
            else "rejected_controlled_effect_practical"
        ),
        "note": (
            "raw +10pp unlocked-vs-violate gap is not a controlled readiness effect; "
            f"controlled ΔAUC status={status}"
        ),
    }
    write_json("review_summary.json", summary)
    return ready["dag_helps_correctness"]


# ---------------------------------------------------------------------------
# 2. Timed diagnosis + fixed gap bins + forgetting rerun
# ---------------------------------------------------------------------------

def diagnose_timed() -> dict:
    print("task2 timed diagnosis…", flush=True)
    z = np.load(TIMED_NPZ)
    u, c, t, y = z["user_id"], z["concept"], z["t_us"], z["correct"]
    day = t.astype(np.int64) // US_PER_DAY
    same_u = u[1:] == u[:-1]
    same_c = c[1:] == c[:-1]
    m = same_u & same_c
    gaps = day[1:][m] - day[:-1][m]
    gap_shares = {
        "n_consecutive_same_exercise": int(m.sum()),
        "share_gap_0": float((gaps == 0).mean()),
        "share_gap_1": float((gaps == 1).mean()),
        "share_gap_2plus": float((gaps >= 2).mean()),
        "unit": "calendar_days_from_t_us_microseconds",
    }
    # next==last raw
    tot = hits = 0
    i = 0
    n = len(u)
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        for k in range(i + 1, j):
            tot += 1
            if c[k] == c[k - 1]:
                hits += 1
        i = j
    share_nel = hits / max(tot, 1)

    # 10 example learners
    rng = np.random.default_rng(SEED)
    # prefer users with length>=12
    lengths = []
    i = 0
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        lengths.append((int(u[i]), i, j, j - i))
        i = j
    long_users = [x for x in lengths if x[3] >= 12]
    pick = rng.choice(len(long_users), size=min(10, len(long_users)), replace=False)
    examples = []
    for pi in pick:
        uid, lo, hi, L = long_users[int(pi)]
        rows = []
        for idx in range(lo, min(lo + 12, hi)):
            rows.append(
                {
                    "user_id": int(u[idx]),
                    "concept": int(c[idx]),
                    "t_us": int(t[idx]),
                    "correct": int(y[idx]),
                    "calendar_day": int(day[idx]),
                }
            )
        examples.append({"user_id": uid, "n_total": L, "first_rows": rows})

    day_ok = gap_shares["share_gap_0"] >= 0.9
    diagnosis = {
        "consecutive_same_exercise_day_gaps": gap_shares,
        "share_next_equals_last_raw": {"share": float(share_nel), "n": tot, "hits": hits},
        "day_variable_ok": bool(day_ok),
        "bug_found": (
            None
            if day_ok
            else "same-day gaps near zero despite high next==last → day unit bug"
        ),
        "gap_bin_bug_previous": (
            "Gap bins used fractional days from sequence start with bin (0,0) for same_day, "
            "so almost no rows matched. Fixed: integer calendar-day gaps."
        ),
        "example_learners": examples,
        "t_us_range": [int(t.min()), int(t.max())],
        "unique_users": int(len(np.unique(u))),
    }
    write_json("followup_timed_diagnosis.json", diagnosis)
    print("day_ok", day_ok, "gap0", gap_shares["share_gap_0"], flush=True)
    return diagnosis


def run_gap_bins_fixed() -> dict:
    """Revisit gap bins using integer calendar days (not fractional (0,0) trap)."""
    print("task2 gap bins (calendar day)…", flush=True)
    z = np.load(TIMED_NPZ)
    u, c, t, y = z["user_id"], z["concept"], z["t_us"], z["correct"]
    day = t.astype(np.int64) // US_PER_DAY
    bin_defs = [(0, 0, "same_day"), (1, 3, "1_3_days"), (4, 14, "4_14_days"), (15, 10_000, "15p_days")]
    bins = {name: {"y": [], "success": [], "gap": []} for *_, name in bin_defs}
    i = 0
    n = len(u)
    n_users = 0
    while i < n and n_users < 8000:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        if j - i < 12:
            i = j
            continue
        n_users += 1
        last_day: dict[int, int] = {}
        stats: dict[int, list[int]] = {}
        for idx in range(i, min(j, i + 200)):
            concept = int(c[idx])
            correct = int(y[idx])
            d = int(day[idx])
            if concept in last_day:
                gap = d - last_day[concept]
                a, s = stats[concept]
                succ = s / a if a else 0.5
                for lo, hi, name in bin_defs:
                    if lo <= gap <= hi:
                        bins[name]["y"].append(correct)
                        bins[name]["success"].append(succ)
                        bins[name]["gap"].append(gap)
                        break
            a, s = stats.get(concept, [0, 0])
            stats[concept] = [a + 1, s + correct]
            last_day[concept] = d
        i = j

    out = {"n_users": n_users, "bins": {}, "day_unit": "integer_calendar_day_from_t_us"}
    for name, payload in bins.items():
        yy = np.asarray(payload["y"], dtype=float)
        if len(yy) < 50:
            out["bins"][name] = {"n": int(len(yy))}
            continue
        succ = np.asarray(payload["success"], dtype=float)
        gap = np.asarray(payload["gap"], dtype=float)
        entry = {
            "n": int(len(yy)),
            "mean_correct": float(yy.mean()),
            "mean_gap_days": float(gap.mean()),
        }
        if len(np.unique(yy)) >= 2:
            entry["auc_success_rate"] = float(roc_auc_score(yy, succ))
            entry["auc_neg_log1p_gap"] = float(roc_auc_score(yy, -np.log1p(gap)))
        out["bins"][name] = entry
    write_json("followup_gap_bins.json", out)
    print("gap bins n", {k: v.get("n") for k, v in out["bins"].items()}, flush=True)
    return out


def run_forgetting_rerun() -> dict:
    """Learner-disjoint timed memory bake-off with fractional day Δt (build_timed_sequences)."""
    print("task2 forgetting rerun…", flush=True)
    rng = np.random.default_rng(SEED)
    seqs, days, user_ids = build_timed_sequences()
    user_ids = np.array(user_ids)
    tr_u, te_u = split_by_id(user_ids)
    tr_seq = [seqs[i] for i, u in enumerate(user_ids) if u in tr_u]
    te_seq = [seqs[i] for i, u in enumerate(user_ids) if u in te_u]
    tr_days = [days[i] for i, u in enumerate(user_ids) if u in tr_u]
    te_days = [days[i] for i, u in enumerate(user_ids) if u in te_u]
    xf, yf = [], []
    for feat, yy, _ in memory_rows_by_sequence(tr_seq, tr_days):
        xf.append(feat)
        yf.append(yy)
        if len(xf) >= 200_000:
            break
    xt, yt = [], []
    for feat, yy, _ in memory_rows_by_sequence(te_seq, te_days):
        xt.append(feat)
        yt.append(yy)
        if len(xt) >= 80_000:
            break
    table, _ = eval_memory_table(np.vstack(xf), np.array(yf), np.vstack(xt), np.array(yt), rng)
    # cite-ability: only after day diagnosis ok
    diag = json.loads((PHASE / "followup_timed_diagnosis.json").read_text())
    table_out = {
        "timed_learner_split": table,
        "n_train_users": len(tr_u),
        "n_test_users": len(te_u),
        "day_variable_ok": diag["day_variable_ok"],
        "can_cite_forgetting_null": bool(
            diag["day_variable_ok"] and "paired_logistic_day_minus_success" in table
        ),
        "paired_day_minus_success": table.get("paired_logistic_day_minus_success"),
        "note": (
            "Fractional event Δt retained for memory model; gap bins use calendar days. "
            "Do not cite forgetting null unless day_variable_ok."
        ),
    }
    write_json("followup_forgetting.json", table_out)
    # also refresh phase3 timed block so PDF can read it
    p3_path = PHASE / "phase3_memory.json"
    if p3_path.exists():
        p3 = json.loads(p3_path.read_text())
        p3["timed_learner_split"] = table
        p3["timed_n_train_users"] = len(tr_u)
        p3["timed_n_test_users"] = len(te_u)
        p3["timed_rerun_note"] = (
            "Rerun after timed diagnosis; day unit=microseconds, gap bins fixed separately."
        )
        p3["timed_day_variable_ok"] = diag["day_variable_ok"]
        write_json("phase3_memory.json", p3)
    print(
        "forgetting",
        table.get("winner"),
        table.get("paired_logistic_day_minus_success"),
        flush=True,
    )
    return table_out


# ---------------------------------------------------------------------------
# 3. KTBD filters on timed
# ---------------------------------------------------------------------------

def run_timed_ktbd_filtered(train_kt, test_kt) -> dict:
    print("task3 KTBD-filtered timed share…", flush=True)
    ktbd = {"share": None, "n": 0}
    hit = tot = 0
    for seq in train_kt + test_kt:
        for t in range(1, len(seq)):
            tot += 1
            if seq[t][0] == seq[t - 1][0]:
                hit += 1
    ktbd = {"share": hit / max(tot, 1), "n": tot, "hits": hit}

    z = np.load(TIMED_NPZ)
    u, c = z["user_id"], z["concept"]
    MIN_LEN, MAX_LEN = 12, 200
    # raw
    raw_hit = raw_tot = 0
    # filtered
    fil_hit = fil_tot = 0
    n_seq_fil = 0
    i = 0
    n = len(u)
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        concepts = c[i:j]
        for k in range(1, len(concepts)):
            raw_tot += 1
            if concepts[k] == concepts[k - 1]:
                raw_hit += 1
        if len(concepts) >= MIN_LEN:
            seq_c = concepts[:MAX_LEN]
            if len(seq_c) >= MIN_LEN:
                n_seq_fil += 1
                for k in range(1, len(seq_c)):
                    fil_tot += 1
                    if seq_c[k] == seq_c[k - 1]:
                        fil_hit += 1
        i = j

    out = {
        "ktbd": {"share_next_equals_last": ktbd, "filter": "EduData ktbd len 12–200"},
        "timed_raw": {
            "share_next_equals_last": {
                "share": raw_hit / max(raw_tot, 1),
                "n": raw_tot,
                "hits": raw_hit,
            },
            "note": "all user streams in timed_interactions.npz (exercise set already ktbd-mapped)",
        },
        "timed_ktbd_filters": {
            "share_next_equals_last": {
                "share": fil_hit / max(fil_tot, 1),
                "n": fil_tot,
                "hits": fil_hit,
            },
            "n_sequences": n_seq_fil,
            "filters": "length>=12, truncate to 200, exercise ids ⊆ ktbd vertex map",
        },
        "remaining_gap": {
            "ktbd_minus_timed_filtered": float(ktbd["share"] - fil_hit / max(fil_tot, 1)),
            "explanation": (
                "Length/exercise filters do not close the gap. EduData ktbd sequences are "
                "not raw per-user streams: they apply session/problem-log construction that "
                "raises consecutive repeats (sticky within-session practice). Timed npz is "
                "chronological user streams (capped extract). Do not treat the two shares as "
                "the same estimand; cite ktbd for ranking eval and timed only for Δt/forgetting."
            ),
            "can_use_both_as_same_share": False,
        },
    }
    write_json("review_timed_ktbd.json", out)
    print(
        "shares ktbd",
        ktbd["share"],
        "timed_raw",
        out["timed_raw"]["share_next_equals_last"]["share"],
        "timed_filt",
        out["timed_ktbd_filters"]["share_next_equals_last"]["share"],
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Duplicates
# ---------------------------------------------------------------------------

def run_duplicates(train_kt, test_kt) -> dict:
    print("task4 duplicates…", flush=True)
    tr_c = Counter(seq_hash(s) for s in train_kt)
    te_c = Counter(seq_hash(s) for s in test_kt)
    overlap = set(tr_c) & set(te_c)
    lengths = []
    test_dup_idx = []
    for i, s in enumerate(test_kt):
        h = seq_hash(s)
        if h in overlap:
            lengths.append(len(s))
            test_dup_idx.append(i)

    # train-train prefix12 baseline
    pref: dict[str, int] = defaultdict(int)
    for s in train_kt:
        pref[prefix12_hash(s)] += 1
    n_tr = len(train_kt)
    n_pref = len(pref)
    n_pref_collide = sum(1 for v in pref.values() if v > 1)
    n_seq_collide = sum(v for v in pref.values() if v > 1)
    # chance-style: expected collision if prefixes uniform over observed support — report empirical rate
    train_prefix = {
        "n_train_sequences": n_tr,
        "n_unique_prefixes": n_pref,
        "n_prefixes_with_gt1_seq": n_pref_collide,
        "n_sequences_in_colliding_prefixes": n_seq_collide,
        "frac_sequences_in_colliding_prefixes": n_seq_collide / max(n_tr, 1),
        "frac_prefixes_that_collide": n_pref_collide / max(n_pref, 1),
        "note": "empirical train-train first-12 collision baseline (not cross-split)",
    }

    te_prefs = {prefix12_hash(s) for s in test_kt}
    tr_prefs = set(pref)
    cross = te_prefs & tr_prefs

    out = {
        "learner_ids": {
            "status": "not_testable",
            "reason": "ktbd-junyi train.json/test.json lines have no learner id field",
        },
        "exact_duplicate_sequences": {
            "n_overlap_hashes": len(overlap),
            "n_test_rows_in_overlap": len(test_dup_idx),
            "n_train_rows_in_overlap": int(sum(tr_c[h] for h in overlap)),
            "lengths_of_test_duplicate_rows": lengths,
            "length_summary": {
                "min": int(min(lengths)) if lengths else None,
                "max": int(max(lengths)) if lengths else None,
                "mean": float(np.mean(lengths)) if lengths else None,
                "median": float(np.median(lengths)) if lengths else None,
                "counter": dict(Counter(lengths)),
            },
        },
        "prefix12_overlap_train_test": {
            "n_overlap_prefixes": len(cross),
            "frac_test_prefixes_in_train": len(cross) / max(len(te_prefs), 1),
        },
        "train_train_prefix12_baseline": train_prefix,
        "test_dup_indices": test_dup_idx,
    }
    write_json("review_overlap.json", out)
    print("dup lengths", out["exact_duplicate_sequences"]["length_summary"], flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. Ranking (both versions) with slices + R@1/MRR + ordering
# ---------------------------------------------------------------------------

def evaluate_ranking_full(test_seq, names, graph, similar_set, sim_weighted, embeddings, bundle, parent_lists, rng, tag: str) -> dict:
    print(f"task5 ranking [{tag}]…", flush=True)
    popularity = np.asarray(bundle["popularity"])
    feat_fn = build_heads(
        embeddings, graph, similar_set, popularity, parent_lists, bundle["logreg"]
    )
    mode_clf, review_clf, advance_clf = bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"]

    n_eval = min(EVAL_SEQS, len(test_seq))
    idx = rng.choice(len(test_seq), size=n_eval, replace=False)

    metric_names = [
        "ragr_r1", "ragr_r5", "ragr_mrr", "ragr_nll",
        "last_item_r1", "last_item_r5", "last_item_mrr",
        "recent5_r1", "recent5_r5", "recent5_mrr",
        "force_review_r1", "force_review_r5", "force_review_mrr",
        "force_advance_r1", "force_advance_r5", "force_advance_mrr",
        "freq_r1", "freq_r5", "freq_mrr",
        "low_success_r1", "low_success_r5", "low_success_mrr",
        "sim_weighted_r1", "sim_weighted_r5", "sim_weighted_mrr",
    ]
    slices = ("continue", "bounce", "far", "advance")
    slice_metrics = ("ragr_r1", "ragr_r5", "ragr_mrr", "last_item_r5", "recent5_r5", "force_review_r5")

    buckets = {m: [] for m in metric_names}
    slice_buckets = {s: {m: [] for m in slice_metrics} for s in slices}
    strata = {s: {m: [] for m in ("ragr_r5", "last_item_r5", "recent5_r5", "force_review_r5")} for s in ("1", "2", "3-5", "6+")}
    mix = {"continue": 0, "revisit_bounce": 0, "revisit_far": 0, "advance": 0}
    revisit_steps = []
    pool_sizes = []
    gate_y, gate_X = [], []

    for si in idx:
        seq = test_seq[int(si)]
        cuts = np.linspace(8, len(seq) - 1, num=EVAL_POS, dtype=int)
        local = {m: [] for m in metric_names}
        local_slice = {s: {m: [] for m in slice_metrics} for s in slices}
        local_strata = {s: {m: [] for m in strata["1"]} for s in strata}

        for cut in cuts:
            cut = int(cut)
            pack = score_features(feat_fn, mode_clf, review_clf, advance_clf, seq, cut, graph, embeddings)
            label = pack["label"]
            last = pack["last"]
            seen = pack["seen"]
            stats, last_step, recent = history_state(seq, cut)
            pool = int(seen.sum())
            pool_sizes.append(pool)
            stratum = pool_stratum(pool)

            act = action_label(label, last_step, last)
            if act == "continue":
                mix["continue"] += 1
                sl = "continue"
            elif act == "revisit":
                steps_since = int(cut - last_step[label])
                revisit_steps.append(steps_since)
                if is_bounce(seq, cut, label, last):
                    mix["revisit_bounce"] += 1
                    sl = "bounce"
                else:
                    mix["revisit_far"] += 1
                    sl = "far"
            else:
                mix["advance"] += 1
                sl = "advance"

            mixed = pack["mixed"]
            probs = scores_to_probs(mixed)
            local["ragr_r1"].append(recall_at(mixed, label, 1))
            local["ragr_r5"].append(recall_at(mixed, label, 5))
            local["ragr_mrr"].append(mrr_at(mixed, label))
            local["ragr_nll"].append(float(-np.log(max(probs[label], 1e-12))))

            last_scores = np.zeros_like(mixed)
            last_scores[last] = 1.0
            local["last_item_r1"].append(recall_at(last_scores, label, 1))
            local["last_item_r5"].append(recall_at(last_scores, label, 5))
            local["last_item_mrr"].append(mrr_at(last_scores, label))

            distinct = []
            for x in reversed(recent):
                if x not in distinct:
                    distinct.append(x)
                if len(distinct) >= 5:
                    break
            r5_scores = np.zeros_like(mixed)
            for rank, item in enumerate(distinct):
                r5_scores[item] = 5 - rank
            local["recent5_r1"].append(recall_at(r5_scores, label, 1))
            local["recent5_r5"].append(recall_at(r5_scores, label, 5))
            local["recent5_mrr"].append(mrr_at(r5_scores, label))

            local["force_review_r1"].append(recall_at(pack["review"], label, 1))
            local["force_review_r5"].append(recall_at(pack["review"], label, 5))
            local["force_review_mrr"].append(mrr_at(pack["review"], label))
            local["force_advance_r1"].append(recall_at(pack["advance"], label, 1))
            local["force_advance_r5"].append(recall_at(pack["advance"], label, 5))
            local["force_advance_mrr"].append(mrr_at(pack["advance"], label))

            freq_scores = np.full_like(mixed, -1e9)
            for ci, (a, _s) in stats.items():
                freq_scores[ci] = float(a)
            local["freq_r1"].append(recall_at(freq_scores, label, 1))
            local["freq_r5"].append(recall_at(freq_scores, label, 5))
            local["freq_mrr"].append(mrr_at(freq_scores, label))

            low_scores = np.full_like(mixed, -1e9)
            for ci, (a, s) in stats.items():
                low_scores[ci] = -(s / a if a else 0.5)
            local["low_success_r1"].append(recall_at(low_scores, label, 1))
            local["low_success_r5"].append(recall_at(low_scores, label, 5))
            local["low_success_mrr"].append(mrr_at(low_scores, label))

            sim_scores = np.full_like(mixed, -1e9)
            for nb, w in sim_weighted.get(last, []):
                sim_scores[nb] = w
            local["sim_weighted_r1"].append(recall_at(sim_scores, label, 1))
            local["sim_weighted_r5"].append(recall_at(sim_scores, label, 5))
            local["sim_weighted_mrr"].append(mrr_at(sim_scores, label))

            for m in slice_metrics:
                local_slice[sl][m].append(local[m][-1])
            for m in ("ragr_r5", "last_item_r5", "recent5_r5", "force_review_r5"):
                local_strata[stratum][m].append(local[m][-1])

            is_review = 1.0 if act != "advance" else 0.0
            attempts_last = 0
            for t in range(cut - 1, -1, -1):
                if seq[t][0] != last:
                    break
                attempts_last += 1
            streak = 0
            for t in range(cut - 1, cut - 1 - attempts_last, -1):
                if seq[t][1] == 1:
                    streak += 1
                else:
                    break
            gate_y.append(is_review)
            gate_X.append([float(streak), float(attempts_last), float(seq[cut - 1][1])])

        for m, vals in local.items():
            if vals:
                buckets[m].append(vals)
        for s in slices:
            for m, vals in local_slice[s].items():
                if vals:
                    slice_buckets[s][m].append(vals)
        for s, md in local_strata.items():
            for m, vals in md.items():
                if vals:
                    strata[s][m].append(vals)

    n_mix = sum(mix.values())
    three_way = {k: {"count": v, "frac": v / max(n_mix, 1)} for k, v in mix.items()}
    three_way["n"] = n_mix
    three_way["revisit_frac"] = (mix["revisit_bounce"] + mix["revisit_far"]) / max(n_mix, 1)
    three_way["continue_frac"] = mix["continue"] / max(n_mix, 1)

    table = {m: seq_bootstrap_mean(buckets[m], rng, N_BOOT) for m in metric_names}
    # ordering by R@5 mean (computed)
    order_r5 = sorted(
        [
            ("ragr", table["ragr_r5"]["mean"]),
            ("force_review", table["force_review_r5"]["mean"]),
            ("recent5", table["recent5_r5"]["mean"]),
            ("last_item", table["last_item_r5"]["mean"]),
            ("freq", table["freq_r5"]["mean"]),
            ("low_success", table["low_success_r5"]["mean"]),
            ("sim_weighted", table["sim_weighted_r5"]["mean"]),
            ("force_advance", table["force_advance_r5"]["mean"]),
        ],
        key=lambda kv: (-1 if kv[1] is None else -kv[1], kv[0]),
    )
    ordering_sentence = " > ".join(
        f"{name}={val:.3f}" for name, val in order_r5 if val is not None
    )

    paired = {
        "ragr_minus_recent5_r5": paired_seq_delta(buckets["ragr_r5"], buckets["recent5_r5"], rng, N_BOOT),
        "ragr_minus_last_item_r5": paired_seq_delta(buckets["ragr_r5"], buckets["last_item_r5"], rng, N_BOOT),
        "ragr_minus_force_review_r5": paired_seq_delta(buckets["ragr_r5"], buckets["force_review_r5"], rng, N_BOOT),
        "ragr_minus_force_review_r1": paired_seq_delta(buckets["ragr_r1"], buckets["force_review_r1"], rng, N_BOOT),
        "ragr_minus_force_review_mrr": paired_seq_delta(buckets["ragr_mrr"], buckets["force_review_mrr"], rng, N_BOOT),
    }

    slices_out = {
        s: {m: seq_bootstrap_mean(slice_buckets[s][m], rng, N_BOOT) for m in slice_metrics}
        for s in slices
    }
    strata_out = {
        s: {m: seq_bootstrap_mean(strata[s][m], rng, N_BOOT) for m in strata[s]} for s in strata
    }

    gate_X_a = np.asarray(gate_X, dtype=float)
    gate_y_a = np.asarray(gate_y, dtype=float)
    g_order = np.arange(n_eval)
    rng.shuffle(g_order)
    n_g_te = max(1, int(0.2 * n_eval))
    te_s = set(g_order[:n_g_te].tolist())
    tr_mask = np.array([i // EVAL_POS not in te_s for i in range(len(gate_y_a))])
    te_mask = ~tr_mask
    gate_out = {"n": int(len(gate_y_a)), "base_rate_review": float(gate_y_a.mean())}
    if tr_mask.sum() > 50 and te_mask.sum() > 20 and len(np.unique(gate_y_a[te_mask])) >= 2:
        gclf = LogisticRegression(max_iter=400).fit(gate_X_a[tr_mask], gate_y_a[tr_mask])
        gp = gclf.predict_proba(gate_X_a[te_mask])[:, 1]
        gate_out.update(
            {
                "auc": float(roc_auc_score(gate_y_a[te_mask], gp)),
                "pr_auc": float(average_precision_score(gate_y_a[te_mask], gp)),
                "features": ["correct_streak_on_last", "attempts_on_last", "last_correct"],
            }
        )

    return {
        "tag": tag,
        "eval": {
            "n_sequences": n_eval,
            "cuts_per_sequence": EVAL_POS,
            "n_queries": n_eval * EVAL_POS,
            "seed": SEED,
            "n_test_pool": len(test_seq),
        },
        "three_way_mix": three_way,
        "revisit": {
            "steps_since_last_seen": {
                "n": len(revisit_steps),
                "mean": float(np.mean(revisit_steps)) if revisit_steps else None,
                "median": float(np.median(revisit_steps)) if revisit_steps else None,
                "p90": float(np.quantile(revisit_steps, 0.9)) if revisit_steps else None,
            },
            "bounce_frac_of_all": mix["revisit_bounce"] / max(n_mix, 1),
            "far_frac_of_all": mix["revisit_far"] / max(n_mix, 1),
            "bounce_among_revisit": mix["revisit_bounce"]
            / max(mix["revisit_bounce"] + mix["revisit_far"], 1),
            "definition_bounce": "label==item_at_cut-2 and label!=last (A,B,A)",
            "definition_far": "revisit and not bounce",
        },
        "pool_size_seen_items": {
            "mean": float(np.mean(pool_sizes)),
            "median": float(np.median(pool_sizes)),
            "p10": float(np.quantile(pool_sizes, 0.1)),
            "p90": float(np.quantile(pool_sizes, 0.9)),
            "n": len(pool_sizes),
        },
        "metrics": table,
        "ordering_by_r5": [{"method": n, "r5": v} for n, v in order_r5],
        "ordering_sentence_r5": ordering_sentence,
        "slices": slices_out,
        "strata_by_pool_r5": strata_out,
        "paired_deltas": paired,
        "primary_statistic": "Recall@5 mean_of_per_sequence_means; paired deltas include RAGR-force_review",
        "gate_streak_logistic": gate_out,
    }


def main() -> None:
    print("start follow-up fixes", flush=True)
    claim4 = relabel_claim4()
    print("claim4", claim4["status"], flush=True)

    diag = diagnose_timed()
    gaps = run_gap_bins_fixed()
    forget = run_forgetting_rerun()

    train_kt = load_sequences(DATA / "train.json")
    test_kt = load_sequences(DATA / "test.json")
    timed = run_timed_ktbd_filtered(train_kt, test_kt)
    overlap = run_duplicates(train_kt, test_kt)

    # ranking both versions
    names = load_names()
    graph, similar_set = load_graph(len(names))
    sim_weighted = load_similarity_weighted(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]
    bundle = joblib.load(BUNDLE)

    overlap_hashes = set()
    tr_hashes = {seq_hash(s) for s in train_kt}
    for s in test_kt:
        h = seq_hash(s)
        if h in tr_hashes:
            overlap_hashes.add(h)
    test_clean = [s for s in test_kt if seq_hash(s) not in overlap_hashes]
    print("test pools", len(test_kt), "clean", len(test_clean), flush=True)

    rng_a = np.random.default_rng(SEED)
    ranking_with = evaluate_ranking_full(
        test_kt, names, graph, similar_set, sim_weighted, embeddings, bundle, parent_lists, rng_a, "with_duplicates"
    )
    write_json("review_ranking.json", ranking_with)
    write_json("followup_ranking_with_dups.json", ranking_with)

    rng_b = np.random.default_rng(SEED)
    ranking_clean = evaluate_ranking_full(
        test_clean, names, graph, similar_set, sim_weighted, embeddings, bundle, parent_lists, rng_b, "duplicates_removed"
    )
    write_json("followup_ranking_no_dups.json", ranking_clean)

    # update summary claims from computed JSON
    summary = json.loads((PHASE / "review_summary.json").read_text())
    summary["claims"]["1_continue_dominates"] = {
        "continue_frac": ranking_with["three_way_mix"]["continue_frac"],
        "computed_from": "review_ranking.three_way_mix",
    }
    summary["claims"]["2_ragr_vs_recency"] = {
        "ragr_r5": ranking_with["metrics"]["ragr_r5"],
        "recent5_r5": ranking_with["metrics"]["recent5_r5"],
        "force_review_r5": ranking_with["metrics"]["force_review_r5"],
        "last_item_r5": ranking_with["metrics"]["last_item_r5"],
        "ordering_sentence_r5": ranking_with["ordering_sentence_r5"],
        "paired": ranking_with["paired_deltas"],
    }
    summary["claims"]["5_learner_ids"] = overlap["learner_ids"]
    summary["claims"]["6_timed_ktbd"] = {
        "ktbd_share": timed["ktbd"]["share_next_equals_last"]["share"],
        "timed_raw_share": timed["timed_raw"]["share_next_equals_last"]["share"],
        "timed_filtered_share": timed["timed_ktbd_filters"]["share_next_equals_last"]["share"],
        "remaining_gap": timed["remaining_gap"],
        "day_variable_ok": diag["day_variable_ok"],
        "can_cite_forgetting_null": forget["can_cite_forgetting_null"],
    }
    summary["followup"] = {
        "claim4": claim4,
        "gap_bins_same_day_n": gaps["bins"].get("same_day", {}).get("n"),
        "duplicate_lengths": overlap["exact_duplicate_sequences"]["length_summary"],
        "ranking_no_dups_ragr_r5": ranking_clean["metrics"]["ragr_r5"],
    }
    write_json("review_summary.json", summary)

    lines_path = OUT / "all_phases_results.txt"
    lines = lines_path.read_text().splitlines() if lines_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("Followup ") and not ln.startswith("Review ")]
    lines += [
        f"Followup claim4={claim4['status']} threshold={PRACTICAL_DELTA_AUC} "
        f"delta={claim4['primary_delta_auc'].get('mean')}",
        f"Followup timed day_ok={diag['day_variable_ok']} gap0={diag['consecutive_same_exercise_day_gaps']['share_gap_0']:.3f}",
        f"Followup shares ktbd={timed['ktbd']['share_next_equals_last']['share']:.3f} "
        f"timed_raw={timed['timed_raw']['share_next_equals_last']['share']:.3f} "
        f"timed_filt={timed['timed_ktbd_filters']['share_next_equals_last']['share']:.3f}",
        f"Followup forgetting cite={forget['can_cite_forgetting_null']} "
        f"paired={forget['paired_day_minus_success']}",
        f"Followup dups lengths={overlap['exact_duplicate_sequences']['length_summary']}",
        f"Followup R@5 order: {ranking_with['ordering_sentence_r5']}",
        f"Followup no_dups R@5 ragr={ranking_clean['metrics']['ragr_r5']['mean']:.3f}",
    ]
    lines_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-10:]), flush=True)
    print("done follow-up", flush=True)


if __name__ == "__main__":
    main()
