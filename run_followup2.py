"""Follow-up 2: repetition table, forgetting power, ranking reconcile,
duplicate chance baseline, XES3G5M contrast. All numbers → JSON.

Run: python3 run_followup2.py
Then: python3 scripts/render_external_review.py
      python3 scripts/render_team_report.py
      python3 scripts/build_dataflow_report.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from junyi_pipeline import DATA, OUT, SEED, load_names, load_sequences
from run_all_phases import write_json
from run_claude_fixes import N_BOOT, prefix12_hash, seq_hash
from run_valid_eval import EVAL_POS, EVAL_SEQS, TIMED_NPZ, memory_rows_by_sequence

PHASE = OUT / "phases"
US_PER_DAY = 1_000_000 * 86_400
PRACTICAL_DELTA_R5 = 0.01
PRACTICAL_DELTA_AUC = 0.005
ROOT = Path(__file__).resolve().parent
EXERCISE_CSV = ROOT / "data/junyi_raw/junyi/junyi_Exercise_table.csv"
ASSIST_CSV = ROOT / "data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv"
XES_Q = ROOT / "data/xes3g5m/XES3G5M/question_level"
XES_KC = ROOT / "data/xes3g5m/XES3G5M/kc_level"


def share_next_eq_last(labels: list) -> dict:
    hit = tot = 0
    for i in range(1, len(labels)):
        tot += 1
        if labels[i] == labels[i - 1]:
            hit += 1
    return {"share": hit / max(tot, 1), "n": tot, "hits": hit}


def share_over_sequences(seq_labels: list[list]) -> dict:
    hit = tot = 0
    for labels in seq_labels:
        for i in range(1, len(labels)):
            tot += 1
            if labels[i] == labels[i - 1]:
                hit += 1
    return {"share": hit / max(tot, 1), "n": tot, "hits": hit, "n_sequences": len(seq_labels)}


def three_way(seq_labels: list[list], max_seq: int = 10_000) -> dict:
    mix = {"continue": 0, "revisit": 0, "advance": 0}
    n = 0
    for labels in seq_labels[:max_seq]:
        seen = set()
        last = None
        for lab in labels:
            if last is not None:
                if lab == last:
                    mix["continue"] += 1
                elif lab in seen:
                    mix["revisit"] += 1
                else:
                    mix["advance"] += 1
                n += 1
            seen.add(lab)
            last = lab
    return {k: {"count": v, "frac": v / max(n, 1)} for k, v in mix.items()} | {"n": n}


# ---------------------------------------------------------------------------
# 1. Repetition comparison
# ---------------------------------------------------------------------------

def load_exercise_topic_map() -> dict[int, str]:
    names = load_names()  # idx -> pretty name with spaces
    # vertex file has underscores
    slug_to_idx = {}
    for line in (DATA / "vertex_id2idx").read_text().splitlines():
        if not line.strip():
            continue
        slug, idx = line.rsplit(",", 1)
        slug_to_idx[slug.strip()] = int(idx)
    ex = pd.read_csv(EXERCISE_CSV)
    idx_to_topic = {}
    for _, row in ex.iterrows():
        slug = str(row["name"]).strip()
        if slug in slug_to_idx:
            idx_to_topic[slug_to_idx[slug]] = str(row["topic"]).strip() if pd.notna(row["topic"]) else "unknown"
    # coverage
    return idx_to_topic


def run_repetition_table() -> dict:
    print("1 repetition table…", flush=True)
    topic_of = load_exercise_topic_map()
    names = load_names()
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")
    ktbd = train + test

    # Junyi ktbd exercise
    ktbd_ex = [[c for c, _ in s] for s in ktbd]
    # Junyi ktbd topic
    ktbd_topic = [[topic_of.get(c, "unknown") for c, _ in s] for s in ktbd]
    mapped = sum(1 for s in ktbd for c, _ in s if c in topic_of)
    total_inter = sum(len(s) for s in ktbd)

    # Junyi timed raw
    z = np.load(TIMED_NPZ)
    u, c, t = z["user_id"], z["concept"], z["t_us"]
    timed_ex_labels = []  # per-user lists
    timed_topic_labels = []
    i = 0
    n = len(u)
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        concepts = [int(x) for x in c[i:j]]
        timed_ex_labels.append(concepts)
        timed_topic_labels.append([topic_of.get(x, "unknown") for x in concepts])
        i = j

    # ASSISTments skill (collapsed)
    raw = pd.read_csv(ASSIST_CSV, encoding="ISO-8859-1", low_memory=False,
                      usecols=["order_id", "user_id", "skill_id", "correct"])
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
    assist_seqs = []
    for _, grp in g.groupby("user_id", sort=False):
        assist_seqs.append(grp["skill_key"].tolist())

    table = {
        "junyi_ktbd_exercise": share_over_sequences(ktbd_ex),
        "junyi_ktbd_topic": share_over_sequences(ktbd_topic),
        "junyi_timed_raw_exercise": share_over_sequences(timed_ex_labels),
        "junyi_timed_raw_topic": share_over_sequences(timed_topic_labels),
        "assistments_skill": share_over_sequences(assist_seqs),
        "topic_map_coverage": {
            "n_exercises_with_topic": len(topic_of),
            "n_ktbd_nodes": len(names),
            "frac_ktbd_interactions_mapped": mapped / max(total_inter, 1),
        },
        "finding_name": "repetition_is_high_on_mastery_based_logs",
        "finding_label": "repetition is high on mastery-based logs (Junyi, ASSISTments)",
        "boolean_rule_deprecated": {
            "old_flag": "repetition_junyi_specific",
            "old_rule": "|assist_skill_share - junyi_ktbd_exercise_share| > 0.10",
            "status": "replaced_by_side_by_side_numbers",
        },
    }
    # optional boolean kept with documented rule comparing mastery logs vs low baseline
    shares = [
        table["junyi_ktbd_exercise"]["share"],
        table["assistments_skill"]["share"],
    ]
    table["high_repetition_boolean"] = {
        "value": bool(min(shares) >= 0.60),
        "rule": "True iff min(Junyi ktbd exercise share, ASSISTments skill share) >= 0.60",
        "label": table["finding_label"],
    }
    write_json("followup2_repetition.json", table)
    print({k: v.get("share") if isinstance(v, dict) and "share" in v else v for k, v in table.items() if k.startswith("junyi") or k.startswith("assist")}, flush=True)
    return table


# ---------------------------------------------------------------------------
# 2. Forgetting power
# ---------------------------------------------------------------------------

def run_forgetting_power() -> dict:
    print("2 forgetting power…", flush=True)
    z = np.load(TIMED_NPZ)
    u, c, t, y = z["user_id"], z["concept"], z["t_us"], z["correct"]
    day = t.astype(np.int64) // US_PER_DAY
    bin_defs = [(0, 0, "same_day"), (1, 3, "1_3_days"), (4, 14, "4_14_days"), (15, 10_000, "15p_days")]
    bins = {name: 0 for *_, name in bin_defs}
    # collect gap>=1 rows for day-vs-success with learner ids
    # feat: [1, log1p(att), rate, log1p(step_dt), log1p(day_gap)]
    rows_by_learner: dict[int, list] = defaultdict(list)

    i = 0
    n = len(u)
    n_users = 0
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        uid = int(u[i])
        n_users += 1
        last_day: dict[int, int] = {}
        last_step: dict[int, int] = {}
        stats: dict[int, list[int]] = {}
        for k, idx in enumerate(range(i, j)):
            concept = int(c[idx])
            correct = int(y[idx])
            d = int(day[idx])
            if concept in last_day:
                gap = d - last_day[concept]
                for lo, hi, name in bin_defs:
                    if lo <= gap <= hi:
                        bins[name] += 1
                        break
                a, s = stats[concept]
                rate = s / a if a else 0.5
                step_dt = float(k - last_step[concept])
                if gap >= 1:
                    feat = np.array(
                        [1.0, np.log1p(a), rate, np.log1p(step_dt), np.log1p(float(gap))],
                        dtype=np.float64,
                    )
                    rows_by_learner[uid].append((feat, float(correct)))
            a, s = stats.get(concept, [0, 0])
            stats[concept] = [a + 1, s + correct]
            last_day[concept] = d
            last_step[concept] = k
        i = j

    # learner-disjoint fit + paired learner bootstrap of day-vs-success ΔAUC
    learners = np.array(sorted(rows_by_learner.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(learners)
    n_te = max(1, int(0.2 * len(learners)))
    te_u = set(learners[:n_te].tolist())
    tr_u = set(learners[n_te:].tolist())
    x_tr, y_tr = [], []
    for uid in tr_u:
        for feat, yy in rows_by_learner[uid]:
            x_tr.append(feat)
            y_tr.append(yy)
    # per-learner test packs
    te_packs = []
    for uid in te_u:
        feats = [f for f, _ in rows_by_learner[uid]]
        ys = [yy for _, yy in rows_by_learner[uid]]
        if feats:
            te_packs.append((np.vstack(feats), np.asarray(ys, dtype=float)))

    n_gap1 = sum(len(v) for v in rows_by_learner.values())
    result = {
        "gap_bins_returns_to_seen_exercise": {
            "same_day": {"n": bins["same_day"]},
            "1_3_days": {"n": bins["1_3_days"]},
            "4_14_days": {"n": bins["4_14_days"]},
            "15p_days": {"n": bins["15p_days"]},
            "n_users_scanned": n_users,
            "unit": "integer_calendar_day",
        },
        "restricted_gap_ge_1_day": {
            "n_rows": n_gap1,
            "n_train_learners": len(tr_u),
            "n_test_learners": len(te_packs),
        },
    }

    claim = "no_evidence_of_forgetting_effect_limited_power"
    paired = None
    if len(x_tr) >= 1000 and te_packs:
        x_tr = np.vstack(x_tr)
        y_tr = np.asarray(y_tr)
        # day model cols 1,2,4; success = col 2
        clf = LogisticRegression(max_iter=500).fit(x_tr[:, [1, 2, 4]], y_tr)
        # point: pool test learners
        y_all = np.concatenate([p[1] for p in te_packs])
        x_all = np.vstack([p[0] for p in te_packs])
        p_day = clf.predict_proba(x_all[:, [1, 2, 4]])[:, 1]
        success = x_all[:, 2]
        point = None
        if len(np.unique(y_all)) >= 2:
            point = float(roc_auc_score(y_all, p_day) - roc_auc_score(y_all, success))
        boots = []
        L = len(te_packs)
        for _ in range(N_BOOT):
            idx = rng.integers(0, L, L)
            yb = np.concatenate([te_packs[i][1] for i in idx])
            if len(np.unique(yb)) < 2:
                continue
            xb = np.vstack([te_packs[i][0] for i in idx])
            pb = clf.predict_proba(xb[:, [1, 2, 4]])[:, 1]
            sb = xb[:, 2]
            boots.append(float(roc_auc_score(yb, pb) - roc_auc_score(yb, sb)))
        if boots:
            arr = np.asarray(boots)
            ci = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
            paired = {
                "mean_delta_auc": point,
                "ci": ci,
                "n_boot": N_BOOT,
                "n_boot_kept": len(boots),
                "statistic": "pooled_auc_day_minus_success_bootstrap_over_learners",
                "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            }
            if n_gap1 >= 20_000 and paired["ci_excludes_zero"]:
                if (point or 0) > 0:
                    claim = "day_delta_t_beats_success"
                else:
                    # powered test; CI clear of zero but day does not help
                    claim = "no_evidence_day_beats_success_powered_negative_or_null"
            else:
                claim = "no_evidence_of_forgetting_effect_limited_power"
        result["restricted_gap_ge_1_day"].update(
            {
                "auc_day": float(roc_auc_score(y_all, p_day)) if len(np.unique(y_all)) >= 2 else None,
                "auc_success": float(roc_auc_score(y_all, success)) if len(np.unique(y_all)) >= 2 else None,
                "paired_day_minus_success": paired,
            }
        )

    result["claim"] = claim
    result["claim_rule"] = (
        "Use 'limited_power' wording only if n_rows(gap>=1) < 20000 OR CI includes 0. "
        "If n>=20000 and CI excludes 0: 'day_delta_t_beats_success' when mean>0, else "
        "'no_evidence_day_beats_success_powered_negative_or_null'."
    )
    result["practical_threshold_note"] = (
        f"Same-day returns dominate (n={bins['same_day']}); gap>=1 n={n_gap1}."
    )
    write_json("followup2_forgetting.json", result)
    print("bins", bins, "gap>=1", n_gap1, "claim", claim, flush=True)
    return result


# ---------------------------------------------------------------------------
# 3. Ranking reconcile
# ---------------------------------------------------------------------------

def run_ranking_reconcile() -> dict:
    print("3 ranking reconcile…", flush=True)
    test = load_sequences(DATA / "test.json")
    train = load_sequences(DATA / "train.json")

    # fresh seed (canonical follow-up ranking)
    rng_fresh = np.random.default_rng(SEED)
    idx_fresh = rng_fresh.choice(len(test), size=min(EVAL_SEQS, len(test)), replace=False)

    # polluted: readiness shuffle on train first (as in run_claude_fixes.main)
    rng_poll = np.random.default_rng(SEED)
    order = np.arange(len(train))
    rng_poll.shuffle(order)  # readiness split
    idx_poll = rng_poll.choice(len(test), size=min(EVAL_SEQS, len(test)), replace=False)

    # load metrics from saved JSONs
    with_dups = json.loads((PHASE / "followup_ranking_with_dups.json").read_text())
    # old 0.875 from first rigorous run — recover from all_phases if present
    old_line = None
    for ln in (OUT / "all_phases_results.txt").read_text().splitlines():
        if "Review R@5 ragr=0.875" in ln or "ragr=0.875" in ln:
            old_line = ln
    # Also check if we saved old review_ranking elsewhere — reconstruct explanation from indices

    ragr = with_dups["metrics"]["ragr_r5"]
    recent = with_dups["metrics"]["recent5_r5"]
    paired = with_dups["paired_deltas"]["ragr_minus_recent5_r5"]
    force = with_dups["metrics"]["force_review_r5"]
    paired_fr = with_dups["paired_deltas"]["ragr_minus_force_review_r5"]

    def label_effect(delta_mean, ci):
        if delta_mean is None:
            return "unknown"
        if abs(delta_mean) < PRACTICAL_DELTA_R5:
            return "negligible"
        if ci and ci[0] is not None and ci[0] > 0:
            return "positive_above_threshold"
        if ci and ci[1] is not None and ci[1] < 0:
            return "negative_above_threshold"
        return "uncertain"

    out = {
        "canonical_run": {
            "source_json": "followup_ranking_with_dups.json",
            "tag": with_dups.get("tag"),
            "config": {
                "seed": SEED,
                "EVAL_SEQS": EVAL_SEQS,
                "EVAL_POS": EVAL_POS,
                "n_boot": N_BOOT,
                "duplicate_handling": "kept (pool=full test.json)",
                "rng_protocol": "np.random.default_rng(SEED) used ONLY for eval sample (fresh)",
            },
            "sample_indices_first20": idx_fresh[:20].tolist(),
            "sample_indices_sum": int(idx_fresh.sum()),
            "sample_indices_sha256": hashlib.sha256(idx_fresh.astype(np.int64).tobytes()).hexdigest(),
            "metrics": {
                "ragr_r5": ragr,
                "recent5_r5": recent,
                "force_review_r5": force,
            },
            "paired_ragr_minus_recent5_r5": paired,
            "paired_ragr_minus_force_review_r5": paired_fr,
        },
        "legacy_polluted_rng_run": {
            "stated_settings_same": True,
            "actual_difference": (
                "run_claude_fixes.py reused one RNG for readiness sequence-shuffle then ranking "
                "sample; that advances the RNG, so the 800 test indices differ from a fresh SEED draw."
            ),
            "config": {
                "seed": SEED,
                "EVAL_SEQS": EVAL_SEQS,
                "EVAL_POS": EVAL_POS,
                "duplicate_handling": "kept",
                "rng_protocol": "default_rng(SEED) → readiness shuffle(train) → then choice(test)",
            },
            "sample_indices_first20": idx_poll[:20].tolist(),
            "sample_indices_sum": int(idx_poll.sum()),
            "sample_indices_sha256": hashlib.sha256(idx_poll.astype(np.int64).tobytes()).hexdigest(),
            "overlap_with_canonical_sample": int(len(set(idx_fresh.tolist()) & set(idx_poll.tolist()))),
            "reported_ragr_r5_from_that_run": 0.875,
            "evidence_line": old_line,
            "note": (
                "0.875 was computed under polluted RNG (113/800 index overlap with canonical). "
                "Canonical follow-up number is 0.891 from fresh RNG. Same seed string, different sample."
            ),
        },
        "practical_threshold_recall_at_5": PRACTICAL_DELTA_R5,
        "effect_labels": {
            "ragr_minus_recent5": label_effect(paired["mean"], paired["ci"]),
            "ragr_minus_force_review": label_effect(paired_fr["mean"], paired_fr["ci"]),
            "rule": f"|ΔR@5| < {PRACTICAL_DELTA_R5} → negligible even if CI excludes 0",
        },
    }
    write_json("followup2_ranking_reconcile.json", out)
    print(
        "overlap",
        out["legacy_polluted_rng_run"]["overlap_with_canonical_sample"],
        "effect",
        out["effect_labels"],
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Duplicate chance baseline
# ---------------------------------------------------------------------------

def run_duplicate_chance() -> dict:
    print("4 duplicate chance baseline…", flush=True)
    train = load_sequences(DATA / "train.json")
    test = load_sequences(DATA / "test.json")
    rng = np.random.default_rng(SEED)

    tr_prefs = [prefix12_hash(s) for s in train]
    te_prefs = {prefix12_hash(s) for s in test}
    tr_pref_set = set(tr_prefs)
    real_overlap = te_prefs & tr_pref_set

    # random pseudo-test of 6290 train sequences
    n_pseudo = min(6290, len(train))
    idx = rng.choice(len(train), size=n_pseudo, replace=False)
    pseudo_prefs = {prefix12_hash(train[i]) for i in idx}
    remain_idx = np.setdiff1d(np.arange(len(train)), idx, assume_unique=False)
    remain_prefs = {prefix12_hash(train[i]) for i in remain_idx}
    chance_overlap = pseudo_prefs & remain_prefs

    out = {
        "real_train_test_prefix12": {
            "n_test_sequences": len(test),
            "n_test_unique_prefixes": len(te_prefs),
            "n_overlap_prefixes": len(real_overlap),
            "frac_test_prefixes_in_train": len(real_overlap) / max(len(te_prefs), 1),
        },
        "chance_baseline_pseudo_test": {
            "n_pseudo_test": n_pseudo,
            "n_remaining_train": int(len(remain_idx)),
            "n_pseudo_unique_prefixes": len(pseudo_prefs),
            "n_overlap_prefixes": len(chance_overlap),
            "frac_pseudo_prefixes_in_remaining_train": len(chance_overlap) / max(len(pseudo_prefs), 1),
            "seed": SEED,
            "rule": "sample 6290 train seqs as pseudo-test; collide prefixes against remaining train",
        },
        "comparison": {
            "real_overlap": len(real_overlap),
            "chance_overlap": len(chance_overlap),
            "real_minus_chance": len(real_overlap) - len(chance_overlap),
            "real_frac": len(real_overlap) / max(len(te_prefs), 1),
            "chance_frac": len(chance_overlap) / max(len(pseudo_prefs), 1),
        },
    }
    write_json("followup2_duplicate_chance.json", out)
    print(out["comparison"], flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. XES3G5M
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part or part == "-1":
            continue
        out.append(int(part))
    return out


def _parse_concepts_list(s: str) -> list[list[int]]:
    """Return list of KC lists per question (multi-KC split by _)."""
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part or part == "-1":
            continue
        kcs = [int(x) for x in part.split("_") if x not in ("", "-1")]
        out.append(kcs if kcs else [-1])
    return out


def load_xes_one_seq_per_uid() -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build one question sequence and one KC sequence per uid from train+test quelevel."""
    frames = []
    for name in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv"):
        path = XES_Q / name
        frames.append(pd.read_csv(path, usecols=["uid", "questions", "concepts", "timestamps"]))
    df = pd.concat(frames, ignore_index=True)
    # keep longest window per uid (windows overlap; longest ≈ fullest local history)
    best_q: dict[int, list[int]] = {}
    best_kc: dict[int, list[int]] = {}
    best_len: dict[int, int] = {}
    for _, row in df.iterrows():
        uid = int(row["uid"])
        qs = _parse_int_list(row["questions"])
        cons = _parse_concepts_list(row["concepts"])
        if len(qs) < 12:
            continue
        if uid not in best_len or len(qs) > best_len[uid]:
            best_len[uid] = len(qs)
            best_q[uid] = qs
            # expand KCs aligned to questions
            kcs = []
            for i, q in enumerate(qs):
                if i < len(cons):
                    kcs.extend(cons[i])
                else:
                    kcs.append(-1)
            best_kc[uid] = kcs
    return best_q, best_kc


def run_xes3g5m() -> dict:
    print("5 XES3G5M…", flush=True)
    if not (XES_Q / "test_quelevel.csv").exists():
        out = {"available": False, "reason": "XES3G5M CSVs missing"}
        write_json("followup2_xes3g5m.json", out)
        return out

    q_by_uid, kc_by_uid = load_xes_one_seq_per_uid()
    uids = np.array(sorted(q_by_uid.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uids)
    n_te = max(1, int(0.2 * len(uids)))
    te = set(uids[:n_te].tolist())
    tr = set(uids[n_te:].tolist())
    assert tr.isdisjoint(te)

    q_all = list(q_by_uid.values())
    kc_all = list(kc_by_uid.values())
    q_te = [q_by_uid[u] for u in te]
    kc_te = [kc_by_uid[u] for u in te]

    out = {
        "available": True,
        "dataset": "XES3G5M (NeurIPS 2023; question_level CSVs; longest window per uid)",
        "split": {
            "unit": "uid",
            "n_learners": int(len(uids)),
            "n_train": len(tr),
            "n_test": len(te),
            "seed": SEED,
            "test_frac": 0.2,
            "overlap": 0,
        },
        "share_next_equals_last": {
            "question_all": share_over_sequences(q_all),
            "question_test": share_over_sequences(q_te),
            "kc_all": share_over_sequences(kc_all),
            "kc_test": share_over_sequences(kc_te),
        },
        "three_way_mix": {
            "question_test": three_way(q_te),
            "kc_test": three_way(kc_te),
        },
        "note": (
            "Re-split by uid before any model training. KC sequences expand multi-KC "
            "question tags (underscore-separated) into consecutive KC ids."
        ),
    }
    write_json("followup2_xes3g5m.json", out)
    print(
        "xes q",
        out["share_next_equals_last"]["question_all"]["share"],
        "kc",
        out["share_next_equals_last"]["kc_all"]["share"],
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# summary + cross-dataset table
# ---------------------------------------------------------------------------

def write_summary(rep, forget, reconcile, chance, xes) -> dict:
    # cross-dataset table for docs
    cross = {
        "columns": [
            "dataset_unit",
            "share_next_equals_last",
            "n",
            "continue_frac",
            "revisit_frac",
            "advance_frac",
        ],
        "rows": [
            {
                "dataset_unit": "Junyi ktbd / exercise",
                "share_next_equals_last": rep["junyi_ktbd_exercise"]["share"],
                "n": rep["junyi_ktbd_exercise"]["n"],
                "continue_frac": None,
                "revisit_frac": None,
                "advance_frac": None,
            },
            {
                "dataset_unit": "Junyi ktbd / topic",
                "share_next_equals_last": rep["junyi_ktbd_topic"]["share"],
                "n": rep["junyi_ktbd_topic"]["n"],
            },
            {
                "dataset_unit": "Junyi timed raw / exercise",
                "share_next_equals_last": rep["junyi_timed_raw_exercise"]["share"],
                "n": rep["junyi_timed_raw_exercise"]["n"],
            },
            {
                "dataset_unit": "Junyi timed raw / topic",
                "share_next_equals_last": rep["junyi_timed_raw_topic"]["share"],
                "n": rep["junyi_timed_raw_topic"]["n"],
            },
            {
                "dataset_unit": "ASSISTments / skill",
                "share_next_equals_last": rep["assistments_skill"]["share"],
                "n": rep["assistments_skill"]["n"],
            },
        ],
    }
    # attach three-way from existing ranking / assist / xes
    ranking = json.loads((PHASE / "review_ranking.json").read_text())
    tw = ranking["three_way_mix"]
    cross["rows"][0].update(
        {
            "continue_frac": tw["continue"]["frac"],
            "revisit_frac": tw["revisit_bounce"]["frac"] + tw["revisit_far"]["frac"],
            "advance_frac": tw["advance"]["frac"],
            "three_way_note": "from 800×5 eval cuts (not full ktbd)",
        }
    )
    if (PHASE / "followup_assistments2009.json").exists():
        a = json.loads((PHASE / "followup_assistments2009.json").read_text())
        mix = a.get("three_way_mix_test_skills", {})
        for row in cross["rows"]:
            if row["dataset_unit"] == "ASSISTments / skill":
                row.update(
                    {
                        "continue_frac": mix.get("continue", {}).get("frac"),
                        "revisit_frac": mix.get("revisit", {}).get("frac"),
                        "advance_frac": mix.get("advance", {}).get("frac"),
                    }
                )
    if xes.get("available"):
        cross["rows"].append(
            {
                "dataset_unit": "XES3G5M / question",
                "share_next_equals_last": xes["share_next_equals_last"]["question_all"]["share"],
                "n": xes["share_next_equals_last"]["question_all"]["n"],
                "continue_frac": xes["three_way_mix"]["question_test"]["continue"]["frac"],
                "revisit_frac": xes["three_way_mix"]["question_test"]["revisit"]["frac"],
                "advance_frac": xes["three_way_mix"]["question_test"]["advance"]["frac"],
            }
        )
        cross["rows"].append(
            {
                "dataset_unit": "XES3G5M / KC",
                "share_next_equals_last": xes["share_next_equals_last"]["kc_all"]["share"],
                "n": xes["share_next_equals_last"]["kc_all"]["n"],
                "continue_frac": xes["three_way_mix"]["kc_test"]["continue"]["frac"],
                "revisit_frac": xes["three_way_mix"]["kc_test"]["revisit"]["frac"],
                "advance_frac": xes["three_way_mix"]["kc_test"]["advance"]["frac"],
            }
        )

    write_json("followup2_cross_dataset.json", cross)

    summary = json.loads((PHASE / "review_summary.json").read_text()) if (PHASE / "review_summary.json").exists() else {"claims": {}}
    summary.setdefault("claims", {})
    summary["claims"]["repetition"] = {
        "finding": rep["finding_label"],
        "high_repetition_boolean": rep["high_repetition_boolean"],
        "table_ref": "followup2_repetition.json / followup2_cross_dataset.json",
    }
    summary["claims"]["forgetting"] = {
        "status": forget["claim"],
        "rule": forget["claim_rule"],
        "gap_bins_n": forget["gap_bins_returns_to_seen_exercise"],
        "restricted": forget["restricted_gap_ge_1_day"],
    }
    summary["claims"]["ranking_reconcile"] = {
        "canonical_ragr_r5": reconcile["canonical_run"]["metrics"]["ragr_r5"]["mean"],
        "legacy_polluted_ragr_r5": reconcile["legacy_polluted_rng_run"]["reported_ragr_r5_from_that_run"],
        "reason": reconcile["legacy_polluted_rng_run"]["actual_difference"],
        "effect_labels": reconcile["effect_labels"],
        "practical_threshold_r5": PRACTICAL_DELTA_R5,
    }
    summary["claims"]["duplicate_chance"] = chance["comparison"]
    summary["claims"]["xes3g5m"] = {
        "available": xes.get("available"),
        "question_share": (xes.get("share_next_equals_last") or {}).get("question_all"),
        "kc_share": (xes.get("share_next_equals_last") or {}).get("kc_all"),
    }
    # remove deprecated junyi_specific wording from assist claim if present
    if "7_assistments_second_dataset" in summary["claims"]:
        summary["claims"]["7_assistments_second_dataset"]["repetition_junyi_specific"] = {
            "deprecated": True,
            "replaced_by": rep["finding_label"],
            "see": "followup2_repetition.json",
        }
    write_json("review_summary.json", summary)
    write_json("followup2_summary.json", {"cross_dataset": cross, "claims": summary["claims"]})
    return summary


def main() -> None:
    print("start follow-up 2", flush=True)
    rep = run_repetition_table()
    forget = run_forgetting_power()
    reconcile = run_ranking_reconcile()
    chance = run_duplicate_chance()
    xes = run_xes3g5m()
    write_summary(rep, forget, reconcile, chance, xes)

    lines_path = OUT / "all_phases_results.txt"
    lines = lines_path.read_text().splitlines() if lines_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("F2 ")]
    lines += [
        f"F2 repetition ktbd_ex={rep['junyi_ktbd_exercise']['share']:.3f} "
        f"ktbd_topic={rep['junyi_ktbd_topic']['share']:.3f} "
        f"timed_ex={rep['junyi_timed_raw_exercise']['share']:.3f} "
        f"assist_skill={rep['assistments_skill']['share']:.3f}",
        f"F2 finding: {rep['finding_label']}",
        f"F2 forgetting claim={forget['claim']} gap>=1 n={forget['restricted_gap_ge_1_day']['n_rows']}",
        f"F2 ranking canonical R@5={reconcile['canonical_run']['metrics']['ragr_r5']['mean']:.3f} "
        f"vs legacy_polluted=0.875 overlap={reconcile['legacy_polluted_rng_run']['overlap_with_canonical_sample']}/800 "
        f"Δragr-recent5={reconcile['canonical_run']['paired_ragr_minus_recent5_r5']['mean']} "
        f"label={reconcile['effect_labels']['ragr_minus_recent5']}",
        f"F2 dup chance real={chance['comparison']['real_overlap']} "
        f"pseudo={chance['comparison']['chance_overlap']}",
        f"F2 XES q={xes.get('share_next_equals_last', {}).get('question_all', {}).get('share')} "
        f"kc={xes.get('share_next_equals_last', {}).get('kc_all', {}).get('share')}",
    ]
    lines_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]), flush=True)
    print("done follow-up 2", flush=True)


if __name__ == "__main__":
    main()
