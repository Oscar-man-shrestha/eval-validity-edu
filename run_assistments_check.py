"""ASSISTments 2009 corrected skill-builder: repetition share + graph-effect check.

- Collapse multi-skill duplicate rows (same order_id → one row, skills joined).
- Learner-disjoint 80/20 split.
- Compare share(next==last) to Junyi ktbd.
- Expert DAG readiness: not testable (no prerequisite map); report mined-edge probe.

Writes: outputs_junyi/phases/followup_assistments2009.json
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from junyi_pipeline import OUT, SEED
from run_all_phases import write_json
from run_claude_fixes import N_BOOT, paired_seq_auc_delta

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv"
MIN_LEN, MAX_LEN = 12, 200
PRACTICAL_DELTA_AUC = 0.005


def collapse_multiskill(df: pd.DataFrame) -> pd.DataFrame:
    """One row per order_id; join skill ids when an attempt tagged with multiple skills."""
    df = df.dropna(subset=["user_id", "order_id", "correct"]).copy()
    df["skill_id"] = df["skill_id"].fillna(-1).astype(int)
    # sort skills for stable collapse key
    g = (
        df.groupby("order_id", sort=False)
        .agg(
            user_id=("user_id", "first"),
            problem_id=("problem_id", "first"),
            correct=("correct", "first"),
            skills=("skill_id", lambda s: tuple(sorted({int(x) for x in s if int(x) >= 0}))),
            n_skill_rows=("skill_id", "size"),
        )
        .reset_index()
    )
    g["skill_key"] = g["skills"].map(lambda t: "_".join(map(str, t)) if t else "none")
    return g


def build_sequences(collapsed: pd.DataFrame, item_col: str):
    """Per-user chronological sequences by order_id."""
    collapsed = collapsed.sort_values(["user_id", "order_id"])
    # map item keys to dense ids
    uniq = {k: i for i, k in enumerate(sorted(collapsed[item_col].unique()))}
    seqs = []
    user_ids = []
    for uid, grp in collapsed.groupby("user_id", sort=False):
        items = [uniq[k] for k in grp[item_col].tolist()]
        ys = [int(c) for c in grp["correct"].tolist()]
        seq = list(zip(items, ys))
        if len(seq) >= MIN_LEN:
            seqs.append(seq[:MAX_LEN])
            user_ids.append(int(uid))
    return seqs, user_ids, uniq


def share_next_eq_last(sequences) -> dict:
    hit = tot = 0
    for seq in sequences:
        for t in range(1, len(seq)):
            tot += 1
            if seq[t][0] == seq[t - 1][0]:
                hit += 1
    return {"share": hit / max(tot, 1), "n": tot, "hits": hit}


def action_mix(sequences, max_seq=5000) -> dict:
    mix = {"continue": 0, "revisit": 0, "advance": 0}
    n = 0
    for seq in sequences[:max_seq]:
        seen = {}
        last = None
        for t, (lab, _y) in enumerate(seq):
            if t >= 1 and last is not None:
                if lab == last:
                    mix["continue"] += 1
                elif lab in seen:
                    mix["revisit"] += 1
                else:
                    mix["advance"] += 1
                n += 1
            seen[lab] = t
            last = lab
    return {k: {"count": v, "frac": v / max(n, 1)} for k, v in mix.items()} | {"n": n}


def mined_edges(train_seqs, n_items: int, min_support: int = 40):
    """Directional skill order: A→B if A appears before B more often than reverse among co-learners."""
    before = np.zeros((n_items, n_items), dtype=np.int32)
    for seq in train_seqs:
        first = {}
        for t, (lab, _) in enumerate(seq):
            if lab not in first:
                first[lab] = t
        labs = list(first.keys())
        for i, a in enumerate(labs):
            for b in labs[i + 1 :]:
                if first[a] < first[b]:
                    before[a, b] += 1
                else:
                    before[b, a] += 1
    edges = []
    parent_lists = [[] for _ in range(n_items)]
    for a in range(n_items):
        for b in range(n_items):
            if a == b:
                continue
            ab, ba = int(before[a, b]), int(before[b, a])
            if ab + ba >= min_support and ab > ba * 1.5:
                edges.append((a, b, ab, ba))
                parent_lists[b].append(a)
    # single strongest parent per node (keeps enough rows for all-parents-observed)
    for b in range(n_items):
        cands = [(a, before[a, b]) for a in parent_lists[b]]
        cands.sort(key=lambda x: -x[1])
        parent_lists[b] = [a for a, _ in cands[:1]]
    return edges, parent_lists


def readiness_probe(train_seqs, te_seqs, parent_lists, rng) -> dict:
    """Same protocol as Junyi readiness (sequence split already done); item acc from train only."""
    item_stats = defaultdict(lambda: [0, 0])
    for seq in train_seqs:
        for c, y in seq:
            item_stats[c][0] += 1
            item_stats[c][1] += int(y)
    item_acc = {c: (s / a if a else 0.5) for c, (a, s) in item_stats.items()}

    def rows(sequences):
        per_seq = []
        for seq in sequences:
            stats, mastery, last_step = {}, {}, {}
            user_a = user_s = 0
            rs = []
            for cut, (label, y) in enumerate(seq):
                if cut >= 1:
                    parents = parent_lists[label]
                    if not parents:
                        pass
                    else:
                        seen_p = [p for p in parents if p in mastery]
                        parents_seen = 1.0 if len(seen_p) == len(parents) else 0.0
                        if parents_seen >= 1.0:
                            ready = float(min(mastery[p] for p in seen_p))
                            a, s = stats.get(label, (0, 0))
                            dt = float(cut - last_step[label]) if label in last_step else 0.0
                            user_acc = user_s / user_a if user_a else 0.5
                            base = [
                                np.log1p(a),
                                (s / a if a else 0.5),
                                np.log1p(dt),
                                item_acc.get(label, 0.5),
                                user_acc,
                                float(cut) / max(len(seq) - 1, 1),
                                parents_seen,
                                float(len(parents)),
                            ]
                            rs.append((base, ready, float(y)))
                a, s = stats.get(label, (0, 0))
                stats[label] = (a + 1, s + int(y))
                last_step[label] = cut
                mastery[label] = stats[label][1] / stats[label][0]
                user_a += 1
                user_s += int(y)
            if rs:
                per_seq.append(rs)
        return per_seq

    tr = rows(train_seqs)
    te = rows(te_seqs)
    if not tr or not te:
        return {"skipped": True, "reason": "insufficient rows with observed parents"}

    def stack(per):
        x0, x1, y, sid = [], [], [], []
        for i, rs in enumerate(per):
            for base, ready, yy in rs:
                x0.append(base)
                x1.append(base + [ready])
                y.append(yy)
                sid.append(i)
        return np.asarray(x0), np.asarray(x1), np.asarray(y), np.asarray(sid)

    x0tr, x1tr, ytr, _ = stack(tr)
    x0te, x1te, yte, ste = stack(te)
    if len(np.unique(yte)) < 2 or len(yte) < 100:
        return {"skipped": True, "n_test": int(len(yte)), "n_train": int(len(ytr))}
    m0 = LogisticRegression(max_iter=500).fit(x0tr, ytr)
    m1 = LogisticRegression(max_iter=500).fit(x1tr, ytr)
    p0 = m0.predict_proba(x0te)[:, 1]
    p1 = m1.predict_proba(x1te)[:, 1]
    y_ps, p0_ps, p1_ps = [], [], []
    for si in sorted(set(ste.tolist())):
        m = ste == si
        y_ps.append(yte[m])
        p0_ps.append(p0[m])
        p1_ps.append(p1[m])
    delta = paired_seq_auc_delta(y_ps, p0_ps, p1_ps, rng, N_BOOT)
    mean = delta.get("mean")
    ci_hi0 = bool(delta.get("ci_above_zero"))
    if mean is not None and mean >= PRACTICAL_DELTA_AUC and ci_hi0:
        status = "supported"
    elif mean is not None and ci_hi0:
        status = "detectable_but_negligible"
    else:
        status = "unresolved"
    return {
        "n_train_rows": int(len(ytr)),
        "n_test_rows": int(len(yte)),
        "auc_base": float(roc_auc_score(yte, p0)),
        "auc_with_ready": float(roc_auc_score(yte, p1)),
        "delta_auc": delta,
        "status": status,
        "practical_threshold_delta_auc": PRACTICAL_DELTA_AUC,
        "graph_type": "mined_single_parent_order_edge_not_expert_dag",
    }


def main() -> None:
    print("ASSISTments 2009 check…", flush=True)
    raw = pd.read_csv(CSV, encoding="ISO-8859-1", low_memory=False)
    n_raw = len(raw)
    collapsed = collapse_multiskill(raw)
    n_multi = int((collapsed["n_skill_rows"] > 1).sum())
    print("collapsed", len(collapsed), "from", n_raw, "multi_skill_attempts", n_multi, flush=True)

    skill_seqs, user_ids, skill_map = build_sequences(collapsed, "skill_key")
    prob_seqs, _, _ = build_sequences(collapsed, "problem_id")
    user_ids = np.array(user_ids)
    rng = np.random.default_rng(SEED)
    # learner-disjoint split
    order = np.arange(len(user_ids))
    rng.shuffle(order)
    n_te = max(1, int(0.2 * len(order)))
    te_idx = set(order[:n_te].tolist())
    tr_skill = [skill_seqs[i] for i in range(len(skill_seqs)) if i not in te_idx]
    te_skill = [skill_seqs[i] for i in range(len(skill_seqs)) if i in te_idx]
    tr_users = [int(user_ids[i]) for i in range(len(user_ids)) if i not in te_idx]
    te_users = [int(user_ids[i]) for i in range(len(user_ids)) if i in te_idx]
    assert set(tr_users).isdisjoint(set(te_users))

    share_skill_all = share_next_eq_last(skill_seqs)
    share_skill_te = share_next_eq_last(te_skill)
    share_prob_all = share_next_eq_last(prob_seqs)
    mix = action_mix(te_skill)

    # Junyi reference from JSON if present
    junyi_share = None
    jt = OUT / "phases" / "review_timed_ktbd.json"
    if jt.exists():
        import json

        junyi_share = json.loads(jt.read_text())["ktbd"]["share_next_equals_last"]["share"]

    edges, parent_lists = mined_edges(tr_skill, len(skill_map))
    ready = readiness_probe(tr_skill, te_skill, parent_lists, rng)

    out = {
        "dataset": "ASSISTments2009-2010 skill-builder corrected (EduData USTC mirror)",
        "source_csv": str(CSV),
        "collapse": {
            "n_raw_rows": n_raw,
            "n_collapsed_attempts": int(len(collapsed)),
            "n_attempts_with_multiple_skills": n_multi,
            "rule": "groupby order_id; skills=sorted unique skill_id joined by _",
        },
        "split": {
            "unit": "learner",
            "n_learners": int(len(user_ids)),
            "n_train_learners": len(tr_users),
            "n_test_learners": len(te_users),
            "overlap_learners": 0,
            "sequence_filter": f"len {MIN_LEN}–{MAX_LEN}",
        },
        "repetition": {
            "share_next_equals_last_skill_all": share_skill_all,
            "share_next_equals_last_skill_test": share_skill_te,
            "share_next_equals_last_problem_all": share_prob_all,
            "junyi_ktbd_share_next_equals_last": junyi_share,
            "abs_delta_vs_junyi_skill_all": (
                abs(share_skill_all["share"] - junyi_share) if junyi_share is not None else None
            ),
            "repetition_junyi_specific": (
                bool(abs(share_skill_all["share"] - junyi_share) > 0.10)
                if junyi_share is not None
                else None
            ),
            "note": (
                "repetition_junyi_specific=True if |share_skill - junyi| > 0.10; "
                "False means similar sticky-continue mass"
            ),
        },
        "three_way_mix_test_skills": mix,
        "expert_dag": {
            "status": "not_testable",
            "reason": "ASSISTments 2009 skill-builder has no expert prerequisite DAG",
        },
        "mined_graph_readiness_probe": {
            "n_mined_edges": len(edges),
            "n_items": len(skill_map),
            "result": ready,
        },
    }
    write_json("followup_assistments2009.json", out)

    # append to summary
    import json

    sp = OUT / "phases" / "review_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
        summary["claims"]["7_assistments_second_dataset"] = {
            "skill_share_next_eq_last": share_skill_all["share"],
            "junyi_share": junyi_share,
            "repetition_junyi_specific": out["repetition"]["repetition_junyi_specific"],
            "expert_dag": "not_testable",
            "mined_readiness_status": ready.get("status"),
        }
        write_json("review_summary.json", summary)

    lines_path = OUT / "all_phases_results.txt"
    lines = lines_path.read_text().splitlines() if lines_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("Assist ")]
    lines += [
        f"Assist skill next==last={share_skill_all['share']:.3f} "
        f"(junyi={junyi_share}) junyi_specific={out['repetition']['repetition_junyi_specific']}",
        f"Assist mix continue={mix['continue']['frac']:.3f} revisit={mix['revisit']['frac']:.3f} "
        f"advance={mix['advance']['frac']:.3f}",
        f"Assist expert_dag=not_testable mined_readiness={ready.get('status')} "
        f"delta={ready.get('delta_auc', {}).get('mean') if isinstance(ready.get('delta_auc'), dict) else None}",
    ]
    lines_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-5:]), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
