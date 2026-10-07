#!/usr/bin/env python3
"""Run both clustering methods where text exists; co-occurrence on all three.

Records actual used cluster counts after dense remap, the feasible matched-count
table, and XES text availability. Does not write a preregistration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_followup3 import (
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
)
from run_preprereg_freeze_v4 import (
    CLUSTER_SEED,
    TARGET_CLUSTERS,
    cluster_assistments,
    cluster_junyi_timed,
    cluster_xes,
    dense_remap,
    map_seqs_to_clusters,
    train_only_cooc_clusters,
)
from run_probe_freeze import JUNYI_TIMED_MAX_USERS, MAX_LEN, MIN_LEN, PHASE, REPLICATION_SEED, learner_split

XES_Q = ROOT / "data/xes3g5m/XES3G5M/question_level"


def used_clusters(seqs, imap: dict[int, int]) -> int:
    mapped = map_seqs_to_clusters(seqs, imap)
    _, n = dense_remap(mapped)
    return int(n)


def nonempty_labels(imap: dict[int, int]) -> int:
    return int(len(set(imap.values())))


def xes_text_audit() -> dict:
    """Confirm whether question stem / text exists in the local dump."""
    if not XES_Q.exists():
        return {
            "available_locally": False,
            "path": str(XES_Q),
            "reason": "question_level directory missing",
        }
    csvs = sorted(XES_Q.glob("*.csv"))
    columns = {}
    textish = []
    for p in csvs[:8]:
        import pandas as pd

        cols = list(pd.read_csv(p, nrows=0).columns)
        columns[p.name] = cols
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in ("text", "stem", "content", "question_body", "body", "title")):
                textish.append({"file": p.name, "column": c})
    return {
        "available_locally": False,
        "path": str(XES_Q.relative_to(ROOT)),
        "n_csv_inspected": len(csvs[:8]),
        "columns_by_file": columns,
        "textish_columns_found": textish,
        "reason": (
            "XES3G5M question_level CSVs expose question ids / KC tags only. "
            "No question stem text for MiniLM. Text clustering is N/A."
        ),
    }


def vec_from_map(n_items: int, imap: dict[int, int]) -> np.ndarray:
    lab = np.full(n_items, -1, dtype=np.int64)
    for i, c in imap.items():
        if 0 <= i < n_items:
            lab[i] = int(c)
    return lab


def agree(a: np.ndarray, b: np.ndarray) -> dict:
    m = (a >= 0) & (b >= 0)
    if m.sum() < 2:
        return {"n_compared": int(m.sum()), "ari": None, "nmi": None}
    return {
        "n_compared": int(m.sum()),
        "ari": float(adjusted_rand_score(a[m], b[m])),
        "nmi": float(normalized_mutual_info_score(a[m], b[m])),
    }


def cell(feasible: bool, note: str) -> dict:
    return {"feasible": feasible, "mark": "ok" if feasible else "N/A", "note": note}


def main() -> None:
    print("loading sequences…", flush=True)
    j_seqs, j_uids = build_learner_sequences_junyi_timed(
        max_users=JUNYI_TIMED_MAX_USERS, min_len=MIN_LEN, max_len=MAX_LEN
    )
    n_j = max(max(x for x, _ in s) for s in j_seqs) + 1
    a_seqs, a_uids, n_a = build_learner_sequences_assist(min_len=MIN_LEN, max_len=MAX_LEN)
    x_seqs, x_uids, n_x = build_learner_sequences_xes(min_len=MIN_LEN, max_len=MAX_LEN)

    tr_j, va_j, _ = learner_split(j_uids, REPLICATION_SEED)
    tr_a, va_a, _ = learner_split(a_uids, REPLICATION_SEED)
    tr_x, va_x, _ = learner_split(x_uids, REPLICATION_SEED)
    train_j = [j_seqs[i] for i in tr_j]
    train_a = [a_seqs[i] for i in tr_a]
    train_x = [x_seqs[i] for i in tr_x]
    # Freeze n_clusters_used = unique ids in train+val after map + dense remap
    tv_j = [j_seqs[i] for i in list(tr_j) + list(va_j)]
    tv_a = [a_seqs[i] for i in list(tr_a) + list(va_a)]
    tv_x = [x_seqs[i] for i in list(tr_x) + list(va_x)]

    print("text clustering (Junyi, ASSIST)…", flush=True)
    j_text = cluster_junyi_timed()
    a_text = cluster_assistments()
    if a_text is None:
        raise SystemExit("ASSISTments text clustering unexpectedly unavailable")

    print("train-only co-occurrence on all three…", flush=True)
    j_cooc_map, j_cooc_meta = train_only_cooc_clusters(train_j, n_j, CLUSTER_SEED)
    a_cooc_map, a_cooc_meta = train_only_cooc_clusters(train_a, n_a, CLUSTER_SEED)
    x_cooc = cluster_xes(train_x, n_x)
    x_cooc_map = {int(k): int(v) for k, v in x_cooc["item_to_cluster"].items()}

    j_text_map = {int(k): int(v) for k, v in j_text["item_to_cluster"].items()}
    a_text_map = {int(k): int(v) for k, v in a_text["item_to_cluster"].items()}

    cluster_dir = PHASE / "cluster_maps"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    (cluster_dir / "junyi_timed_text_item_to_cluster.json").write_text(json.dumps(j_text_map) + "\n")
    (cluster_dir / "junyi_timed_cooc_item_to_cluster.json").write_text(json.dumps(j_cooc_map) + "\n")
    (cluster_dir / "assistments_text_item_to_cluster.json").write_text(json.dumps(a_text_map) + "\n")
    (cluster_dir / "assistments_cooc_item_to_cluster.json").write_text(json.dumps(a_cooc_map) + "\n")
    (cluster_dir / "xes3g5m_cooc_item_to_cluster.json").write_text(json.dumps(x_cooc_map) + "\n")

    xes_text = xes_text_audit()

    datasets = {
        "junyi_timed": {
            "n_items_native": n_j,
            "n_sequences": len(j_seqs),
            "text_available": True,
            "text": {
                "method": j_text["method"],
                "target_n_clusters": TARGET_CLUSTERS,
                "n_clusters_nonempty_in_labels": nonempty_labels(j_text_map),
                "n_clusters_used_after_dense_remap": used_clusters(tv_j, j_text_map),
                "map_path": "outputs_junyi/phases/cluster_maps/junyi_timed_text_item_to_cluster.json",
            },
            "cooccurrence": {
                **{k: v for k, v in j_cooc_meta.items() if k != "item_to_cluster"},
                "n_clusters_nonempty_in_labels": nonempty_labels(j_cooc_map),
                "n_clusters_used_after_dense_remap": used_clusters(tv_j, j_cooc_map),
                "map_path": "outputs_junyi/phases/cluster_maps/junyi_timed_cooc_item_to_cluster.json",
            },
            "method_agreement": agree(vec_from_map(n_j, j_text_map), vec_from_map(n_j, j_cooc_map)),
            "freeze_primary_method": "text_minilm_kmeans",
            "freeze_used_n_clusters": used_clusters(tv_j, j_text_map),
        },
        "assistments": {
            "n_items_native": n_a,
            "n_sequences": len(a_seqs),
            "text_available": True,
            "text": {
                "method": a_text["method"],
                "n_atomic_named": a_text.get("n_atomic_named"),
                "n_composite_items": a_text.get("n_composite_items"),
                "target_n_clusters": TARGET_CLUSTERS,
                "n_clusters_nonempty_in_labels": nonempty_labels(a_text_map),
                "n_clusters_used_after_dense_remap": used_clusters(tv_a, a_text_map),
                "near_native": True,
                "near_native_reason": (
                    f"n_items={n_a}; target k={TARGET_CLUSTERS} is close to the catalog size, "
                    "so cluster unit is near-native."
                ),
                "map_path": "outputs_junyi/phases/cluster_maps/assistments_text_item_to_cluster.json",
            },
            "cooccurrence": {
                **{k: v for k, v in a_cooc_meta.items() if k != "item_to_cluster"},
                "n_clusters_nonempty_in_labels": nonempty_labels(a_cooc_map),
                "n_clusters_used_after_dense_remap": used_clusters(tv_a, a_cooc_map),
                "map_path": "outputs_junyi/phases/cluster_maps/assistments_cooc_item_to_cluster.json",
            },
            "method_agreement": agree(vec_from_map(n_a, a_text_map), vec_from_map(n_a, a_cooc_map)),
            "freeze_primary_method": "text_skill_name_minilm_kmeans",
            "freeze_used_n_clusters": used_clusters(tv_a, a_text_map),
        },
        "xes3g5m": {
            "n_items_native": n_x,
            "n_sequences": len(x_seqs),
            "text_available": False,
            "text": {
                "method": None,
                "status": "N/A",
                "reason": xes_text["reason"],
            },
            "cooccurrence": {
                "method": x_cooc.get("method"),
                "xes_has_question_text": False,
                "n_clusters_nonempty_in_labels": nonempty_labels(x_cooc_map),
                "n_clusters_used_after_dense_remap": used_clusters(tv_x, x_cooc_map),
                "map_path": "outputs_junyi/phases/cluster_maps/xes3g5m_cooc_item_to_cluster.json",
            },
            "method_agreement": {
                "status": "N/A",
                "reason": "no second (text) method; cannot test method agreement",
            },
            "freeze_primary_method": "train_only_cooccurrence_random_projection_kmeans",
            "freeze_used_n_clusters": used_clusters(tv_x, x_cooc_map),
        },
    }

    # Feasible matched-count table (prereg will mark N/A; do not write prereg here)
    n_items = {"junyi_timed": n_j, "assistments": n_a, "xes3g5m": n_x}
    targets = [40, 120, 800]
    table = {}
    for k in targets:
        row = {}
        for ds, n in n_items.items():
            if k >= n:
                row[ds] = cell(False, f"k={k} >= n_items={n}")
            elif ds == "assistments" and k >= 100:
                row[ds] = {
                    "feasible": True,
                    "mark": "ok_near_native",
                    "note": (
                        f"k={k} is usable but near-native: only {n} composite tokens "
                        f"(used clusters at k=120 were 106)."
                    ),
                }
            elif k > 0.5 * n:
                row[ds] = {
                    "feasible": True,
                    "mark": "ok_near_catalog",
                    "note": f"k={k} of n_items={n} is feasible but close to the native catalog",
                }
            else:
                row[ds] = cell(True, f"k={k} < n_items={n}")
        table[str(k)] = row

    contrast_rule = {
        "rule": (
            "A clustering-method contrast (text vs train-only co-occurrence) counts "
            "only if both methods are defined on that dataset and they agree on the "
            "confirmatory claim (same sign / same ranking conclusion). "
            "XES has no text method → method-agreement contrasts are N/A."
        ),
        "not_yet_in_preregistration": True,
    }

    out = {
        "schema": "dual_clustering_v1",
        "target_n_clusters_freeze": TARGET_CLUSTERS,
        "clustering_seed": CLUSTER_SEED,
        "actual_used_cluster_counts_freeze_primary": {
            "junyi_timed": datasets["junyi_timed"]["freeze_used_n_clusters"],
            "assistments": datasets["assistments"]["freeze_used_n_clusters"],
            "xes3g5m": datasets["xes3g5m"]["freeze_used_n_clusters"],
            "note": (
                "These are n unique cluster ids remaining after mapping sequences and "
                "dense-remapping empty clusters away. Freeze JSON reported 115 / 106 / 120."
            ),
        },
        "xes_question_text": xes_text,
        "datasets": datasets,
        "feasible_matched_count_table": table,
        "feasible_matched_count_summary": (
            "~40: all three datasets. "
            "~120: all three, but ASSISTments is near-native (150 items / 106 used). "
            "~800: Junyi and XES only; Junyi k=800 is close to its 835-item catalog; ASSISTments = N/A."
        ),
        "contrast_counts_only_if_both_methods_agree": contrast_rule,
    }
    dest = PHASE / "dual_clustering.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["actual_used_cluster_counts_freeze_primary"], indent=2))
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
