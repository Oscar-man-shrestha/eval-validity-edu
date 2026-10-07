"""
NeuroTrace-DAG closed-loop pipeline on Junyi Academy.

Why this dataset: one public corpus has student, exercise, correctness,
and an expert prerequisite DAG. OULAD module BBB cannot join to MOOCCubeX
concepts, so it cannot support per-concept memory or next-concept ranking.

Citation: Chang, Lin, and Chen, EDM 2015, "Modeling Exercise Relationships
in E-Learning: A Unified Approach". Official dump mirrored by EduData
(USTC): http://base.ustc.edu.cn/data/ktbd/junyi/
DataShop: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198

Run:
    python3 junyi_pipeline.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "junyi_ktbd"
OUT = ROOT / "outputs_junyi"
SEED = 0
MIN_LEN = 12
MAX_LEN = 200
HLR_ROWS = 180_000
RANK_TRAIN_POS = 8_000
EVAL_SEQS = 500
EVAL_POS_PER_SEQ = 3
BOOTSTRAP = 400
K_LIST = (5, 10)


def load_names() -> dict[int, str]:
    names = {}
    for line in (DATA / "vertex_id2idx").read_text().splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit(",", 1)
        names[int(idx)] = name.replace("_", " ")
    return names


def load_graph(n_nodes: int) -> tuple[nx.DiGraph, set[tuple[int, int]]]:
    edges = [tuple(e) for e in json.loads((DATA / "prerequisite.json").read_text())]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)
    similar = {(int(e[0]), int(e[1])) for e in json.loads((DATA / "similarity.json").read_text())}
    similar |= {(b, a) for a, b in similar}
    return graph, similar


def load_sequences(path: Path) -> list[list[tuple[int, int]]]:
    sequences = []
    with path.open() as handle:
        for line in handle:
            seq = json.loads(line)
            if seq and isinstance(seq[0][0], list):
                seq = seq[0]
            cleaned = [(int(item), int(correct)) for item, correct in seq[:MAX_LEN]]
            if len(cleaned) >= MIN_LEN:
                sequences.append(cleaned)
    return sequences


def pretty_name(raw: str) -> str:
    return " ".join(part for part in raw.replace("-", " ").split() if part)


def encode_names(names: dict[int, str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    texts = [pretty_name(names[i]) for i in range(len(names))]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def pair_auc(embeddings: np.ndarray, edges: list[tuple[int, int]], rng: np.random.Generator) -> dict:
    linked = [(s, t) for s, t in edges if s != t]
    scores = [float(embeddings[s] @ embeddings[t]) for s, t in linked]
    random_scores = []
    n = embeddings.shape[0]
    while len(random_scores) < len(scores):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            random_scores.append(float(embeddings[i] @ embeddings[j]))
    y = np.array([1] * len(scores) + [0] * len(random_scores))
    x = np.array(scores + random_scores)
    return {
        "n_linked": len(scores),
        "linked_mean": float(np.mean(scores)),
        "random_mean": float(np.mean(random_scores)),
        "pair_auc": float(roc_auc_score(y, x)),
    }


def hlr_features(attempts: int, successes: int, delta: float) -> np.ndarray:
    rate = successes / attempts if attempts else 0.0
    return np.array(
        [1.0, math.log1p(attempts), rate, math.log1p(max(delta, 0.0))],
        dtype=np.float64,
    )


def iter_memory_rows(sequences: list[list[tuple[int, int]]]):
    for seq in sequences:
        last_step: dict[int, int] = {}
        stats: dict[int, list[int]] = {}
        for step, (concept, correct) in enumerate(seq):
            attempts, successes = stats.get(concept, [0, 0])
            if concept in last_step:
                delta = float(step - last_step[concept])
                yield hlr_features(attempts, successes, delta), float(correct), delta
            last_step[concept] = step
            stats[concept] = [attempts + 1, successes + int(correct)]


def fit_hlr(rows_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        half = np.clip(2.0 ** (x @ weights), 1e-3, 1e6)
        delta = np.expm1(x[:, 3])
        return np.clip(2.0 ** (-delta / half), 1e-6, 1 - 1e-6)

    weights = np.zeros(rows_x.shape[1], dtype=np.float64)
    for _ in range(80):
        pred = predict(weights, rows_x)
        grad = rows_x.T @ (pred - y) / len(y)
        weights -= 0.35 * grad
    return weights


def hlr_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    half = np.clip(2.0 ** (x @ weights), 1e-3, 1e6)
    delta = np.expm1(x[:, 3])
    return np.clip(2.0 ** (-delta / half), 1e-6, 1 - 1e-6)


def bootstrap_metric(y: np.ndarray, pred: np.ndarray, fn, rng: np.random.Generator) -> tuple[float, float, float]:
    values = []
    n = len(y)
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            values.append(fn(y[idx], pred[idx]))
        except ValueError:
            continue
    arr = np.array(values)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def evaluate_memory(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict:
    auc, auc_lo, auc_hi = bootstrap_metric(y, pred, roc_auc_score, rng)
    ap, ap_lo, ap_hi = bootstrap_metric(y, pred, average_precision_score, rng)
    return {
        "auc": auc,
        "auc_ci": [auc_lo, auc_hi],
        "pr_auc": ap,
        "pr_auc_ci": [ap_lo, ap_hi],
        "log_loss": float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def unlock_score(graph: nx.DiGraph, mastery: dict[int, float], concept: int) -> float:
    parents = list(graph.predecessors(concept))
    if not parents:
        return 1.0
    return float(np.mean([mastery.get(p, 0.0) for p in parents]))


def history_state(seq: list[tuple[int, int]], end: int):
    last_step: dict[int, int] = {}
    stats: dict[int, list[int]] = {}
    recent = []
    for step, (concept, correct) in enumerate(seq[:end]):
        attempts, successes = stats.get(concept, [0, 0])
        stats[concept] = [attempts + 1, successes + int(correct)]
        last_step[concept] = step
        recent.append(concept)
    return stats, last_step, recent


def all_candidate_features(
    end: int,
    last_concept: int,
    stats: dict[int, list[int]],
    last_step: dict[int, int],
    query: np.ndarray,
    embeddings: np.ndarray,
    graph: nx.DiGraph,
    similar: set[tuple[int, int]],
    popularity: np.ndarray,
    hlr_weights: np.ndarray,
    mastery: dict[int, float],
    parent_lists: list[list[int]],
) -> np.ndarray:
    n = embeddings.shape[0]
    semantic = embeddings @ query
    pop = np.log1p(popularity)
    attempts = np.zeros(n)
    successes = np.zeros(n)
    recency = np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    for concept, (a, s) in stats.items():
        attempts[concept] = a
        successes[concept] = s
    for concept, step in last_step.items():
        recency[concept] = 1.0 / (1.0 + (end - step))
        seen[concept] = True
    rate = np.divide(successes, attempts, out=np.zeros(n), where=attempts > 0)
    delta = np.zeros(n)
    for concept, step in last_step.items():
        delta[concept] = float(end - step)
    x = np.column_stack(
        [np.ones(n), np.log1p(attempts), rate, np.log1p(delta)]
    )
    p_mem = hlr_predict(hlr_weights, x)
    p_mem = np.where(seen, p_mem, 0.55)
    unlock = np.ones(n)
    for concept, parents in enumerate(parent_lists):
        if parents:
            unlock[concept] = float(np.mean([mastery.get(p, 0.0) for p in parents]))
    is_sim = np.zeros(n)
    is_next = np.zeros(n)
    for concept in range(n):
        if (last_concept, concept) in similar:
            is_sim[concept] = 1.0
        if graph.has_edge(last_concept, concept):
            is_next[concept] = 1.0
    return np.column_stack([semantic, 1.0 - p_mem, unlock, pop, recency, is_sim, is_next])


def collect_rank_examples(
    sequences: list[list[tuple[int, int]]],
    embeddings: np.ndarray,
    graph: nx.DiGraph,
    similar: set[tuple[int, int]],
    popularity: np.ndarray,
    hlr_weights: np.ndarray,
    rng: np.random.Generator,
    n_pos: int,
    n_neg: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    parent_lists = [list(graph.predecessors(c)) for c in range(embeddings.shape[0])]
    chosen = rng.choice(len(sequences), size=min(len(sequences), n_pos), replace=False)
    for idx in chosen:
        seq = sequences[int(idx)]
        cut = max(8, int(0.8 * len(seq)) - 1)
        label = seq[cut][0]
        stats, last_step, recent = history_state(seq, cut)
        query = embeddings[recent[-5:]].mean(axis=0)
        query = query / max(np.linalg.norm(query), 1e-8)
        mastery = {
            c: (s / a if a else 0.0) for c, (a, s) in stats.items()
        }
        last_concept = recent[-1]
        feats = all_candidate_features(
            cut, last_concept, stats, last_step, query, embeddings, graph, similar, popularity, hlr_weights, mastery, parent_lists
        )
        xs.append(feats[label])
        ys.append(1.0)
        negatives = set()
        while len(negatives) < n_neg:
            cand = int(rng.integers(0, embeddings.shape[0]))
            if cand != label:
                negatives.add(cand)
        for cand in negatives:
            xs.append(feats[cand])
            ys.append(0.0)
    return np.vstack(xs), np.array(ys)


def ndcg_at_k(rank: int, k: int) -> float:
    if rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def evaluate_ranker(
    sequences: list[list[tuple[int, int]]],
    embeddings: np.ndarray,
    graph: nx.DiGraph,
    similar: set[tuple[int, int]],
    popularity: np.ndarray,
    hlr_weights: np.ndarray,
    ranker,
    rng: np.random.Generator,
    ablation: str = "full",
) -> dict:
    n_items = embeddings.shape[0]
    parent_lists = [list(graph.predecessors(c)) for c in range(n_items)]
    recalls = {k: [] for k in K_LIST}
    ndcgs = {k: [] for k in K_LIST}
    violations = []
    changed = 0
    for seq in sequences:
        positions = np.linspace(8, len(seq) - 1, num=EVAL_POS_PER_SEQ, dtype=int)
        for cut in positions:
            label = seq[cut][0]
            stats, last_step, recent = history_state(seq, cut)
            query = embeddings[recent[-5:]].mean(axis=0)
            query = query / max(np.linalg.norm(query), 1e-8)
            mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
            last_concept = recent[-1]
            feats = all_candidate_features(
                cut, last_concept, stats, last_step, query, embeddings, graph, similar, popularity, hlr_weights, mastery, parent_lists
            )
            if ablation == "no_semantics":
                feats[:, 0] = 0
            elif ablation == "no_memory":
                feats[:, 1] = 0
                feats[:, 4] = 0
            elif ablation == "no_graph":
                feats[:, 2] = 0
                feats[:, 5] = 0
                feats[:, 6] = 0
            elif ablation == "popularity":
                feats = np.zeros_like(feats)
                feats[:, 3] = np.log1p(popularity)
            elif ablation == "semantic":
                keep = feats[:, 0].copy()
                feats = np.zeros_like(feats)
                feats[:, 0] = keep
            elif ablation == "graph":
                keep = feats[:, [2, 6]].copy()
                feats = np.zeros_like(feats)
                feats[:, 2] = keep[:, 0]
                feats[:, 6] = keep[:, 1]
            scores = ranker.decision_function(feats) if ablation == "full" or ablation.startswith("no_") else feats.sum(axis=1)
            if ablation == "full":
                full_top = int(np.argmax(scores))
                no_mem = feats.copy()
                no_mem[:, 1] = 0
                no_mem[:, 4] = 0
                if int(np.argmax(ranker.decision_function(no_mem))) != full_top:
                    changed += 1
            order = np.argsort(-scores)
            rank = int(np.where(order == label)[0][0])
            for k in K_LIST:
                recalls[k].append(1.0 if rank < k else 0.0)
                ndcgs[k].append(ndcg_at_k(rank, k))
            parents = list(graph.predecessors(label))
            if parents and all(mastery.get(p, 0.0) < 0.5 for p in parents):
                violations.append(1.0)
            else:
                violations.append(0.0)
    result = {
        f"recall@{k}": float(np.mean(recalls[k])) for k in K_LIST
    }
    result.update({f"ndcg@{k}": float(np.mean(ndcgs[k])) for k in K_LIST})
    result["prereq_violation_rate"] = float(np.mean(violations))
    result["n_queries"] = len(violations)
    if ablation == "full":
        result["memory_changes_top1_frac"] = changed / max(len(violations), 1)
    return result


def draw_topic_dag(graph: nx.DiGraph, names: dict[int, str], seed_name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    inv = {v: k for k, v in names.items()}
    key = None
    for raw, idx in inv.items():
        if seed_name in raw:
            key = idx
            break
    if key is None:
        key = next(n for n in graph.nodes if graph.degree(n) > 0)
    nodes = {key}
    nodes.update(nx.ancestors(graph, key))
    nodes.update(nx.descendants(graph, key))
    if len(nodes) > 36:
        close = set(graph.predecessors(key)) | set(graph.successors(key)) | {key}
        extra = [n for n in nodes if n not in close]
        nodes = close | set(extra[: max(0, 36 - len(close))])
    sub = graph.subgraph(nodes).copy()
    plt.figure(figsize=(11, 8))
    try:
        pos = nx.nx_agraph.graphviz_layout(sub, prog="dot")
    except Exception:
        pos = nx.spring_layout(sub, k=1.2, seed=0)
    nx.draw_networkx_nodes(sub, pos, node_color="#d8efe8", node_size=1400, edgecolors="#1f7a6d")
    nx.draw_networkx_edges(sub, pos, arrows=True, arrowsize=12, edge_color="#55646f")
    labels = {n: names[n][:22] for n in sub.nodes}
    nx.draw_networkx_labels(sub, pos, labels, font_size=7)
    plt.axis("off")
    plt.title("Junyi prerequisite subgraph (learn source before target)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, similar = load_graph(len(names))
    acyclic = bool(nx.is_directed_acyclic_graph(graph))
    print(f"nx.is_directed_acyclic_graph(G) = {acyclic}")
    if not acyclic:
        raise SystemExit("Junyi prerequisite graph has a cycle")

    train_seq = load_sequences(DATA / "train.json")
    test_seq = load_sequences(DATA / "test.json")
    print(f"sequences train={len(train_seq)} test={len(test_seq)} nodes={len(names)} edges={graph.number_of_edges()}")

    popularity = np.zeros(len(names), dtype=np.float64)
    for seq in train_seq:
        for concept, _ in seq:
            popularity[concept] += 1

    print("Encoding exercise names with all-MiniLM-L6-v2")
    embeddings = encode_names(names)
    np.save(OUT / "concept_embeddings.npy", embeddings)
    emb_stats = pair_auc(embeddings, list(graph.edges()), rng)
    print("embedding pair-AUC", emb_stats)

    print("Building memory rows")
    xs, ys, deltas = [], [], []
    for feat, y, delta in iter_memory_rows(train_seq):
        xs.append(feat)
        ys.append(y)
        deltas.append(delta)
        if len(xs) >= HLR_ROWS:
            break
    x = np.vstack(xs)
    y = np.array(ys)
    split = int(0.8 * len(y))
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]

    mean_pred = np.full_like(y_te, y_tr.mean())
    success_pred = x_te[:, 2]
    attempt_pred = x_te[:, 1]
    logreg = LogisticRegression(max_iter=400).fit(x_tr[:, 1:], y_tr)
    logreg_pred = logreg.predict_proba(x_te[:, 1:])[:, 1]
    ridge = Ridge(alpha=1.0).fit(x_tr[:, 1:], y_tr)
    ridge_pred = np.clip(ridge.predict(x_te[:, 1:]), 0, 1)
    hlr_w = fit_hlr(x_tr, y_tr)
    hlr_pred = hlr_predict(hlr_w, x_te)

    memory_table = {}
    for label, pred in [
        ("mean", mean_pred),
        ("success_rate", success_pred),
        ("log_attempts", attempt_pred),
        ("logistic_regression", logreg_pred),
        ("ridge", ridge_pred),
        ("hlr", hlr_pred),
    ]:
        memory_table[label] = evaluate_memory(y_te, pred, rng)
        print(label, memory_table[label])

    print("Training linear ranker")
    rank_x, rank_y = collect_rank_examples(
        train_seq, embeddings, graph, similar, popularity, hlr_w, rng, RANK_TRAIN_POS
    )
    ranker = LogisticRegression(max_iter=500).fit(rank_x, rank_y)

    eval_idx = rng.choice(len(test_seq), size=min(EVAL_SEQS, len(test_seq)), replace=False)
    eval_seq = [test_seq[int(i)] for i in eval_idx]
    ranking = {}
    for ablation in ["full", "no_memory", "no_graph", "no_semantics", "popularity", "semantic", "graph"]:
        ranking[ablation] = evaluate_ranker(
            eval_seq, embeddings, graph, similar, popularity, hlr_w, ranker, rng, ablation
        )
        print(ablation, ranking[ablation])

    draw_topic_dag(graph, names, "quadratic", OUT / "junyi_quadratic_dag.png")

    summary = {
        "dataset": "Junyi Academy Math Practicing Log (Chang et al., EDM 2015); EduData ktbd-junyi split",
        "why_not_oulad": "OULAD BBB has one placeholder concept and cannot join to MOOCCubeX names.",
        "nodes": len(names),
        "prerequisite_edges": graph.number_of_edges(),
        "nx.is_directed_acyclic_graph": acyclic,
        "longest_path_edges": int(nx.dag_longest_path_length(graph)),
        "weak_components": nx.number_weakly_connected_components(graph),
        "train_sequences_used": len(train_seq),
        "test_sequences_used": len(test_seq),
        "class_balance_memory": float(y.mean()),
        "embedding": emb_stats,
        "memory": memory_table,
        "ranking": ranking,
        "hlr_weights": hlr_w.tolist(),
        "ranker_weights": {
            name: float(w)
            for name, w in zip(
                ["semantic", "forget", "unlock", "popularity", "recency", "similar", "graph_child"],
                ranker.coef_[0],
            )
        },
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "NeuroTrace-DAG Junyi results",
        f"nx.is_directed_acyclic_graph(G) = {acyclic}",
        f"nodes={len(names)} edges={graph.number_of_edges()} longest_path={summary['longest_path_edges']}",
        f"embedding pair-AUC={emb_stats['pair_auc']:.3f} linked={emb_stats['linked_mean']:.3f} random={emb_stats['random_mean']:.3f}",
        "Memory (held-out attempt rows, bootstrap 95% CI):",
    ]
    for name, row in memory_table.items():
        lines.append(
            f"  {name:20s} AUC={row['auc']:.3f} [{row['auc_ci'][0]:.3f},{row['auc_ci'][1]:.3f}] "
            f"PR-AUC={row['pr_auc']:.3f} RMSE={row['rmse']:.3f} MAE={row['mae']:.3f}"
        )
    lines.append("Next-concept ranking:")
    for name, row in ranking.items():
        lines.append(
            f"  {name:14s} R@5={row['recall@5']:.3f} R@10={row['recall@10']:.3f} "
            f"nDCG@10={row['ndcg@10']:.3f} violate={row['prereq_violation_rate']:.3f}"
        )
    (OUT / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
