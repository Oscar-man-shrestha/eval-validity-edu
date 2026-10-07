"""
Phase 2 - Semantic Embeddings
NeuroTrace-DAG

What this script does:
1. Loads concepts_clean.csv (287 concepts from Phase 1).
2. Encodes text_for_embedding with paraphrase-multilingual-MiniLM-L12-v2.
   The working concept text is English. This model embeds English, and it is the
   encoder already used for this phase.
3. Saves one vector per concept, in the same row order as concepts_clean.csv.
4. Sanity-checks nearest neighbours, reported separately for concepts that have
   real context text and concepts that fall back to the bare name.

No training. The encoder is used as-is. At 287 concepts, cosine similarity is
a dot product of L2-normalised vectors — FAISS is not used.

Run:
    pip install sentence-transformers
    python phase2_semantic_embeddings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONCEPTS_PATH = ROOT / "concepts_clean.csv"
EDGES_PATH = ROOT / "prerequisite_edges.csv"

# Aligned with concepts_clean.csv row order. Row i of the .npy file is the
# embedding of row i of concepts_clean.csv.
EMBEDDINGS_PATH = ROOT / "concept_embeddings.npy"
NEIGHBORS_PATH = ROOT / "phase2_sanity_neighbors.csv"
SUMMARY_PATH = ROOT / "phase2_sanity_summary.txt"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Five concepts with real context, five with the bare name only.
# Chosen from the prerequisite graph so "related" has a checkable meaning:
# a neighbour that shares an edge with the query.
CONTEXT_QUERIES = [
    "interrupt controller",
    "interrupt mechanism",
    "storage allocation",
    "B-tree",
    "paging",
]
NAME_ONLY_QUERIES = [
    "interrupt service routine",
    "kernel service",
    "Internet Protocol",
    "file allocation",
    "processor scheduling",
]
TOP_K = 5
RANDOM_SEED = 42


def load_concepts(path: Path = CONCEPTS_PATH) -> pd.DataFrame:
    """Load the Phase 1 concept table. Do not substitute the raw concepts.csv."""
    df = pd.read_csv(path)
    required = {"id", "name", "domain", "context_len", "text_for_embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df["text_for_embedding"].isna().any():
        raise ValueError("text_for_embedding has empty cells; Phase 2 cannot encode them.")
    df = df.reset_index(drop=True)
    df["has_context"] = df["context_len"] > 0
    return df


def encode_concepts(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Encode each string with the pretrained multilingual model.

    Vectors are L2-normalised so later phases can use a dot product as cosine
    similarity. The model truncates each input to its max sequence length
    (128 tokens for this checkpoint).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError(
            f"Expected embeddings shape ({len(texts)}, dim), got {embeddings.shape}"
        )
    return embeddings


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    """Load the saved matrix. Row i matches concepts_clean.csv row i."""
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {embeddings.shape}")
    return embeddings


def top_k_neighbors(embeddings: np.ndarray, row: int, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, cosine similarities) of the k nearest concepts, excluding self."""
    if row < 0 or row >= len(embeddings):
        raise IndexError(f"row {row} is outside 0..{len(embeddings) - 1}")
    if k >= len(embeddings):
        raise ValueError("k must be smaller than the number of concepts")
    scores = embeddings @ embeddings[row]
    scores = scores.copy()
    scores[row] = -np.inf
    # argpartition is unsorted; sort only the shortlist.
    shortlist = np.argpartition(-scores, k)[:k]
    order = np.argsort(-scores[shortlist])
    indices = shortlist[order]
    return indices, scores[indices]


def related_names(edges: pd.DataFrame) -> dict[str, set[str]]:
    """Undirected adjacency from confirmed prerequisite edges. Either direction counts as related."""
    related: dict[str, set[str]] = {}
    for source, target in zip(edges["source_concept_name"], edges["target_concept_name"]):
        related.setdefault(source, set()).add(target)
        related.setdefault(target, set()).add(source)
    return related


def _pair_cosine(embeddings: np.ndarray, name_to_row: dict[str, int], left: str, right: str) -> float:
    return float(embeddings[name_to_row[left]] @ embeddings[name_to_row[right]])


def sanity_check(concepts: pd.DataFrame, embeddings: np.ndarray, edges: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Two checks from the Phase 2 spec:

    1. For 10 known concepts (5 with context, 5 name-only), list the top-5
       nearest neighbours and mark which of them share a prerequisite edge.
    2. Compare mean cosine of prerequisite-linked pairs with random pairs,
       split by whether the concepts have context text.
    """
    name_to_row = {name: i for i, name in enumerate(concepts["name"])}
    if len(name_to_row) != len(concepts):
        raise ValueError("concept names are not unique; row alignment would be ambiguous")
    related = related_names(edges)

    queries = [(name, True) for name in CONTEXT_QUERIES] + [(name, False) for name in NAME_ONLY_QUERIES]
    neighbor_rows = []
    for name, expect_context in queries:
        if name not in name_to_row:
            raise ValueError(f"sanity-check concept {name!r} is not in concepts_clean.csv")
        row = name_to_row[name]
        has_context = bool(concepts.at[row, "has_context"])
        if has_context != expect_context:
            raise ValueError(
                f"{name!r} has_context={has_context}, expected {expect_context}. "
                "The sanity-check lists need to be updated against concepts_clean.csv."
            )
        indices, scores = top_k_neighbors(embeddings, row, TOP_K)
        graph_neighbors = related.get(name, set())
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            neighbor_name = concepts.at[idx, "name"]
            neighbor_rows.append({
                "query": name,
                "query_has_context": has_context,
                "query_graph_degree": len(graph_neighbors),
                "rank": rank,
                "neighbor": neighbor_name,
                "neighbor_has_context": bool(concepts.at[idx, "has_context"]),
                "cosine": round(float(score), 4),
                "is_graph_neighbor": neighbor_name in graph_neighbors,
            })
    neighbors = pd.DataFrame(neighbor_rows)

    rng = np.random.default_rng(RANDOM_SEED)
    linked = list(zip(edges["source_concept_name"], edges["target_concept_name"]))
    names = concepts["name"].to_numpy()
    has_context = concepts["has_context"].to_numpy()
    random_pairs = []
    seen = set()
    while len(random_pairs) < len(linked):
        i, j = rng.choice(len(names), size=2, replace=False)
        key = (names[i], names[j])
        if key in seen:
            continue
        seen.add(key)
        random_pairs.append(key)

    def pair_stats(pairs, label: str) -> list[str]:
        lines = [label]
        buckets = {
            "all": [],
            "both_have_context": [],
            "mixed": [],
            "both_name_only": [],
        }
        for left, right in pairs:
            score = _pair_cosine(embeddings, name_to_row, left, right)
            left_ctx = bool(has_context[name_to_row[left]])
            right_ctx = bool(has_context[name_to_row[right]])
            buckets["all"].append(score)
            if left_ctx and right_ctx:
                buckets["both_have_context"].append(score)
            elif left_ctx or right_ctx:
                buckets["mixed"].append(score)
            else:
                buckets["both_name_only"].append(score)
        for bucket, scores in buckets.items():
            if not scores:
                lines.append(f"  {bucket}: n=0")
                continue
            arr = np.asarray(scores, dtype=np.float64)
            lines.append(
                f"  {bucket}: n={len(arr)}  mean={arr.mean():.4f}  median={np.median(arr):.4f}"
            )
        return lines

    hit_lines = ["Top-5 contains at least one prerequisite-graph neighbour:"]
    for label, mask in (
        ("context-rich queries", neighbors["query_has_context"]),
        ("name-only queries", ~neighbors["query_has_context"]),
    ):
        subset = neighbors[mask]
        queries_in_group = subset["query"].nunique()
        hits = subset.groupby("query")["is_graph_neighbor"].any().sum()
        hit_lines.append(f"  {label}: {int(hits)}/{queries_in_group}")

    n_context = int(concepts["has_context"].sum())
    n_name_only = int((~concepts["has_context"]).sum())
    summary_lines = [
        "Phase 2 sanity check",
        f"model: {MODEL_NAME}",
        f"concepts: {len(concepts)}  embeddings: {embeddings.shape}",
        f"context-rich: {n_context}  name-only: {n_name_only}",
        f"vectors are L2-normalised; cosine similarity is a dot product",
        "",
        *hit_lines,
        "",
        *pair_stats(linked, "Prerequisite-linked pairs (should score higher than random if embeddings carry meaning):"),
        "",
        *pair_stats(random_pairs, "Random pairs (same count, seed 42):"),
        "",
        "Limitation: 73 concepts are embedded from the bare name only.",
        "Limitation: this model truncates each text to 128 tokens, so long context is only partly used.",
        "Limitation: a graph neighbour missing from the top-5 is not by itself a failure.",
        "Semantic neighbours and prerequisite neighbours are related but not the same relation.",
    ]
    return neighbors, "\n".join(summary_lines) + "\n"


def main() -> None:
    concepts = load_concepts()
    print(f"Loaded {CONCEPTS_PATH.name}: {len(concepts)} concepts")
    print(
        f"  context-rich: {int(concepts['has_context'].sum())}  "
        f"name-only: {int((~concepts['has_context']).sum())}"
    )

    print(f"Encoding with {MODEL_NAME} ...")
    embeddings = encode_concepts(concepts["text_for_embedding"].tolist())
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved {EMBEDDINGS_PATH.name}  shape={embeddings.shape}  dtype={embeddings.dtype}")

    # Reload through the same path Phase 5 will use, so the sanity check
    # cannot pass on an array that was never written.
    embeddings = load_embeddings()
    if len(embeddings) != len(concepts):
        raise RuntimeError("Saved embedding rows do not match concepts_clean.csv")

    edges = pd.read_csv(EDGES_PATH)
    neighbors, summary = sanity_check(concepts, embeddings, edges)
    neighbors.to_csv(NEIGHBORS_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Saved {NEIGHBORS_PATH.name} and {SUMMARY_PATH.name}\n")
    print(summary)


if __name__ == "__main__":
    main()
