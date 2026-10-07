"""
Phase 4 - Prerequisite Graph Module
NeuroTrace-DAG

Loads prerequisite_edges.csv into a networkx DiGraph and exposes
is_unlocked(mastery, concept, threshold), the function Phase 5 needs to
filter candidate recommendations.

This module only needs concepts_clean.csv / prerequisite_edges.csv - it
does NOT depend on the OULAD-to-concept mapping question (see
phase4_oulad_mapping.py and the note in this file's __main__ block).
"""

import networkx as nx
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDGES_PATH = ROOT / "prerequisite_edges.csv"
CONCEPTS_PATH = ROOT / "concepts_clean.csv"

DEFAULT_MASTERY_THRESHOLD = 0.70  # matches the value already named in the pipeline plan


def build_graph(edges_path: Path = EDGES_PATH) -> nx.DiGraph:
    """Build the prerequisite DiGraph. source_concept_name -> target_concept_name
    means source is a prerequisite of target (must be learned first)."""
    edges = pd.read_csv(edges_path)
    G = nx.DiGraph()
    G.add_edges_from(zip(edges["source_concept_name"], edges["target_concept_name"]))
    return G


def is_unlocked(G: nx.DiGraph, mastery: dict, concept: str,
                 threshold: float = DEFAULT_MASTERY_THRESHOLD) -> bool:
    """
    A concept is unlocked if every one of its DIRECT prerequisites has
    mastery >= threshold in the mastery dict. A concept the graph has
    never seen, or with no prerequisites at all, is always unlocked.

    mastery: dict mapping concept name -> a 0-1 score (e.g. Phase 3's
    p_recall for that concept, if the student has studied it; concepts
    they've never touched should simply be absent from the dict, which
    this treats as mastery 0.0, i.e. locked).
    """
    if concept not in G:
        return True
    prereqs = list(G.predecessors(concept))
    if not prereqs:
        return True
    return all(mastery.get(p, 0.0) >= threshold for p in prereqs)


def unlocked_frontier(G: nx.DiGraph, mastery: dict, threshold: float = DEFAULT_MASTERY_THRESHOLD):
    """Convenience: every concept in the graph that is currently unlocked
    for a given mastery dict. Useful for sanity-checking and for Phase 5
    candidate generation."""
    return [c for c in G.nodes if is_unlocked(G, mastery, c, threshold)]


def main():
    G = build_graph()
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")

    # --- Demo / sanity check on a real chain from the actual data ---
    # Pick a concept with exactly one prerequisite to demonstrate clearly.
    demo_concept = None
    demo_prereq = None
    for c in G.nodes:
        preds = list(G.predecessors(c))
        if len(preds) == 1:
            demo_concept, demo_prereq = c, preds[0]
            break

    print(f"\nDemo: '{demo_concept}' requires '{demo_prereq}'")
    print(f"  Locked (no mastery info):        "
          f"{is_unlocked(G, {}, demo_concept)}")
    print(f"  Locked (prereq mastery 0.40):     "
          f"{is_unlocked(G, {demo_prereq: 0.40}, demo_concept)}")
    print(f"  Unlocked (prereq mastery 0.85):   "
          f"{is_unlocked(G, {demo_prereq: 0.85}, demo_concept)}")

    frontier = unlocked_frontier(G, {})
    print(f"\nWith zero mastery info, {len(frontier)} of {G.number_of_nodes()} "
          f"concepts are unlocked (i.e. have no prerequisites at all).")


if __name__ == "__main__":
    main()