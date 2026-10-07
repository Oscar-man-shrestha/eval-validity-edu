"""
Put the MOOCCubeX concept tables into English and draw the prerequisite DAG.

Chinese originals are kept in concepts_clean_zh.csv and prerequisite_edges_zh.csv.
concepts_clean.csv and prerequisite_edges.csv become the English working copies.
name_zh / text_zh (and the edge equivalents) stay on the English files so a row
can be traced back.

Concept names are the checked glossary in english_concept_names.py.
Longer concept text is translated with Helsinki-NLP/opus-mt-zh-en.
The student log interactions_BBB.csv is already English and is left as it is.
"""

import json
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import pandas as pd

from english_concept_names import ENGLISH_DOMAINS, ENGLISH_NAMES

ROOT = Path(__file__).resolve().parent
ZH_CONCEPTS = ROOT / "concepts_clean_zh.csv"
ZH_EDGES = ROOT / "prerequisite_edges_zh.csv"
CONCEPTS = ROOT / "concepts_clean.csv"
EDGES = ROOT / "prerequisite_edges.csv"
TEXT_CACHE = ROOT / "english_text_cache.json"
DOT_PATH = ROOT / "prerequisite_dag.dot"
PDF_PATH = ROOT / "prerequisite_dag.pdf"
PNG_PATH = ROOT / "prerequisite_dag.png"
SUMMARY_PATH = ROOT / "prerequisite_dag_summary.txt"
DOT_BIN = "/opt/homebrew/bin/dot"


def source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Chinese originals. The first run copies them aside."""
    if not ZH_CONCEPTS.exists():
        shutil.copyfile(CONCEPTS, ZH_CONCEPTS)
        shutil.copyfile(EDGES, ZH_EDGES)
        print(f"Saved Chinese originals as {ZH_CONCEPTS.name} and {ZH_EDGES.name}")
    concepts = pd.read_csv(ZH_CONCEPTS)
    edges = pd.read_csv(ZH_EDGES)
    missing = [n for n in concepts["name"] if n not in ENGLISH_NAMES]
    if missing:
        raise SystemExit(f"Glossary is missing {len(missing)} names, e.g. {missing[:5]}")
    return concepts, edges


def translate_texts(texts: list[str]) -> list[str]:
    """Translate Chinese concept descriptions. Name-only strings are handled by the caller."""
    cache = {}
    if TEXT_CACHE.exists():
        cache = json.loads(TEXT_CACHE.read_text(encoding="utf-8"))

    pending = [t for t in dict.fromkeys(texts) if t not in cache]
    if pending:
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-zh-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        batch_size = 4
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=400,
            )
            generated = model.generate(**encoded, max_new_tokens=280)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for source, target in zip(chunk, decoded):
                cache[source] = " ".join(target.split())
            TEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"  translated {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

    return [cache[t] for t in texts]


def write_english_tables(concepts: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    name_en = concepts["name"].map(ENGLISH_NAMES)
    has_context = concepts["context_len"] > 0
    print(f"Translating {int(has_context.sum())} concept descriptions...")
    translated = translate_texts(concepts.loc[has_context, "text_for_embedding"].tolist())

    text_en = []
    cursor = 0
    for english_name, chinese_text, keep in zip(name_en, concepts["text_for_embedding"], has_context):
        if not keep:
            text_en.append(english_name)
            continue
        body = translated[cursor]
        cursor += 1
        text_en.append(f"{english_name}. {body}".strip())

    english = pd.DataFrame({
        "id": concepts["id"],
        "name": name_en,
        "name_zh": concepts["name"],
        "domain": concepts["domain"].map(ENGLISH_DOMAINS),
        "domain_zh": concepts["domain"],
        "context_len": concepts["context_len"],
        "text_for_embedding": text_en,
        "text_zh": concepts["text_for_embedding"],
    })
    english.to_csv(CONCEPTS, index=False, encoding="utf-8-sig")

    edge_en = pd.DataFrame({
        "source_concept_name": edges["source_concept_name"].map(ENGLISH_NAMES),
        "target_concept_name": edges["target_concept_name"].map(ENGLISH_NAMES),
        "source_zh": edges["source_concept_name"],
        "target_zh": edges["target_concept_name"],
    })
    if edge_en[["source_concept_name", "target_concept_name"]].isna().any().any():
        raise SystemExit("An edge name is missing from the glossary.")
    edge_en.to_csv(EDGES, index=False, encoding="utf-8-sig")
    print(f"Wrote {CONCEPTS.name} and {EDGES.name}")
    return english


def build_graph(concepts: pd.DataFrame) -> nx.DiGraph:
    edges = pd.read_csv(EDGES)
    graph = nx.DiGraph()
    has_context = dict(zip(concepts["name"], concepts["context_len"] > 0))
    for name in concepts["name"]:
        graph.add_node(name, has_context=bool(has_context[name]))
    graph.add_edges_from(zip(edges["source_concept_name"], edges["target_concept_name"]))
    return graph


def write_dot(graph: nx.DiGraph) -> None:
    lines = [
        "digraph prerequisites {",
        "  rankdir=TB;",
        '  bgcolor="white";',
        '  label="NeuroTrace-DAG prerequisite graph\\nAn arrow means: learn the source before the target.\\nGreen nodes have context text. Grey nodes are the bare name only.";',
        "  labelloc=t;",
        "  fontsize=16;",
        '  fontname="Helvetica";',
        "  nodesep=0.22;",
        "  ranksep=0.42;",
        "  splines=true;",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.07,0.04\"];",
        "  edge [color=\"#1F6F6A\", arrowsize=0.55, penwidth=0.7];",
    ]
    for node, data in graph.nodes(data=True):
        if data.get("has_context"):
            fill, stroke = "#E6F4EF", "#1B7A4E"
        else:
            fill, stroke = "#F3F2EF", "#6B6258"
        label = node.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'  "{label}" [label="{label}", fillcolor="{fill}", color="{stroke}"];'
        )
    for source, target in graph.edges():
        left = source.replace("\\", "\\\\").replace('"', '\\"')
        right = target.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{left}" -> "{right}";')
    lines.append("}")
    DOT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_graph() -> None:
    for fmt, dest in (("pdf", PDF_PATH), ("png", PNG_PATH)):
        cmd = [DOT_BIN, f"-T{fmt}", "-Gdpi=110", str(DOT_PATH), "-o", str(dest)]
        print(" ", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"  wrote {dest.name} ({dest.stat().st_size} bytes)", flush=True)


def write_summary(graph: nx.DiGraph) -> None:
    components = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    lines = [
        "Prerequisite DAG",
        f"nodes: {graph.number_of_nodes()}",
        f"edges: {graph.number_of_edges()}",
        f"is_directed_acyclic_graph: {nx.is_directed_acyclic_graph(graph)}",
        f"longest path (edges): {nx.dag_longest_path_length(graph)}",
        f"weakly connected components: {len(components)}  sizes={components}",
        f"with context text: {sum(1 for _, d in graph.nodes(data=True) if d.get('has_context'))}",
        f"name only: {sum(1 for _, d in graph.nodes(data=True) if not d.get('has_context'))}",
        "Arrow direction: source must be learned before target.",
        "Labels are the English concept names. Chinese names are in name_zh.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    concepts_zh, edges_zh = source_tables()
    # Edges can be written before the slow text translation, using the glossary alone.
    preview_edges = pd.DataFrame({
        "source_concept_name": edges_zh["source_concept_name"].map(ENGLISH_NAMES),
        "target_concept_name": edges_zh["target_concept_name"].map(ENGLISH_NAMES),
        "source_zh": edges_zh["source_concept_name"],
        "target_zh": edges_zh["target_concept_name"],
    })
    preview_edges.to_csv(EDGES, index=False, encoding="utf-8-sig")
    preview_concepts = pd.DataFrame({
        "name": concepts_zh["name"].map(ENGLISH_NAMES),
        "context_len": concepts_zh["context_len"],
    })
    graph = nx.DiGraph()
    for name, context_len in zip(preview_concepts["name"], preview_concepts["context_len"]):
        graph.add_node(name, has_context=bool(context_len > 0))
    graph.add_edges_from(zip(preview_edges["source_concept_name"], preview_edges["target_concept_name"]))
    if not nx.is_directed_acyclic_graph(graph):
        raise SystemExit("The English graph has a cycle. Fix the data before drawing it.")
    write_dot(graph)
    print("Rendering the DAG...")
    render_graph()
    write_summary(graph)

    english = write_english_tables(concepts_zh, edges_zh)
    # Confirm the drawn graph still matches the saved tables.
    saved = build_graph(english)
    if set(saved.edges()) != set(graph.edges()):
        raise SystemExit("Saved edges do not match the drawn graph.")
    print("English tables match the drawn graph.")


if __name__ == "__main__":
    main()
