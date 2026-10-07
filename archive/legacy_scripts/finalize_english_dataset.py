"""
Write English-only working tables and a readable prerequisite DAG.

Reads the Chinese backups only. Does not modify concepts_clean_zh.csv
or prerequisite_edges_zh.csv. Working CSVs have no Chinese columns and
no CJK characters.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from english_concept_names import ENGLISH_DOMAINS, ENGLISH_NAMES

ROOT = Path(__file__).resolve().parent
ZH_CONCEPTS = ROOT / "concepts_clean_zh.csv"
ZH_EDGES = ROOT / "prerequisite_edges_zh.csv"
CONCEPTS = ROOT / "concepts_clean.csv"
EDGES = ROOT / "prerequisite_edges.csv"
CACHE_PATH = ROOT / "english_translation_cache.json"
DOT_PATH = ROOT / "prerequisite_dag.dot"
PDF_PATH = ROOT / "prerequisite_dag.pdf"
PNG_PATH = ROOT / "prerequisite_dag.png"
SUMMARY_PATH = ROOT / "prerequisite_dag_summary.txt"
DOT_BIN = "/opt/homebrew/bin/dot"

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
MAX_PER_RANK = 8


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not cleaned:
        raise ValueError(f"Cannot build an id slug from {text!r}")
    return cleaned


def english_id(name: str, domain: str, used: set[str]) -> str:
    base = f"K_{slug(name)}_{slug(domain)}"
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def translate_google(text: str) -> str:
    import translators as ts

    last_error = None
    for attempt in range(5):
        try:
            out = ts.translate_text(
                text,
                translator="google",
                from_language="zh",
                to_language="en",
            )
            if out and out.strip():
                return " ".join(out.split())
        except Exception as exc:  # network and rate-limit errors vary by backend
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Translation failed: {last_error}")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def translated_body(chinese: str, cache: dict) -> str:
    if chinese in cache:
        return cache[chinese]
    english = translate_google(chinese)
    cache[chinese] = english
    save_cache(cache)
    return english


def strip_leftover_cjk(text: str, cache: dict) -> str:
    """Re-translate any Chinese fragment a first pass left behind."""
    if not CJK.search(text):
        return text

    def replace(match: re.Match) -> str:
        fragment = match.group(0)
        if fragment in ENGLISH_NAMES:
            return ENGLISH_NAMES[fragment]
        if fragment in ENGLISH_DOMAINS:
            return ENGLISH_DOMAINS[fragment]
        return translated_body(fragment, cache)

    cleaned = CJK_RUN.sub(lambda m: " " + replace(m) + " ", text)
    return " ".join(cleaned.split())


def build_tables() -> pd.DataFrame:
    concepts = pd.read_csv(ZH_CONCEPTS)
    edges = pd.read_csv(ZH_EDGES)
    missing = sorted(set(concepts["name"]) - set(ENGLISH_NAMES))
    if missing:
        raise SystemExit(f"Glossary missing {missing[:8]}")
    unknown_domains = sorted(set(concepts["domain"]) - set(ENGLISH_DOMAINS))
    if unknown_domains:
        raise SystemExit(f"Unknown domains {unknown_domains}")
    english_values = list(ENGLISH_NAMES[n] for n in concepts["name"])
    if len(english_values) != len(set(english_values)):
        raise SystemExit("English concept names are not unique")

    cache = load_cache()
    names = []
    domains = []
    texts = []
    ids = []
    used_ids: set[str] = set()
    total = int((concepts["context_len"] > 0).sum())
    done = 0
    for row in concepts.itertuples(index=False):
        name = ENGLISH_NAMES[row.name]
        domain = ENGLISH_DOMAINS[row.domain]
        names.append(name)
        domains.append(domain)
        ids.append(english_id(name, domain, used_ids))
        if row.context_len > 0:
            done += 1
            print(f"  text {done}/{total}: {name}", flush=True)
            body = translated_body(str(row.text_for_embedding), cache)
            body = strip_leftover_cjk(body, cache)
            if body.lower().startswith(name.lower()):
                texts.append(body)
            else:
                texts.append(f"{name}. {body}")
        else:
            texts.append(name)

    english = pd.DataFrame({
        "id": ids,
        "name": names,
        "domain": domains,
        "context_len": concepts["context_len"].astype(int),
        "text_for_embedding": texts,
    })
    leftovers = []
    for column in english.columns:
        for value in english[column].astype(str):
            if CJK.search(value):
                leftovers.append((column, value[:80]))
    if leftovers:
        raise SystemExit(f"CJK still present in {len(leftovers)} cells, e.g. {leftovers[:5]}")

    english.to_csv(CONCEPTS, index=False, encoding="utf-8")
    edge_en = pd.DataFrame({
        "source_concept_name": edges["source_concept_name"].map(ENGLISH_NAMES),
        "target_concept_name": edges["target_concept_name"].map(ENGLISH_NAMES),
    })
    if edge_en.isna().any().any():
        raise SystemExit("An edge endpoint is missing from the glossary")
    edge_blob = "\n".join(edge_en["source_concept_name"] + edge_en["target_concept_name"])
    if CJK.search(edge_blob):
        raise SystemExit("CJK still present in prerequisite edges")
    edge_en.to_csv(EDGES, index=False, encoding="utf-8")
    print(f"Wrote {CONCEPTS.name} and {EDGES.name} with no Chinese columns")
    return english


def width_limited_ranks(graph: nx.DiGraph) -> list[list[str]]:
    ranks = []
    for generation in nx.topological_generations(graph):
        ordered = sorted(generation)
        for start in range(0, len(ordered), MAX_PER_RANK):
            ranks.append(ordered[start:start + MAX_PER_RANK])
    return ranks


def quote(label: str) -> str:
    return '"' + label.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_dot(graph: nx.DiGraph, ranks: list[list[str]]) -> None:
    lines = [
        "digraph prerequisites {",
        "  rankdir=TB;",
        '  bgcolor="white";',
        '  pad="0.4";',
        '  label="NeuroTrace-DAG prerequisite graph\\nAn arrow means: learn the source before the target.\\nGreen: the concept has context text. Grey: the bare English name only.";',
        "  labelloc=t;",
        "  fontsize=18;",
        '  fontname="Helvetica";',
        "  nodesep=0.35;",
        "  ranksep=0.55;",
        "  splines=polyline;",
        '  page="17,11";',
        "  pagedir=TB;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.08,0.05"];',
        '  edge [color="#1F6F6A", arrowsize=0.6, penwidth=0.8];',
    ]
    for node, data in graph.nodes(data=True):
        fill, stroke = ("#E6F4EF", "#1B7A4E") if data.get("has_context") else ("#F3F2EF", "#6B6258")
        lines.append(f"  {quote(node)} [fillcolor=\"{fill}\", color=\"{stroke}\"];")
    for index, rank in enumerate(ranks):
        members = " ".join(quote(node) for node in rank)
        lines.append(f"  {{ rank=same; {members}; }}")
        if index + 1 < len(ranks):
            anchor = rank[0]
            for node in ranks[index + 1]:
                lines.append(
                    f"  {quote(anchor)} -> {quote(node)} [style=invis, weight=8, constraint=true];"
                )
    for source, target in graph.edges():
        lines.append(f"  {quote(source)} -> {quote(target)};")
    lines.append("}")
    DOT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render() -> None:
    for fmt, dest in (("pdf", PDF_PATH), ("png", PNG_PATH)):
        subprocess.run(
            [DOT_BIN, f"-T{fmt}", "-Gdpi=120", str(DOT_PATH), "-o", str(dest)],
            check=True,
        )
        print(f"Wrote {dest.name} ({dest.stat().st_size} bytes)")


def summarize(graph: nx.DiGraph, ranks: list[list[str]]) -> None:
    components = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    lines = [
        "Prerequisite DAG",
        f"nodes: {graph.number_of_nodes()}",
        f"edges: {graph.number_of_edges()}",
        f"acyclic: {nx.is_directed_acyclic_graph(graph)}",
        f"longest path in edges: {nx.dag_longest_path_length(graph)}",
        f"layout rows: {len(ranks)}  max nodes on a row: {max(len(r) for r in ranks)}",
        f"weak components: {len(components)} sizes={components}",
        "Arrow direction: learn the source before the target.",
        "Labels are English. The wide single-page layout was replaced with rows of at most 8 concepts.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def draw(concepts: pd.DataFrame) -> None:
    edges = pd.read_csv(EDGES)
    graph = nx.DiGraph()
    has_context = dict(zip(concepts["name"], concepts["context_len"] > 0))
    for name in concepts["name"]:
        graph.add_node(name, has_context=bool(has_context[name]))
    graph.add_edges_from(zip(edges["source_concept_name"], edges["target_concept_name"]))
    if not nx.is_directed_acyclic_graph(graph):
        raise SystemExit("The graph has a cycle")
    ranks = width_limited_ranks(graph)
    write_dot(graph, ranks)
    render()
    summarize(graph, ranks)


def main() -> None:
    print("Building English tables from the Chinese backups...")
    concepts = build_tables()
    print("Drawing the width-limited DAG...")
    draw(concepts)


if __name__ == "__main__":
    main()
