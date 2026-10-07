"""Render readable Junyi prerequisite DAG figures via Graphviz `dot`.

Standalone (no sklearn) so it stays fast.

Produces:
  outputs_junyi/figures/dag_quadratic.png
  outputs_junyi/figures/dag_addition.png
  outputs_junyi/figures/dag_overview.png
  outputs_junyi/junyi_quadratic_dag.png  (compat path used by the PDF)
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "junyi_ktbd"
OUT = ROOT / "outputs_junyi"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_names() -> dict[int, str]:
    names = {}
    for line in (DATA / "vertex_id2idx").read_text().splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit(",", 1)
        names[int(idx)] = name.replace("_", " ")
    return names


def load_graph(n_nodes: int) -> nx.DiGraph:
    edges = [tuple(e) for e in json.loads((DATA / "prerequisite.json").read_text())]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)
    return graph


def wrap_label(name: str, width: int = 16) -> str:
    return "\\n".join(textwrap.wrap(name.replace("_", " "), width=width) or [name])


def neighbourhood(graph: nx.DiGraph, seed: int, max_nodes: int = 28) -> set[int]:
    """Two-hop ego neighbourhood (parents/children + their parents/children)."""
    close = {seed}
    close |= set(graph.predecessors(seed))
    close |= set(graph.successors(seed))
    ring = list(close)
    for n in ring:
        if n == seed:
            continue
        close |= set(graph.predecessors(n))
        close |= set(graph.successors(n))
    # Prefer nodes on paths through the seed
    ranked = sorted(
        close,
        key=lambda n: (
            0 if n == seed else 1,
            0 if nx.has_path(graph, n, seed) or nx.has_path(graph, seed, n) else 1,
            -graph.degree(n),
            n,
        ),
    )
    return set(ranked[:max_nodes])


def find_seed(names: dict[int, str], needle: str) -> int:
    needle = needle.lower()
    exact = [i for i, n in names.items() if n.lower() == needle]
    if exact:
        return exact[0]
    hits = [i for i, n in names.items() if needle in n.lower()]
    if not hits:
        raise KeyError(needle)
    return hits[0]


def write_dot(sub: nx.DiGraph, names: dict[int, str], seed: int, path: Path, title: str) -> None:
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        '  bgcolor="white";',
        '  pad="0.4";',
        '  nodesep="0.5";',
        '  ranksep="0.7";',
        "  splines=true;",
        "  overlap=false;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10,',
        '        color="#0f766e", fillcolor="#d0f0eb", fontcolor="#0a1628",',
        '        margin="0.14,0.10"];',
        '  edge [color="#475569", arrowsize=0.8, penwidth=1.35];',
    ]
    for n in sub.nodes:
        label = wrap_label(names[n], 18)
        if n == seed:
            lines.append(
                f'  n{n} [label="{label}", fillcolor="#0d9488", fontcolor="white", '
                f'color="#115e59", penwidth=2.4, fontsize=11];'
            )
        elif sub.in_degree(n) == 0:
            lines.append(f'  n{n} [label="{label}", fillcolor="#e0f2fe", color="#1e4d7b"];')
        elif sub.out_degree(n) == 0:
            lines.append(f'  n{n} [label="{label}", fillcolor="#fef3c7", color="#92400e"];')
        else:
            lines.append(f'  n{n} [label="{label}"];')
    for u, v in sub.edges:
        lines.append(f"  n{u} -> n{v};")
    lines.append('  labelloc="t";')
    lines.append(f'  label="{title}\\narrow = learn source before target  ·  teal = focus  ·  blue = roots  ·  amber = leaves";')
    lines.append('  fontsize=12; fontname="Helvetica-Bold"; fontcolor="#0a1628";')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def render_dot(dot_path: Path, png_path: Path) -> None:
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=180", str(dot_path), "-o", str(png_path)],
        check=True,
    )


def draw_topic(graph, names, needle: str, out_name: str, title: str, max_nodes: int = 24) -> Path:
    seed = find_seed(names, needle)
    nodes = neighbourhood(graph, seed, max_nodes=max_nodes)
    sub = graph.subgraph(nodes).copy()
    isolates = [n for n in list(sub.nodes) if sub.degree(n) == 0 and n != seed]
    sub.remove_nodes_from(isolates)
    dot_path = FIG / f"{out_name}.dot"
    png_path = FIG / f"{out_name}.png"
    write_dot(sub, names, seed, dot_path, title)
    render_dot(dot_path, png_path)
    print(f"wrote {png_path} nodes={sub.number_of_nodes()} edges={sub.number_of_edges()}", flush=True)
    return png_path


def draw_overview(graph: nx.DiGraph, names: dict[int, str]) -> Path:
    degrees = [d for _, d in graph.degree()]
    longest = nx.dag_longest_path(graph)
    path_names = [names[i] for i in longest[:14]]
    if len(longest) > 14:
        path_names.append(f"… +{len(longest) - 14} more")
    comps = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    axes[0].hist(degrees, bins=20, color="#0d9488", edgecolor="white")
    axes[0].set_title(f"Node degree (n={graph.number_of_nodes()})")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")

    axes[1].axis("off")
    axes[1].set_title(f"Longest path ({len(longest) - 1} edges)")
    y = 0.95
    for i, name in enumerate(path_names):
        axes[1].text(0.02, y, f"{i + 1}. {name}", fontsize=7.5, transform=axes[1].transAxes, va="top")
        y -= 0.065

    axes[2].bar(range(min(12, len(comps))), comps[:12], color="#1e4d7b")
    axes[2].set_title(f"Weak components (k={len(comps)})")
    axes[2].set_xlabel("Component rank")
    axes[2].set_ylabel("Nodes")

    fig.suptitle(
        f"Junyi expert DAG · {graph.number_of_edges()} edges · "
        f"acyclic={nx.is_directed_acyclic_graph(graph)}",
        fontsize=11,
        y=1.02,
    )
    out = FIG / "dag_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, flush=True)
    return out


def main() -> None:
    print("loading graph…", flush=True)
    names = load_names()
    graph = load_graph(len(names))
    print(f"nodes={len(names)} edges={graph.number_of_edges()}", flush=True)

    q = draw_topic(
        graph,
        names,
        "solving quadratics by factoring",
        "dag_quadratic",
        "Junyi expert DAG — quadratic neighbourhood",
        max_nodes=22,
    )
    compat = OUT / "junyi_quadratic_dag.png"
    compat.write_bytes(q.read_bytes())
    print("wrote", compat, flush=True)

    draw_topic(
        graph,
        names,
        "addition 2",
        "dag_addition",
        "Junyi expert DAG — early arithmetic neighbourhood",
        max_nodes=20,
    )
    draw_overview(graph, names)
    print("done", flush=True)


if __name__ == "__main__":
    main()
