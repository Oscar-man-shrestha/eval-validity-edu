"""Draw the English prerequisite DAG on pages a person can read.

Each row holds at most six concepts, in prerequisite order: an upper
concept is learned before a lower one. The PDF is split into bands of
rows. The PNG is the same layout as one tall image.
"""

from pathlib import Path

import networkx as nx
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import Color, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "prerequisite_dag.pdf"
PNG_PATH = ROOT / "prerequisite_dag.png"
SUMMARY_PATH = ROOT / "prerequisite_dag_summary.txt"

pdfmetrics.registerFont(TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))

NAVY = Color(0.106, 0.227, 0.294)
TEAL = Color(0.122, 0.435, 0.416)
GREEN = Color(0.106, 0.478, 0.306)
GREEN_BG = Color(0.910, 0.957, 0.933)
GREY = Color(0.420, 0.400, 0.380)
GREY_BG = Color(0.953, 0.949, 0.941)
INK = Color(0.145, 0.165, 0.176)

PAGE_W, PAGE_H = 17 * 72, 11 * 72
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 36, 58, 36
COLS = 6
ROWS_PER_PAGE = 7
BOX_W, BOX_H = 168, 40
GAP_X, GAP_Y = 18, 48

PNG_SCALE = 2


def ranks_of(graph: nx.DiGraph) -> list[list[str]]:
    rows = []
    for generation in nx.topological_generations(graph):
        ordered = sorted(generation)
        for start in range(0, len(ordered), COLS):
            rows.append(ordered[start:start + COLS])
    return rows


def load_graph() -> tuple[nx.DiGraph, list[list[str]]]:
    concepts = pd.read_csv(ROOT / "concepts_clean.csv")
    edges = pd.read_csv(ROOT / "prerequisite_edges.csv")
    graph = nx.DiGraph()
    for name, context_len in zip(concepts["name"], concepts["context_len"]):
        graph.add_node(name, has_context=bool(context_len > 0))
    graph.add_edges_from(zip(edges["source_concept_name"], edges["target_concept_name"]))
    acyclic = nx.is_directed_acyclic_graph(graph)
    print("nx.is_directed_acyclic_graph(G) =", acyclic)
    if not acyclic:
        raise SystemExit("The prerequisite graph has a cycle")
    # source_concept_name is the prerequisite. topological_generations
    # lists prerequisites before the concepts that depend on them, and
    # ranks_of keeps that order from top to bottom. An edge that pointed
    # the other way would land the source on a lower row.
    rows = ranks_of(graph)
    row_of = {name: index for index, row in enumerate(rows) for name in row}
    backward = [
        (source, target)
        for source, target in graph.edges()
        if row_of[source] >= row_of[target]
    ]
    if backward:
        raise SystemExit(
            f"{len(backward)} edges do not run from an earlier concept to a later one, "
            f"for example {backward[:3]}"
        )
    return graph, rows


def wrap(text: str, width: int = 26) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def positions(rows: list[list[str]]) -> dict[str, tuple[int, int, int]]:
    """Map a concept to (row, column, page)."""
    placed = {}
    for index, row in enumerate(rows):
        page = index // ROWS_PER_PAGE
        for column, name in enumerate(row):
            placed[name] = (index, column, page)
    return placed


def page_origin(row_on_page: int) -> tuple[float, float]:
    usable_w = COLS * BOX_W + (COLS - 1) * GAP_X
    left = (PAGE_W - usable_w) / 2
    top = PAGE_H - MARGIN_TOP - row_on_page * (BOX_H + GAP_Y)
    return left, top


def draw_pdf(graph: nx.DiGraph, rows: list[list[str]]) -> None:
    placed = positions(rows)
    page_count = (len(rows) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=(PAGE_W, PAGE_H))
    pdf.setTitle("NeuroTrace-DAG prerequisite graph")

    for page in range(page_count):
        first = page * ROWS_PER_PAGE
        last = min(first + ROWS_PER_PAGE, len(rows))
        visible = {name for row in rows[first:last] for name in row}
        pdf.setFillColor(NAVY)
        pdf.rect(0, PAGE_H - 28, PAGE_W, 28, fill=1, stroke=0)
        pdf.setFillColor(white)
        pdf.setFont("Arial-Bold", 10)
        pdf.drawString(MARGIN_X, PAGE_H - 18, "NeuroTrace-DAG prerequisite graph")
        pdf.setFont("Arial", 9)
        pdf.drawRightString(
            PAGE_W - MARGIN_X,
            PAGE_H - 18,
            f"Page {page + 1} of {page_count}   ·   learn the upper concept before the lower one",
        )
        pdf.setFillColor(INK)
        pdf.setFont("Arial", 8)
        pdf.drawString(
            MARGIN_X,
            PAGE_H - 44,
            "Green: context text. Grey: English name only. Arrows are drawn when both concepts are on this page.",
        )

        # Edges first, under the boxes.
        pdf.setStrokeColor(TEAL)
        pdf.setFillColor(TEAL)
        pdf.setLineWidth(0.7)
        for source, target in graph.edges():
            if source not in visible and target not in visible:
                continue
            s_row, s_col, _ = placed[source]
            t_row, t_col, _ = placed[target]
            if source in visible and target in visible:
                s_left, s_top = page_origin(s_row - first)
                t_left, t_top = page_origin(t_row - first)
                x1 = s_left + s_col * (BOX_W + GAP_X) + BOX_W / 2
                y1 = s_top - BOX_H
                x2 = t_left + t_col * (BOX_W + GAP_X) + BOX_W / 2
                y2 = t_top
                pdf.line(x1, y1, x2, y2 + 4)
                path = pdf.beginPath()
                path.moveTo(x2, y2 + 1)
                path.lineTo(x2 - 3.2, y2 + 7)
                path.lineTo(x2 + 3.2, y2 + 7)
                path.close()
                pdf.drawPath(path, fill=1, stroke=0)

        for row_index in range(first, last):
            left, top = page_origin(row_index - first)
            for column, name in enumerate(rows[row_index]):
                x = left + column * (BOX_W + GAP_X)
                y = top - BOX_H
                has_context = graph.nodes[name].get("has_context")
                pdf.setFillColor(GREEN_BG if has_context else GREY_BG)
                pdf.setStrokeColor(GREEN if has_context else GREY)
                pdf.setLineWidth(1)
                pdf.roundRect(x, y, BOX_W, BOX_H, 4, fill=1, stroke=1)
                pdf.setFillColor(INK)
                pdf.setFont("Arial", 8)
                lines = wrap(name)
                text_top = y + BOX_H / 2 + (5 if len(lines) > 1 else 0)
                for line_index, line in enumerate(lines):
                    pdf.drawCentredString(x + BOX_W / 2, text_top - line_index * 10, line)

        pdf.setFillColor(TEAL)
        pdf.rect(0, 0, PAGE_W, 18, fill=1, stroke=0)
        pdf.setFillColor(white)
        pdf.setFont("Arial", 8)
        pdf.drawString(MARGIN_X, 6, f"Rows {first + 1}–{last} of {len(rows)}")
        pdf.showPage()
    pdf.save()


def draw_png(graph: nx.DiGraph, rows: list[list[str]]) -> None:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15 * PNG_SCALE // 2)
    s = PNG_SCALE
    width = int((MARGIN_X * 2 + COLS * BOX_W + (COLS - 1) * GAP_X) * s)
    height = int((70 + len(rows) * (BOX_H + GAP_Y) + 30) * s)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 36 * s], fill=(27, 58, 75))
    draw.text(
        (24 * s, 10 * s),
        "NeuroTrace-DAG prerequisite graph    ·    learn the upper concept before the lower one",
        fill="white",
        font=font,
    )
    placed = {}
    for index, row in enumerate(rows):
        for column, name in enumerate(row):
            x = (MARGIN_X + column * (BOX_W + GAP_X)) * s
            y = (58 + index * (BOX_H + GAP_Y)) * s
            placed[name] = (x, y)
    teal = (31, 111, 106)
    for source, target in graph.edges():
        x1, y1 = placed[source]
        x2, y2 = placed[target]
        start = (x1 + BOX_W * s / 2, y1 + BOX_H * s)
        end = (x2 + BOX_W * s / 2, y2)
        draw.line([start, end], fill=teal, width=max(1, s // 2))
    for name, (x, y) in placed.items():
        has_context = graph.nodes[name].get("has_context")
        fill = (232, 244, 238) if has_context else (243, 242, 239)
        outline = (27, 122, 78) if has_context else (107, 102, 96)
        draw.rounded_rectangle(
            [x, y, x + BOX_W * s, y + BOX_H * s],
            radius=4 * s,
            fill=fill,
            outline=outline,
            width=s,
        )
        lines = wrap(name)
        for line_index, line in enumerate(lines):
            draw.text(
                (x + 8 * s, y + (8 + line_index * 16) * s),
                line,
                fill=(37, 42, 45),
                font=font,
            )
    image.save(PNG_PATH, optimize=True)


def main() -> None:
    graph, rows = load_graph()
    draw_pdf(graph, rows)
    draw_png(graph, rows)
    components = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    summary = "\n".join([
        "Prerequisite DAG",
        f"nodes: {graph.number_of_nodes()}",
        f"edges: {graph.number_of_edges()}",
        f"nx.is_directed_acyclic_graph(G) = {nx.is_directed_acyclic_graph(graph)}",
        f"longest path in edges: {nx.dag_longest_path_length(graph)}",
        f"layout rows: {len(rows)}  columns: {COLS}",
        f"weak components: {len(components)} sizes={components}",
        "PDF pages show bands of these rows. An arrow points from the earlier concept to the later one.",
        f"pdf: {PDF_PATH.name}",
        f"png: {PNG_PATH.name}",
    ])
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"pdf bytes {PDF_PATH.stat().st_size}  png bytes {PNG_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
