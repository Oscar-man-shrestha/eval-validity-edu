"""Build NeuroTrace-DAG_Pipeline.pdf from the agreed methodology and completed phases."""

from reportlab.lib.colors import Color, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    Flowable,
)

pdfmetrics.registerFont(TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("ArialUni", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))

NAVY = Color(0.106, 0.227, 0.294)
TEAL = Color(0.122, 0.435, 0.416)
GREEN = Color(0.106, 0.478, 0.306)
GREEN_BG = Color(0.910, 0.957, 0.933)
AMBER = Color(0.541, 0.416, 0.184)
AMBER_BG = Color(0.976, 0.953, 0.894)
SLATE = Color(0.345, 0.329, 0.310)
SLATE_BG = Color(0.965, 0.961, 0.953)
INK = Color(0.145, 0.165, 0.176)
MUTED = Color(0.333, 0.376, 0.400)
LINE = Color(0.820, 0.855, 0.843)
RULE = Color(0.106, 0.227, 0.294)
PAPER = Color(0.973, 0.980, 0.976)

PAGE_W, PAGE_H = A4
MARGIN_L = 16 * mm
MARGIN_R = 16 * mm
MARGIN_T = 18 * mm
MARGIN_B = 16 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R


def S(name, **kw):
    base = dict(
        fontName="Arial",
        fontSize=10,
        leading=14,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "h1": S("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=8, spaceAfter=6, keepWithNext=True),
    "h2": S("h2", fontName="Arial-Bold", fontSize=12, leading=16, textColor=TEAL, spaceBefore=10, spaceAfter=4, keepWithNext=True),
    "body": S("body", fontSize=10, leading=14, spaceAfter=6),
    "small": S("small", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=3),
    "bullet": S("bullet", fontSize=10, leading=13.5, leftIndent=2),
    "cell": S("cell", fontName="ArialUni", fontSize=8.5, leading=11.5),
    "cell_b": S("cell_b", fontName="Arial-Bold", fontSize=8.5, leading=11.5, textColor=NAVY),
    "th": S("th", fontName="Arial-Bold", fontSize=8, leading=10.5, textColor=white),
    "formula": S("formula", fontName="ArialUni", fontSize=10, leading=14, textColor=NAVY, alignment=TA_CENTER),
    "caption": S("caption", fontName="Arial", fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=2, spaceAfter=8),
    "status": S("status", fontName="Arial-Bold", fontSize=9, leading=12, textColor=GREEN),
    "cover_kicker": S("cover_kicker", fontName="Arial-Bold", fontSize=9, leading=12, textColor=TEAL),
    "cover_title": S("cover_title", fontName="Arial-Bold", fontSize=22, leading=26, textColor=NAVY, spaceBefore=2, spaceAfter=2),
    "cover_sub": S("cover_sub", fontName="Arial", fontSize=11, leading=15, textColor=INK, spaceBefore=2, spaceAfter=4),
}


def P(text, style="body"):
    return Paragraph(text, STYLES[style])


def bullets(items):
    flow = []
    for item in items:
        flow.append(Paragraph("•  " + item, STYLES["bullet"]))
        flow.append(Spacer(1, 2))
    flow.append(Spacer(1, 4))
    return flow


def section_title(text):
    return KeepTogether([Spacer(1, 4), P(text, "h1"), rule()])


def rule():
    data = [[""]]
    t = Table(data, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.0, TEAL),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def make_table(headers, rows, col_widths):
    head = [Paragraph(h, STYLES["th"]) for h in headers]
    body = []
    for row in rows:
        body.append([Paragraph(str(cell), STYLES["cell"]) for cell in row])
    table = Table([head] + body, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, PAPER]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    table.setStyle(TableStyle(style_cmds))
    return table


def formula_box(text):
    inner = Paragraph(text, STYLES["formula"])
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


class PipelineDiagram(Flowable):
    """Vertical pipeline: sources → phase 1 → three files → phases 2/3/4 → 5 → 6."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.height = 274

    def _box(self, c, x, y, w, h, title, subtitle, kind):
        fills = {"done": GREEN_BG, "ready": AMBER_BG, "blocked": SLATE_BG, "file": white, "source": white}
        strokes = {"done": GREEN, "ready": AMBER, "blocked": SLATE, "file": TEAL, "source": NAVY}
        c.setFillColor(fills[kind])
        c.setStrokeColor(strokes[kind])
        c.setLineWidth(1.1)
        c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Arial-Bold", 8)
        c.drawCentredString(x + w / 2, y + h / 2 + (5 if subtitle else 0), title)
        if subtitle:
            c.setFillColor(MUTED)
            c.setFont("Arial", 7)
            c.drawCentredString(x + w / 2, y + h / 2 - 8, subtitle)

    def _arrow(self, c, x1, y1, x2, y2):
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.0)
        c.line(x1, y1, x2, y2)
        # small downward triangle at (x2, y2) assuming y2 < y1
        path = c.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(x2 - 3.2, y2 + 5)
        path.lineTo(x2 + 3.2, y2 + 5)
        path.close()
        c.drawPath(path, fill=1, stroke=0)

    def draw(self):
        c = self.canv
        w = self.width
        gap = 8
        box_w = (w - 2 * gap) / 3
        h_src, h_phase, h_file, h_mid, h_late = 32, 36, 30, 42, 32
        y6 = 2
        y5 = y6 + h_late + 12
        ymid = y5 + h_late + 14
        yfile = ymid + h_mid + 12
        y1 = yfile + h_file + 12
        ysrc = y1 + h_phase + 12

        # sources
        src_w = (w - gap) / 2
        self._box(c, 0, ysrc, src_w, h_src, "OULAD", "student logs, module BBB", "source")
        self._box(c, src_w + gap, ysrc, src_w, h_src, "MOOCCubeX", "concepts + prerequisite pairs", "source")
        self._arrow(c, src_w / 2, ysrc, w / 2 - 18, y1 + h_phase)
        self._arrow(c, src_w + gap + src_w / 2, ysrc, w / 2 + 18, y1 + h_phase)

        self._box(c, 0, y1, w, h_phase, "Phase 1  ·  Data preprocessing", "Complete  ·  clean tables for every later phase", "done")

        files = [
            ("interactions_BBB.csv", "42,636 rows  ·  Phase 3"),
            ("concepts_clean.csv", "287 concepts  ·  Phase 2"),
            ("prerequisite_edges.csv", "725 edges  ·  Phase 4"),
        ]
        for i, (title, sub) in enumerate(files):
            x = i * (box_w + gap)
            self._arrow(c, w / 2, y1, x + box_w / 2, yfile + h_file)
            self._box(c, x, yfile, box_w, h_file, title, sub, "file")
            self._arrow(c, x + box_w / 2, yfile, x + box_w / 2, ymid + h_mid)

        mids = [
            ("Phase 2", "Semantic embeddings", "done", "Complete"),
            ("Phase 3", "Memory model (HLR)", "ready", "Can start"),
            ("Phase 4", "English DAG drawn", "done", "Graph drawn"),
        ]
        for i, (title, sub, kind, _status) in enumerate(mids):
            x = i * (box_w + gap)
            self._box(c, x, ymid, box_w, h_mid, title, sub, kind)
            self._arrow(c, x + box_w / 2, ymid, w / 2, y5 + h_late)

        self._box(c, 0, y5, w, h_late, "Phase 5  ·  Hybrid ranking", "Blocked until Phases 2, 3 and 4 each work on their own", "blocked")
        self._arrow(c, w / 2, y5, w / 2, y6 + h_late)
        self._box(c, 0, y6, w, h_late, "Phase 6  ·  Evaluation", "Blocked until the ranker in Phase 5 is wired up", "blocked")


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Arial-Bold", 8)
    canvas.drawString(MARGIN_L, PAGE_H - 7.6 * mm, "NeuroTrace-DAG")
    canvas.setFont("Arial", 8)
    canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 7.6 * mm, "Implementation pipeline")
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Arial", 8)
    canvas.drawString(MARGIN_L, 3.4 * mm, "Phase 1 complete  ·  Phase 2 complete  ·  Phases 3–6 pending")
    canvas.drawRightString(PAGE_W - MARGIN_R, 3.4 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    out = "/Users/oscar/Desktop/neurotrace-dag/NeuroTrace-DAG_Pipeline.pdf"
    doc = BaseDocTemplate(
        out,
        pagesize=A4,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
        title="NeuroTrace-DAG — Implementation Pipeline",
        author="NeuroTrace-DAG",
    )
    frame = Frame(MARGIN_L, MARGIN_B, CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B, id="main", showBoundary=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])

    story = []
    story.append(Spacer(1, 6))
    story.append(P("PROJECT PIPELINE", "cover_kicker"))
    story.append(P("NeuroTrace-DAG", "cover_title"))
    story.append(P(
        "Recommend the right learning resource, to the right student, at the right time — "
        "accounting for what they have probably forgotten and what they are ready to learn next.",
        "cover_sub",
    ))
    story.append(P("Status as of 26 September 2026. Working data, embeddings, and the prerequisite graph are in English.", "small"))
    story.append(Spacer(1, 6))

    story.append(section_title("1.  What this system is"))
    story.append(P(
        "NeuroTrace-DAG is a pipeline of four components, combined at the end into one ranked list. "
        "Only one of them is a model we train."
    ))
    story.append(make_table(
        ["Component", "Training?", "Role"],
        [
            ["Semantic embeddings", "No. A pretrained model is used as-is.", "Phase 2. How similar two texts are."],
            ["Memory retention (HLR)", "Yes. The one trained model.", "Phase 3. Probability the student still remembers."],
            ["Prerequisite DAG", "No. Built from confirmed edges.", "Phase 4. Whether the student is allowed to move on."],
            ["Ranking", "No in this version. Weights can be tuned later.", "Phase 5. One score, then the top of the list."],
        ],
        [CONTENT_W * 0.28, CONTENT_W * 0.34, CONTENT_W * 0.38],
    ))
    story.append(Spacer(1, 8))
    story.append(P(
        "Phases 2, 3 and 4 do not depend on each other. They can be built in parallel. "
        "Phase 5 waits until all three produce their outputs. Phase 6 waits on Phase 5."
    ))

    story.append(section_title("2.  The pipeline"))
    story.append(P(
        "Raw OULAD and MOOCCubeX enter Phase 1. Three clean tables come out. "
        "Each table feeds one middle phase. Those three phases meet in the ranker."
    ))
    story.append(Spacer(1, 4))
    story.append(PipelineDiagram(CONTENT_W))
    story.append(P(
        "Green is finished. Amber can start now. Grey is blocked on earlier phases. "
        "Teal boxes are the files that carry data from one phase to the next.",
        "caption",
    ))

    story.append(make_table(
        ["Phase", "Status", "Reads", "Writes"],
        [
            ["1. Data preprocessing", "Complete", "Raw OULAD + MOOCCubeX", "Three clean tables"],
            ["2. Semantic embeddings", "Complete", "concepts_clean.csv", "concept_embeddings.npy"],
            ["3. Memory model (HLR)", "Can start", "interactions_BBB.csv", "A function p(t)"],
            ["4. Prerequisite DAG", "Graph drawn", "prerequisite_edges.csv", "prerequisite_dag.pdf. Unlock rule still open."],
            ["5. Hybrid ranking", "Blocked", "Outputs of 2, 3 and 4", "Top-K recommendations"],
            ["6. Evaluation", "Blocked", "A working ranker", "Metrics and ablations"],
        ],
        [CONTENT_W * 0.28, CONTENT_W * 0.16, CONTENT_W * 0.28, CONTENT_W * 0.28],
    ))
    story.append(Spacer(1, 8))

    story.append(section_title("3.  Phase 1 — Data preprocessing"))
    story.append(P("Status: complete. This phase turns two differently structured datasets into tables the later phases can read.", "status"))
    story.append(Spacer(1, 4))
    story.append(P("3.1  OULAD", "h2"))
    story.append(P(
        "OULAD records who did what, and when, inside a real online course. "
        "Only module BBB was processed, from studentInfo, studentAssessment, studentVle and assessments. "
        "For every assessment attempt the script computed two fields:"
    ))
    story.extend(bullets([
        "<b>delta_t</b> — days since that student’s last click on the course site before the assessment. This is the time gap the memory model will use.",
        "<b>recall_label</b> — 1 if the score was at least 40, otherwise 0. This is the pass/fail target for Phase 3.",
    ]))
    story.append(P("Output: <font face='Arial-Bold'>interactions_BBB.csv</font>, 42,636 rows. Phase 2 does not read this file. Phase 3 does."))

    story.append(P("3.2  MOOCCubeX", "h2"))
    story.append(P(
        "MOOCCubeX records what a topic means and which topic must come first. "
        "concept.json (637,572 concepts) and cs.json (492,102 candidate pairs) were downloaded from the public links and parsed as JSON Lines, one object per line. "
        "Pairs were kept only when a human had marked them as a real prerequisite (ground_truth = 1). "
        "That leaves 725 edges and 287 concepts. Duplicate concept entries, one per academic domain, were resolved by preferring the Computer Science tag and the richest context text."
    ))
    story.append(P(
        "Each concept received a <font face='Arial-Bold'>text_for_embedding</font> field: the joined context when it exists, or the bare concept name when it does not. "
        "214 concepts have real context. 73 fall back to the name alone. "
        "The source text was Chinese. The working tables are now English: concept names are checked computer-science terms, and the descriptions were translated. "
        "A scan of concepts_clean.csv and prerequisite_edges.csv found no Chinese characters."
    ))

    story.append(P("3.3  Files Phase 1 handed on", "h2"))
    story.append(make_table(
        ["File", "Rows", "Used by"],
        [
            ["interactions_BBB.csv", "42,636", "Phase 3 only"],
            ["concepts_clean.csv", "287", "Phase 2. Columns: id, name, domain, context_len, text_for_embedding"],
            ["prerequisite_edges.csv", "725", "Phase 4. Columns: source_concept_name, target_concept_name"],
            ["concepts_clean_zh.csv", "287", "Chinese backup only. Do not feed it to a later phase."],
            ["prerequisite_edges_zh.csv", "725", "Chinese backup of the edges. Do not feed it to a later phase."],
        ],
        [CONTENT_W * 0.34, CONTENT_W * 0.16, CONTENT_W * 0.50],
    ))
    story.append(Spacer(1, 8))
    story.append(P(
        "The two datasets are not linked by an ID. interactions_BBB.csv describes one whole module. "
        "The 287 concepts are fine-grained topics, now named in English. "
        "Joining a student to those concepts is still a Phase 4 mapping, because one side is a whole module and the other is 287 separate topics."
    ))

    story.append(section_title("4.  Phase 2 — Semantic embeddings"))
    story.append(P("Status: complete. No training. The encoder is used exactly as published.", "status"))
    story.append(Spacer(1, 4))
    story.append(P(
        "Keyword match cannot tell that two different phrases are about the same idea. "
        "Phase 2 turns each concept’s text into a vector of 384 numbers so relevance can be measured by how close those vectors are. "
        "The input is the text_for_embedding column of concepts_clean.csv, all 287 rows."
    ))
    story.append(P(
        "The working concept text is English. The encoder is "
        "<font face='Arial-Bold'>paraphrase-multilingual-MiniLM-L12-v2</font>, used as published. "
        "It embeds this English text. The matrix was rebuilt after the tables were translated, so every vector matches the English row it sits on."
    ))
    story.append(P("What the script does:", "h2"))
    story.extend(bullets([
        "Load concepts_clean.csv and refuse to run if text_for_embedding has empty cells.",
        "Encode every text with the pretrained model. Vectors are L2-normalised, so cosine similarity is a dot product.",
        "Save one vector per concept in concept_embeddings.npy. Row i is the concept on row i of concepts_clean.csv. Shape (287, 384), float32.",
        "Check neighbours on 10 concepts: 5 with real context and 5 with the bare name only. Also compare all 725 prerequisite pairs with 725 random pairs.",
    ]))
    story.append(P(
        "At 287 concepts a full comparison is cheap. No approximate search index is used. "
        "The model keeps the first 128 tokens of each text, so a long encyclopedia snippet is only partly read. "
        "The concept name sits at the start of the text, so that part is kept."
    ))

    story.append(P("4.1  Sanity check", "h2"))
    story.append(P(
        "Prerequisite-linked pairs are closer than random pairs of the same count. "
        "Mean cosine is 0.292 for linked pairs and 0.194 for random pairs. "
        "The gap holds inside the context-rich group (0.291 versus 0.207) and inside the name-only group (0.343 versus 0.215). "
        "Name-only vectors sit closer together in general, so those 73 embeddings are less specific."
    ))
    story.append(make_table(
        ["Probe", "Context?", "Nearest concepts, in order"],
        [
            ["interrupt controller", "Yes", "interrupt signal, interrupt control, interrupt mechanism"],
            ["storage allocation", "Yes", "dynamic memory allocation, segmented memory management, auxiliary memory"],
            ["B-tree", "Yes", "leaf node (its only graph neighbour, rank 1)"],
            ["paging", "Yes", "logical address space, auxiliary memory, page fault"],
            ["interrupt service routine", "Name only", "interrupt enable, interrupt control, enable interrupts"],
            ["processor scheduling", "Name only", "multiprocessor scheduling (cosine 0.85)"],
            ["Internet Protocol", "Name only", "Routing Information Protocol, connection-oriented protocol"],
        ],
        [CONTENT_W * 0.30, CONTENT_W * 0.16, CONTENT_W * 0.54],
    ))
    story.append(P(
        "Full top-5 lists are in phase2_sanity_neighbors.csv. The numeric summary is in phase2_sanity_summary.txt.",
        "caption",
    ))
    story.append(P(
        "A graph neighbour and a semantic neighbour are related ideas, and they are not the same relation. "
        "Paging’s only confirmed prerequisite is interrupt mechanism, which did not appear in its top 5. "
        "The nearest vectors were still about memory: logical address space, auxiliary memory, and page fault. "
        "The embedding is doing the job Phase 2 asked of it."
    ))

    story.append(P("4.2  Files Phase 2 handed on", "h2"))
    story.append(make_table(
        ["File", "What it is"],
        [
            ["phase2_semantic_embeddings.py", "Encodes the 287 texts and writes the sanity check."],
            ["concept_embeddings.npy", "The matrix later phases load. Row order matches concepts_clean.csv."],
            ["phase2_sanity_neighbors.csv", "Top-5 neighbours for the 10 probe concepts."],
            ["phase2_sanity_summary.txt", "Pair scores, split by context, and the written limitations."],
        ],
        [CONTENT_W * 0.40, CONTENT_W * 0.60],
    ))
    story.append(Spacer(1, 6))
    story.append(P(
        "Phase 5 should load concept_embeddings.npy and compare a query vector with those rows by dot product. "
        "It should not re-encode the catalog unless the concept table changes."
    ))

    story.append(section_title("5.  Working language"))
    story.append(P(
        "The pipeline now runs in English. concepts_clean.csv, prerequisite_edges.csv, the embedding matrix, and the drawn graph all use English concept names."
    ))
    story.append(P(
        "MOOCCubeX arrived in Chinese. Concept names were set to checked computer-science terms, such as paging, virtual page, file allocation, and interrupt controller. "
        "Descriptions were translated into English, then scanned: the working tables contain no Chinese characters. "
        "concepts_clean_zh.csv and prerequisite_edges_zh.csv keep the Chinese originals as backups. Later phases do not read those backups. "
        "interactions_BBB.csv was already English."
    ))
    story.append(P(
        "Phase 2 was run again after the translation. Row i of concept_embeddings.npy is still row i of concepts_clean.csv, now from the English text."
    ))

    story.append(section_title("6.  Phase 3 — Memory retention"))
    story.append(P("Status: not started. This is the only phase that trains a model. It can start now. It does not wait on Phase 2 or Phase 4.", "status"))
    story.append(Spacer(1, 4))
    story.append(P(
        "Given a student and a topic they studied before, predict the probability they still remember it. "
        "The input is interactions_BBB.csv. The curve is the half-life form from Settles and Meeder (2016):"
    ))
    story.append(Spacer(1, 4))
    story.append(formula_box("p(t)  =  2^(−Δt / h)"))
    story.append(Spacer(1, 6))
    story.append(P(
        "p(t) is the probability of recall. Δt is delta_t, in days. "
        "h is the half-life: the number of days until recall probability falls to one half. The model learns h."
    ))
    story.append(Spacer(1, 2))
    story.append(formula_box("h  =  2^(w · x)"))
    story.append(Spacer(1, 6))
    story.append(P(
        "x is a feature vector built from columns the interactions file actually supports, such as delta_t, past attempts and past success. "
        "w is the weight vector. Training compares predicted p(t) with the observed recall label, using binary cross-entropy or the squared error from the original paper. "
        "The split is by student, about 80/20, so the test students were never seen in training. "
        "A first pass can use scikit-learn. A small PyTorch loop can match the paper’s formula more closely. Report MAE and AUC on the held-out students."
    ))
    story.append(P("Label imbalance, to decide before training", "h2"))
    story.append(P(
        "At a pass threshold of 40, recall_label is about 96.7% pass and 3.3% fail. "
        "A model that always says pass would look accurate and would have learned nothing. AUC is unstable with so few failures. "
        "The Phase 3 owner picks one of these and writes the choice down:"
    ))
    story.extend(bullets([
        "Raise the pass threshold and state the new value.",
        "Use the raw score, scaled to the range 0–1, as a continuous target. This matches a probability and avoids the split into pass and fail.",
        "Keep the binary label, weight the classes, and report AUC and precision-recall AUC rather than accuracy.",
    ]))
    story.append(P("Output: a function that takes a student, a concept and the current time, and returns p(t)."))

    story.append(section_title("7.  Phase 4 — Prerequisite graph"))
    story.append(P("Status: the English graph is drawn and it has no cycles. The unlock rule and the OULAD mapping are still open.", "status"))
    story.append(Spacer(1, 4))
    story.extend(bullets([
        "The directed graph is already built from the English edges. An edge runs from source_concept_name to target_concept_name: learn the source before the target. The drawing is prerequisite_dag.pdf (9 pages) and prerequisite_dag.png.",
        "The graph has no cycles: 287 concepts, 725 edges, longest path 26 edges.",
        "A concept is unlocked when every predecessor has mastery above a threshold (a working value is 0.70, taken from the Phase 3 recall probability).",
        "Expose is_unlocked(student, concept), returning true or false.",
    ]))
    story.append(P("Linking OULAD to MOOCCubeX", "h2"))
    story.append(P(
        "Student mastery comes from OULAD. The graph comes from MOOCCubeX. "
        "A full automatic join is not realistic: there are no shared IDs, and one side is a whole module while the other is 287 fine-grained concepts. Both sides are now in English. "
        "The plan is a small, manually curated mapping from module BBB’s topics to a subset of those 287 concepts. "
        "Each row of that table should say how the mapping was chosen. Reviewers will ask. "
        "The graph will constrain only the mapped subset. Many concepts have few or no prerequisites, because 725 edges on 287 concepts is a benchmark subset, so those concepts will look unlocked immediately."
    ))

    story.append(section_title("8.  Phase 5 — Hybrid ranking"))
    story.append(P("Status: blocked. Start only after Phases 2, 3 and 4 each work on their own.", "status"))
    story.append(Spacer(1, 4))
    story.append(P("For one student and one query, produce a ranked list."))
    story.extend(bullets([
        "Candidates: concepts whose vectors are close to the query (Phase 2).",
        "Filter: drop any candidate that is not unlocked, meaning a prerequisite is still below the mastery threshold (Phase 4).",
        "Score the rest and return the top of the list.",
    ]))
    story.append(Spacer(1, 2))
    story.append(formula_box(
        "Score(r)  =  w1 · SemanticRelevance(query, r)  +  w2 · (1 − p(t))  +  w3 · CurriculumProgress(r)"
    ))
    story.append(Spacer(1, 6))
    story.append(P(
        "SemanticRelevance is the cosine from Phase 2. "
        "(1 − p(t)) is larger when the memory model thinks the concept has been forgotten. "
        "CurriculumProgress is a measure of how far the concept sits along the graph relative to what the student has already mastered, for example normalised depth from the student’s frontier. "
        "Start the three weights equal, at one third each. Hand-set weights are the weakest part of this design. "
        "Once the pipeline runs, search them against a validation metric, or record them as a limitation."
    ))

    story.append(section_title("9.  Phase 6 — Evaluation"))
    story.append(P("Status: blocked. It measures a pipeline that already runs.", "status"))
    story.append(Spacer(1, 4))
    story.append(make_table(
        ["Metric", "What it measures"],
        [
            ["MAE", "How far HLR’s predicted recall is from the held-out outcome."],
            ["AUC", "How well HLR separates “will recall” from “will not”. Read it next to the class balance from Phase 3."],
            ["NDCG@K", "Whether the ranked list puts relevant concepts first. This needs relevance labels, which the datasets do not provide directly."],
            ["Prerequisite violation rate", "The share of recommendations whose prerequisites were not unlocked."],
            ["Inference latency", "End-to-end time for one recommendation, averaged over many queries."],
        ],
        [CONTENT_W * 0.32, CONTENT_W * 0.68],
    ))
    story.append(Spacer(1, 8))
    story.append(P("Ablations, so each component has to earn its place:", "h2"))
    story.extend(bullets([
        "Semantic only: rank by cosine, with no memory term and no graph filter.",
        "No DAG: the full score, with the prerequisite filter removed.",
        "No memory: the full pipeline with the forgetting weight set to zero.",
    ]))
    story.append(P(
        "State the scope in the write-up: results cover OULAD module BBB only, the graph applies only to the manually mapped concepts, "
        "and NDCG needs labels that must be constructed."
    ))

    story.append(section_title("10.  Build order"))
    story.extend(bullets([
        "Phase 1 is done. The three tables are the contract for everything after it.",
        "Phase 2 is done. concept_embeddings.npy is the contract for semantic relevance.",
        "Phase 3 can start from interactions_BBB.csv. The English DAG is already drawn; Phase 4 still needs is_unlocked() and the manual mapping from module BBB.",
        "Wire Phases 2, 3 and 4 into the Phase 5 score.",
        "Run Phase 6, including the three ablations.",
    ]))

    story.append(section_title("11.  Limitations to carry into the report"))
    story.extend(bullets([
        "The working concept text is English. Names are checked terms. Long descriptions are translations of noisy encyclopedia snippets, so a sentence can still be awkward even though no Chinese remains.",
        "73 of 287 concepts have no context text. Their embeddings are weaker and less specific. The Phase 2 numbers above are split so this can be seen.",
        "The model reads 128 tokens. Long context is only partly used. Neighbours were still on topic.",
        "The prerequisite graph is a sparse benchmark subset: 725 edges on 287 concepts, not a full curriculum.",
        "OULAD and the concept graph still meet only through a manual mapping, which Phase 4 still has to write. Language is no longer the obstacle.",
        "recall_label is about 96.7% pass. How to handle that is a Phase 3 decision, still open.",
        "Only OULAD module BBB is used.",
        "Ranking weights are hand-set until Phase 5 has a validation metric to tune them against.",
    ]))

    story.append(section_title("12.  Tools"))
    story.append(make_table(
        ["Step", "Library"],
        [
            ["Preprocessing", "pandas, numpy"],
            ["Embeddings", "sentence-transformers, model paraphrase-multilingual-MiniLM-L12-v2"],
            ["Similarity", "numpy dot product on normalised vectors"],
            ["HLR training", "scikit-learn for a first pass; PyTorch if the paper’s formula is coded directly"],
            ["Graph", "networkx and Graphviz. English drawing: prerequisite_dag.pdf"],
            ["Evaluation", "scikit-learn.metrics"],
            ["Hardware", "CPU. No GPU is required at this scale."],
        ],
        [CONTENT_W * 0.28, CONTENT_W * 0.72],
    ))
    story.append(Spacer(1, 10))
    story.append(P(
        "Scripts in the project: phase1_oulad_preprocessing.py, phase1_mooccubex_preprocessing.py, phase2_semantic_embeddings.py. "
        "Re-running Phase 2 rewrites concept_embeddings.npy and the two sanity files. It does not change the Phase 1 tables.",
        "small",
    ))

    doc.build(story)
    print(out)


if __name__ == "__main__":
    build()
