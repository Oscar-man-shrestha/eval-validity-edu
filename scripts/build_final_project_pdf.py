"""Build the shareable NeuroTrace-DAG report from audited experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs_junyi" / "phases" / "research_integrity_audit.json"
OUT = ROOT / "output" / "pdf" / "NeuroTrace-DAG_Research_Integrity_Report.pdf"

INK = HexColor("#13233B")
BLUE = HexColor("#246BCE")
TEAL = HexColor("#137A74")
GOLD = HexColor("#B7791F")
PALE = HexColor("#EEF4FB")
LINE = HexColor("#CAD6E5")
MUTED = HexColor("#536579")


def f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def ci(metric: dict, digits: int = 4) -> str:
    low, high = metric["ci"]
    return f"[{f(low, digits)}, {f(high, digits)}]"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.8 * cm, "NeuroTrace-DAG | Research Integrity Report | Generated from audited experiment JSON")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.8 * cm, f"Page {doc.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=INK, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=base["BodyText"], fontSize=12, leading=17, textColor=MUTED, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK, spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=14, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=MUTED),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontSize=10, leading=14, textColor=INK, leftIndent=10, rightIndent=10, spaceBefore=5, spaceAfter=5),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontSize=8.5, leading=10.5, textColor=INK),
    }


def table(data, widths, s):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 10.5),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]))
    return t


def head(text: str, s: dict) -> Paragraph:
    return Paragraph(f'<font color="#FFFFFF"><b>{text}</b></font>', s["table"])


def build() -> None:
    audit = json.loads(AUDIT.read_text())
    findings = audit["validated_findings"]
    repeat = findings["cross_dataset_concept_repetition"]["rows"]
    rec = findings["ragr_vs_recent5"]
    ready = findings["readiness_correctness"]
    forget = findings["forgetting_gap_ge_1_day"]
    s = styles()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=1.65 * cm, rightMargin=1.65 * cm, topMargin=1.55 * cm, bottomMargin=1.75 * cm)
    doc.addPageTemplates(PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")], onPage=footer))
    story = []

    story += [
        Spacer(1, 1.3 * cm),
        Paragraph("NeuroTrace-DAG", s["title"]),
        Paragraph("When next-item metrics mislead: a multi-dataset evaluation study in educational logs", s["subtitle"]),
        Table([[Paragraph("FINAL PROJECT SCOPE", s["small"]), Paragraph("Multi-dataset evaluation-validity and protocol study", s["body"])]], colWidths=[4.1 * cm, 12.5 * cm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), TEAL), ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
            ("BACKGROUND", (1, 0), (1, 0), PALE), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ])),
        Spacer(1, 0.55 * cm),
        Paragraph("Executive summary", s["h1"]),
        Paragraph("This project audits whether next-item recommendation metrics in educational logs reflect educational progression or simply the repeat behavior generated by a platform. It introduces a reproducible protocol that separates continue, revisit, and advance actions; compares simple baselines; uses leakage controls and sequence-level confidence intervals; and keeps negligible and unresolved findings in the final report.", s["body"]),
        Spacer(1, 0.25 * cm),
    ]

    headline = [
        [head("What we found", s), head("Evidence-based conclusion", s)],
        [Paragraph("Action structure", s["table"]), Paragraph("Concept-level repetition differs strongly across datasets, so aggregate item metrics are not directly comparable.", s["table"])],
        [Paragraph("Prototype ranking", s["table"]), Paragraph(f"RAGR - Recent-5 Recall@5 = {f(rec['paired_delta_r5']['mean'], 4)} {ci(rec['paired_delta_r5'])}; below the practical threshold {f(rec['practical_threshold_r5'], 2)}.", s["table"])],
        [Paragraph("DAG readiness", s["table"]), Paragraph(f"Delta AUC = {f(ready['primary']['mean'], 4)} {ci(ready['primary'])}; status: {ready['status']}.", s["table"])],
        [Paragraph("Forgetting", s["table"]), Paragraph(f"Gap>=1-day delta AUC = {f(forget['delta_log_gap_minus_success']['mean'], 4)} {ci(forget['delta_log_gap_minus_success'])}; status: {forget['status']}.", s["table"])],
    ]
    story += [table(headline, [4.3 * cm, 12.3 * cm], s), Spacer(1, 0.35 * cm), Paragraph("The contribution is the audit protocol, not a claim that the prototype is a superior teaching recommender.", s["callout"]), PageBreak()]

    story += [Paragraph("Dataset and implementation flow", s["h1"]), Paragraph("The study uses three public educational logs after aligning the comparison to a concept unit: Junyi topics, ASSISTments skills, and XES3G5M knowledge components (KCs). This prevents raw exercise or question identifiers from creating a misleading cross-dataset comparison.", s["body"])]
    concept_data = [[head("Dataset / unit", s), head("Transitions", s), head("Continue", s), head("Revisit", s), head("Advance", s)]]
    for row in repeat:
        mix = row["three_way_mix"]
        concept_data.append([
            Paragraph(row["dataset_unit"], s["table"]), Paragraph(str(mix["n"]), s["table"]),
            Paragraph(f"{mix['continue']['frac']:.1%}", s["table"]), Paragraph(f"{mix['revisit']['frac']:.1%}", s["table"]), Paragraph(f"{mix['advance']['frac']:.1%}", s["table"]),
        ])
    story += [table(concept_data, [5.8 * cm, 3.1 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm], s), Spacer(1, 0.4 * cm)]
    story += [
        Paragraph("Actual data flow", s["h2"]),
        Paragraph("Public log and expert graph -> sequence-level split -> action relabelling (continue / revisit / advance) -> baseline and prototype ranking -> sequence-level paired bootstrap -> practical-threshold interpretation -> final audit JSON -> this report.", s["callout"]),
        Paragraph("Leakage safeguards", s["h2"]),
        Paragraph("The evaluation keeps a fixed sequence sample, records duplicate-sequence checks, computes confidence intervals by resampling sequences rather than individual rows, and distinguishes what can and cannot be tested when learner identifiers are unavailable.", s["body"]),
        Paragraph("How the DAG is used", s["h2"]),
        Paragraph("The directed acyclic graph encodes declared prerequisite relations. It is evaluated as a readiness feature for correctness prediction and visualized as a learning-map structure. Observational correlation is not presented as proof that forcing the graph improves learning outcomes.", s["body"]),
        PageBreak(),
    ]

    story += [Paragraph("Results and interpretation", s["h1"])]
    results = [
        [head("Analysis", s), head("Measured output", s), head("Interpretation", s)],
        [Paragraph("Ranking", s["table"]), Paragraph(f"RAGR R@5 {f(rec['ragr_r5'])}; Recent-5 {f(rec['recent5_r5'])}", s["table"]), Paragraph("The difference is statistically detectable but below the predeclared practical bar.", s["table"])],
        [Paragraph("Readiness", s["table"]), Paragraph(f"Delta AUC {f(ready['primary']['mean'], 4)}; practical bar {f(ready['practical_threshold_delta_auc'], 3)}", s["table"]), Paragraph("Detectable but negligible; do not make a causal pedagogy claim.", s["table"])],
        [Paragraph("Forgetting", s["table"]), Paragraph(f"Success-only AUC {f(forget['auc']['success_only'])}; +log gap AUC {f(forget['auc']['success_plus_log1p_gap'])}", s["table"]), Paragraph("Gap information has unresolved value in the tested data.", s["table"])],
    ]
    story += [table(results, [3.2 * cm, 6.2 * cm, 7.0 * cm], s), Spacer(1, 0.4 * cm)]
    story += [
        Paragraph("What the live interface shows", s["h2"]),
        Paragraph("The interface is an exploratory prototype viewer. A user enters a short practice history, the system returns a scored continue/revisit or advance action type, ranks candidate concepts, and shows nearby prerequisite relationships. It is intentionally labelled as model inspection rather than a validated lesson prescription.", s["body"]),
        Paragraph("Accurate method labels", s["h2"]),
        Paragraph("The local neural rank probes are not faithful reproductions of published GRU4Rec or SASRec. They are retained only as GRU-sequence and attention-sequence exploratory probes. This avoids overstating a small two-model experiment as a benchmark comparison.", s["body"]),
        Paragraph("Viva-ready explanation", s["h2"]),
        Paragraph("We found that next-item accuracy can reward a platform's repetition structure rather than meaningful educational progression. Our contribution is a reproducible evaluation protocol that makes this visible across topic, skill, and KC units, compares simple recency baselines, and reports practically negligible or unresolved results honestly.", s["callout"]),
        PageBreak(),
    ]

    refs = audit["references"]
    story += [Paragraph("Research integrity and future work", s["h1"]), Paragraph("Claims deliberately excluded", s["h2"])]
    for claim in audit["positioning"]["prohibited_method_claims"]:
        story.append(Paragraph(f"- {claim}", s["body"]))
    story += [Paragraph("What would strengthen this into a teaching-intervention study", s["h2"]), Paragraph("A future study should preregister a learner-level intervention, compare the recommendation policy against robust baselines, and measure downstream learning gain rather than only the next logged click. That evidence is required before claiming the system improves student learning.", s["body"]), Paragraph("Key references", s["h2"])]
    for ref in refs:
        story.append(Paragraph(f"<b>{ref['citation']}</b><br/>{ref['use']}", s["body"]))
    story += [Spacer(1, 0.5 * cm), Paragraph("Reproducibility", s["h2"]), Paragraph("Source of truth: outputs_junyi/phases/research_integrity_audit.json. Regenerate with scripts/build_research_integrity_audit.py followed by scripts/render_research_integrity_docs.py and this PDF builder.", s["small"])]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
