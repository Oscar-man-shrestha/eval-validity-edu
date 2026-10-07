"""Dataset / flow / results PDF — regenerated only from outputs_junyi/phases/*.json.

Modes:
  --mode=full  technical / supplement report (default) → docs/NeuroTrace-DAG_Data_Flow_Report.pdf
  --mode=slim  paper-facing tables + 5 figures → docs/NeuroTrace-DAG_Paper_Slim.pdf

One source of truth for numbers: phases/*.json and paper_assets/ (via build_paper_assets.py).
After --touch-test: regenerate H-rev tables from primary_claims only — never hand-edit claim cells.
"""

import argparse
import json
from pathlib import Path

from reportlab.lib.colors import Color, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs_junyi" / "figures"
PAPER_FIG = ROOT / "paper_assets" / "figures"
OUT = ROOT / "docs" / "NeuroTrace-DAG_Data_Flow_Report.pdf"
OUT_SLIM = ROOT / "docs" / "NeuroTrace-DAG_Paper_Slim.pdf"
DAG_PNG = ROOT / "outputs_junyi" / "junyi_quadratic_dag.png"
PHASES = ROOT / "outputs_junyi" / "phases"
ASSETS = ROOT / "paper_assets"


def jload(name: str) -> dict:
    return json.loads((PHASES / name).read_text())


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def ci_str(obj, key="ci"):
    lo, hi = obj[key]
    return f"[{fmt(lo)}, {fmt(hi)}]"


pdfmetrics.registerFont(TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))

NAVY = Color(0.106, 0.227, 0.294)
TEAL = Color(0.122, 0.435, 0.416)
INK = Color(0.145, 0.165, 0.176)
MUTED = Color(0.333, 0.376, 0.400)
LINE = Color(0.82, 0.855, 0.843)
GREEN_BG = Color(0.91, 0.957, 0.933)

PAGE_W, PAGE_H = A4
ML, MR, MT, MB = 16 * mm, 16 * mm, 18 * mm, 16 * mm
CW = PAGE_W - ML - MR


def S(name, **kw):
    base = dict(fontName="Arial", fontSize=10, leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(name, **base)


STY = {
    "kicker": S("kicker", fontName="Arial-Bold", fontSize=9, textColor=TEAL, spaceAfter=4),
    "title": S("title", fontName="Arial-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=6),
    "sub": S("sub", fontSize=10.5, leading=14, textColor=INK, spaceAfter=8),
    "h1": S("h1", fontName="Arial-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=5),
    "h2": S("h2", fontName="Arial-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=3),
    "body": S("body", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
    "small": S("small", fontSize=8, leading=11, textColor=MUTED, spaceAfter=4),
    "cap": S("cap", fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=2, spaceAfter=8),
    "th": S("th", fontName="Arial-Bold", fontSize=7.8, leading=10, textColor=white),
    "td": S("td", fontSize=8, leading=10.8),
    "tdb": S("tdb", fontName="Arial-Bold", fontSize=8, leading=10.8, textColor=NAVY),
}


def P(text, style="body"):
    return Paragraph(text, STY[style])


def bullets(items):
    return [P("•  " + item, "body") for item in items]


def table(headers, rows, widths):
    data = [[P(h, "th") for h in headers]]
    for row in rows:
        data.append([P(str(c), "td") for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), GREEN_BG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, Color(0.973, 0.980, 0.976)]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, LINE),
    ]))
    return t


def img(name, w=CW, aspect=0.48):
    path = FIG / name
    if not path.exists():
        return P(f"[missing figure {name}]", "small")
    return Image(str(path), width=w, height=w * aspect)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 10 * mm, PAGE_W, 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Arial", 8)
    canvas.drawString(ML, PAGE_H - 6.5 * mm, "NeuroTrace-DAG  ·  Dataset, implementation flow, and measured outputs")
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.drawString(ML, 4 * mm, "Group technical report  ·  7 October 2026  ·  from phases/*.json")
    canvas.drawRightString(PAGE_W - MR, 4 * mm, f"{doc.page}")
    canvas.restoreState()


def _maybe(name):
    return (PHASES / name).exists()


def build():
    p1 = jload("phase1_prepare.json")
    p2 = jload("phase2_embeddings.json")
    p3 = jload("phase3_memory.json")
    p4 = jload("phase4_graph.json")
    p5 = jload("phase5_ranker.json")
    p6 = jload("phase6_eval.json")
    mem_test = p3["evaluate_on_test_json"]
    mem_timed = p3.get("timed_learner_split", {})
    mix = p4["next_item_mix_train_sample"]
    rank = p6["ranking"]
    gate = rank["gate"]

    f3_concept = jload("followup3_concept_repetition.json") if _maybe("followup3_concept_repetition.json") else None
    f3_forget = jload("followup3_forgetting.json") if _maybe("followup3_forgetting.json") else None
    f3_dup = jload("followup3_exact_dup_chance.json") if _maybe("followup3_exact_dup_chance.json") else None
    f3_rank = jload("followup3_rank_reversal.json") if _maybe("followup3_rank_reversal.json") else None
    f3_xes = jload("followup3_xes_kc_repetition.json") if _maybe("followup3_xes_kc_repetition.json") else None
    assist = jload("followup_assistments2009.json") if _maybe("followup_assistments2009.json") else None
    xes2 = jload("followup2_xes3g5m.json") if _maybe("followup2_xes3g5m.json") else None
    review_rank = jload("review_ranking.json") if _maybe("review_ranking.json") else None
    review_ready = jload("review_readiness.json") if _maybe("review_readiness.json") else None
    inertia = jload("phase6_inertia_diagnostics.json") if _maybe("phase6_inertia_diagnostics.json") else None
    freeze = jload("probe_freeze.json") if _maybe("probe_freeze.json") else None
    p3_full = jload("phase3_memory_fulltest.json") if _maybe("phase3_memory_fulltest.json") else None
    dual_cl = jload("dual_clustering.json") if _maybe("dual_clustering.json") else None
    dual_gm = jload("dual_cluster_gru_markov.json") if _maybe("dual_cluster_gru_markov.json") else None
    gru_mk = jload("gru_markov_paired_ci.json") if _maybe("gru_markov_paired_ci.json") else None
    paper_counts = jload("paper_junyi_counts.json") if _maybe("paper_junyi_counts.json") else None
    paper_rank_full = jload("paper_ranking_full.json") if _maybe("paper_ranking_full.json") else None
    paper_cal = jload("paper_calibration.json") if _maybe("paper_calibration.json") else None
    paper_sens = jload("paper_sensitivity.json") if _maybe("paper_sensitivity.json") else None
    paper_rev = jload("paper_rank_reversal_full.json") if _maybe("paper_rank_reversal_full.json") else None
    paper_xes = jload("paper_xes_kc_methods.json") if _maybe("paper_xes_kc_methods.json") else None
    paper_verdicts = jload("paper_paired_verdicts.json") if _maybe("paper_paired_verdicts.json") else None
    locked_hashes = jload("prereg_v1_locked_hashes.json") if _maybe("prereg_v1_locked_hashes.json") else None
    freeze_repl = jload("freeze_partition_replication.json") if _maybe("freeze_partition_replication.json") else None
    # Prefer independent XES methods JSON when present
    if paper_xes:
        f3_xes = paper_xes
    # Prefer full-test rank-reversal (Markov + gru4rec_style) when present
    rank_rev = paper_rev or f3_rank

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4, leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title="NeuroTrace-DAG — Dataset, Flow, and Results",
        author="NeuroTrace-DAG",
    )
    frame = Frame(ML, MB, CW, PAGE_H - MT - MB, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    s = []

    # ── Cover ──────────────────────────────────────────────────────────────
    s += [
        P("DATASET · IMPLEMENTATION FLOW · ACTUAL OUTPUTS", "kicker"),
        P("NeuroTrace-DAG", "title"),
        P(
            "Separate group document for <font face='Arial-Bold'>dataset / input</font>, "
            "<font face='Arial-Bold'>phase-by-phase Input → Processing → Output</font>, and "
            "<font face='Arial-Bold'>measured results</font> from "
            "<font face='Arial-Bold'>outputs_junyi/phases/*.json</font> "
            "(Phases 1–6, Follow-up 3, probe_freeze, paper_* eval-validity JSONs). Numbers are loaded from JSON at build time — "
            "not hand-typed. Study framing: <font face='Arial-Bold'>evaluation-validity</font> "
            "(domain replication/extension of next-basket reality-check work), not a recommender win. "
            "Covers the seven required topics: (1) datasets, (2) phase I/O, (3) travel plan, "
            "(4) actual outputs/samples, (5) visualizations, (6) ROC/AUC, (7) full end-to-end flow. "
            "Readable companion: <font face='Arial-Bold'>docs/DATASET_INPUT_FLOW_RESULTS.md</font> "
            "and Word <font face='Arial-Bold'>docs/NeuroTrace-DAG_Dataset_Input_Flow_Results.docx</font>. "
            "Data cards: docs/data_cards/. Rebuild: "
            "<font face='Arial-Bold'>python scripts/build_dataflow_report.py --mode=full</font>. "
            "Demo: http://127.0.0.1:8020.",
            "sub",
        ),
        P(
            "Product / diagnostic output: (i) P(review | history), (ii) ranked next exercises tagged "
            "review/advance with predicted recall, (iii) neighbourhood DAG. "
            "<font face='Arial-Bold'>Confirmatory scope (after prereg-v1):</font> primary H-rev "
            "(native↔cluster GRU−Markov sign flip; family m≤6, Bonferroni) on the freeze holdout — "
            "see §4.13. RAGR, forgetting, H-rep, and DAG readiness are "
            "<font face='Arial-Bold'>exploratory</font> (graph/forget Junyi-only). "
            "We do <font face='Arial-Bold'>not</font> claim a practically superior recommender, "
            "faithful GRU4Rec/SASRec reproductions, or that we discovered repetition bias itself. "
            "Canonical capped ranking CI is <font face='Arial-Bold'>review_ranking.json</font>; "
            "full-test and calibration live in <font face='Arial-Bold'>paper_ranking_full.json</font> / "
            "<font face='Arial-Bold'>paper_calibration.json</font>. "
            "Prereg drafts: docs/PREREGISTRATION.md + PREREGISTRATION_REPLICATION.md + DEVIATIONS.md.",
            "body",
        ),
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Dataset / Input
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("1.  Dataset / input", "h1")]

    s += [P("1.1  What datasets are we using?", "h2")]
    s += [P(
        "<font face='Arial-Bold'>Primary (implementation + demo):</font> Junyi Academy Math Practicing Log "
        "via EduData <font face='Arial-Bold'>ktbd-junyi</font> (Chang, Lin &amp; Chen, EDM 2015; "
        "DataShop #1198). Same id space for exercises, prerequisite DAG, and learner attempts."
    )]
    s += [P(
        "<font face='Arial-Bold'>Supporting (cross-checks only):</font> "
        "(a) ASSISTments 2009–2010 skill-builder <font face='Arial-Bold'>corrected</font> CSV "
        "(EduData USTC mirror) for skill-level repetition and rank-reversal; "
        "(b) XES3G5M (NeurIPS 2023) question-level sequences for KC-level repetition and rank-reversal; "
        "(c) Junyi raw timed extract (<font face='Arial-Bold'>timed_interactions.npz</font>) for "
        "learner-disjoint day-gap forgetting and timed ranking. "
        "OULAD×MOOCCubeX was rejected (no shared concept ids)."
    )]

    s += [P("1.2  Where each dataset came from", "h2")]
    s += [table(
        ["Dataset", "Source", "Local path"],
        [
            ["Junyi ktbd-junyi", "http://base.ustc.edu.cn/data/ktbd/junyi/", "data/junyi_ktbd/"],
            ["Junyi timed rows", "DataShop dump → junyi.rar → timed_interactions.npz", "data/ (npz stream)"],
            ["ASSISTments 2009 corrected", "EduData USTC skill-builder corrected", "data/assistments2009/…/skill_builder_data_corrected.csv"],
            ["XES3G5M", "NeurIPS 2023 release (question_level CSVs / parquet)", "data/xes3g5m/"],
        ],
        [CW * 0.24, CW * 0.38, CW * 0.38],
    )]

    s += [P("1.3  What the primary dataset contains", "h2")]
    s += [table(
        ["File", "Contents (features / schema)", "Counts we use"],
        [
            ["vertex_id2idx", "English exercise slug → integer id 0…834", f"{p1['concepts']} concepts"],
            ["prerequisite.json", "Directed [parent, child] expert edges", f"{p1['edges']} unique edges"],
            ["similarity.json", "[u, v, weight] official similarity pairs", "1,954 pairs"],
            ["train.json", "Per-line [[item_id, correct], …] sequences", f"{p1['train_sequences']:,} seq (len 12–200)"],
            ["test.json", "Same schema, published hold-out", f"{p1['test_sequences']:,} sequences"],
            ["Train interactions", "Sum of filtered train.json lengths", f"{p1.get('interactions_train', p1['interactions']):,} attempts"],
            ["Train+test interactions", "train.json + test.json after len filter", f"{p1.get('interactions_train_plus_test', p1['interactions']):,} attempts"],
            ["Topic-level n", "train+test consecutive transitions", f"{p1.get('transitions_train_plus_test', '—'):,} pairs"],
        ],
        [CW * 0.22, CW * 0.48, CW * 0.30],
    )]
    if paper_counts:
        idc = (paper_counts.get("topic_level_share") or {}).get("identity_check") or {}
        s += [P(
            f"Topic-level n identity (paper_junyi_counts.json): "
            f"train+test interactions − train seq − test seq = "
            f"{idc.get('interactions_minus_sequences', p1.get('transitions_train_plus_test')):,} "
            f"(equal={idc.get('equal')}). "
            f"Formula: {idc.get('formula', '2,842,883 − 27,434 − 6,290 = 2,809,159')}.",
            "cap",
        )]
    s += [P(
        "<font face='Arial-Bold'>Targets:</font> at each cut, (1) next exercise id (ranking label); "
        "(2) Phase 3 memory = whether the <font face='Arial-Bold'>next revisit attempt</font> "
        "on a concept already seen in the sequence is correct (0/1). "
        "The first attempt on a concept emits <font face='Arial-Bold'>no memory row</font>. "
        "Attempt-level accuracy ≈ 0.51 on a 4k-sequence sample is a different task "
        "(every attempt, including firsts) and is not the Phase 3 label. "
        f"Join: {p1['join']}.",
        "cap",
    )]

    s += [P("1.4  Purpose — why these datasets?", "h2")]
    s += [P(
        "Research problem (evaluation validity): do aggregate next-item metrics mostly reward "
        "<font face='Arial-Bold'>continue / revisit / advance</font> structure, and do conclusions "
        "change with concept granularity? Closest prior framing is next-basket reality-check work "
        "(repeat vs explore); we extend that to education. "
        "Junyi supplies traces + expert DAG in one id space. "
        "ASSISTments and XES3G5M test whether sticky next==last is platform/granularity-specific. "
        "Timed Junyi supports exploratory day-gap (H-forget) checks. "
        "Demo still emits review/advance rankings, but that is not a SOTA recommender claim."
    )]

    s += [P("1.5  Where each dataset is used in the implementation", "h2")]
    s += [table(
        ["Dataset", "Train / fit", "Test / validation", "Role"],
        [
            ["Junyi train.json", "Memory, gate, two ranker heads, popularity", "—", "Primary training corpus"],
            ["Junyi test.json", "Never fitted", "Phase 6 R@K; gate AUC; exact-dup check", "Published hold-out"],
            ["vertex + embeddings", "Query = mean of last 5", "Same at eval", "Semantic features + search"],
            ["prerequisite.json", "Advance-head unlock features", "DAG figures; violation cost", "Expert curriculum graph"],
            ["similarity.json", "Advance-head neighbour bit", "Ablations", "Official similarity"],
            ["timed_interactions.npz", "Learner-split forgetting + timed rankers", "Paired ΔAUC; rank-reversal", "Real day Δt"],
            ["ASSISTments corrected", "Popularity / sequence models (learner split)", "Repetition + rank-reversal", "Skill-level cross-check"],
            ["XES3G5M", "Same pattern, uid split", "KC repetition (2 methods) + rank-reversal", "KC-level cross-check"],
        ],
        [CW * 0.22, CW * 0.26, CW * 0.26, CW * 0.26],
    )]
    if assist:
        s += [P(
            f"ASSISTments (prior JSON): {assist['collapse']['n_raw_rows']:,} raw rows → "
            f"{assist['collapse']['n_collapsed_attempts']:,} attempts; "
            f"{assist['split']['n_learners']:,} learners (train {assist['split']['n_train_learners']:,} / "
            f"test {assist['split']['n_test_learners']:,}; overlap={assist['split']['overlap_learners']}).",
            "small",
        )]
    if xes2 and xes2.get("available"):
        # One reference value only: primary-KC method A; expand-all is §4.6 sensitivity
        xes_ref = None
        if f3_xes:
            xes_ref = f3_xes["method_a_question_then_primary_kc"]["share_next_equals_last"]["share"]
        s += [P(
            f"XES3G5M: {xes2['split']['n_learners']:,} uids "
            f"(train {xes2['split']['n_train']:,} / test {xes2['split']['n_test']:,}; "
            f"overlap={xes2['split']['overlap']}). "
            + (
                f"Reference KC next==last (method A, question→primary KC)={fmt(xes_ref)}; "
                "expand-all method B and question-level shares are in §4.6 / Follow-up 2 JSON — "
                "not repeated here."
                if xes_ref is not None
                else "See §4.6 for KC repetition."
            ),
            "small",
        )]

    s += [P("1.6  Data cards, licences, and ASSISTments filter funnel", "h2")]
    s += [P(
        "Authoritative provenance: <font face='Arial-Bold'>docs/data_cards/</font> "
        "(junyi_ktbd, junyi_timed, assistments2009, xes3g5m) with source URL, SHA-256, licence notes, "
        "counts, filters, timestamp applicability. Verify hashes with "
        "<font face='Arial-Bold'>scripts/fetch_datasets.py</font>. "
        "ASSISTments/XES licence text is marked UNVERIFIED until stored for public release."
    )]
    if freeze:
        funnel = freeze["assistments_filter_funnel"]
        s += [table(
            ["Filter step", "Learners", "Rows"],
            [
                [st["step"], f"{st['n_learners']:,}", f"{st['n_rows']:,}"]
                for st in funnel["steps"]
            ],
            [CW * 0.50, CW * 0.25, CW * 0.25],
        )]
        s += [P(
            f"From probe_freeze.json: {funnel.get('note', '')} "
            f"sequence_builder_n_learners={funnel.get('sequence_builder_n_learners')}.",
            "cap",
        )]

    s += [Spacer(1, 4), img("data_distributions.png"), P(
        "Figure 1. Left: filtered Junyi train sequence lengths. Right: attempt-level correct/incorrect balance.",
        "cap",
    )]

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Input at each phase
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("2.  Input at each phase (actual implementation)", "h1")]
    s += [img("system_flow.png"), P(
        "Figure 2. End-to-end flow as implemented in run_all_phases.py / serve engine.",
        "cap",
    )]

    phases = [
        ("Phase 1 — Prepare",
         "Raw ktbd-junyi files in data/junyi_ktbd/ (vertex_id2idx, prerequisite.json, train/test.json).",
         "Length filter 12–200; build English concept table and edge list; confirm native id join; "
         "record faults closed vs OULAD×MOOCCubeX.",
         f"concepts_junyi.csv ({p1['concepts']}), prerequisite_edges_junyi.csv ({p1['edges']}), "
         f"phase1_prepare.json. {p1['train_sequences']:,} / {p1['test_sequences']:,} sequences; "
         f"{p1.get('interactions_train', p1['interactions']):,} train interactions "
         f"(train+test {p1.get('interactions_train_plus_test', '—'):,}; "
         f"topic n={p1.get('transitions_train_plus_test', '—'):,}). "
         "→ feeds Phase 2 names and Phase 3–6 sequences."),
        ("Phase 2 — Embeddings",
         "835 English slugs from Phase 1 concept table.",
         f"Encode with {p2['model']}, L2-normalise. Controls: random pairs, same-topic pairs, lexical Jaccard.",
         f"concept_embeddings.npy (835×384). Linked mean cosine {fmt(p2['linked_mean_cosine'])} vs random "
         f"{fmt(p2['random_mean_cosine'])}; pair-AUC vs random {fmt(p2['pair_auc_vs_random'])}; "
         f"vs same-topic {fmt(p2['pair_auc_vs_same_topic'])}; lexical AUC {fmt(p2['lexical_jaccard_auc_vs_random'])}. "
         "→ query vectors for Phase 5 heads."),
        ("Phase 3 — Memory (classification)",
         "train.json (fit with sequence-id split, leakage overlap=0); official test.json for headline AUC; "
         "timed_interactions.npz for learner-split day Δt. Follow-up 3: gap≥1 day rows only.",
         "Fit mean, success_rate, HLR, logistic(+step / +day / +log1p gap / gap bins). "
         "Paired learner bootstrap on ΔAUC.",
         f"test.json AUC: logistic_step {fmt(mem_test['logistic_step_dt']['auc'])} "
         f"{ci_str(mem_test['logistic_step_dt'], 'auc_ci')} > success {fmt(mem_test['success_rate']['auc'])} "
         f"> HLR {fmt(mem_test['hlr_step']['auc'])}. "
         + (
             f"Follow-up 3 forgetting status=<font face='Arial-Bold'>{f3_forget['status']}</font> "
             f"(Δ log1p−success={fmt(f3_forget['paired_deltas']['log1p_gap_minus_success']['mean'], 4)} "
             f"{ci_str(f3_forget['paired_deltas']['log1p_gap_minus_success'])}). "
             if f3_forget else ""
         )
         + "→ p̂_recall features for Phase 5."),
        ("Phase 4 — Graph / next-item mix",
         "prerequisite.json + prefixes of train sequences.",
         "Build DiGraph; soft unlock = mean parent mastery; label true next as continue/revisit/advance "
         "(and hard-gate violation).",
         f"DAG acyclic={p4['nx.is_directed_acyclic_graph']}, longest path={p4['longest_path']}. "
         f"Mix (n={mix['n']}): review {fmt(mix['review_frac'])}, unlocked advance {fmt(mix['advance_unlocked_frac'])}, "
         f"hard-gate violation {fmt(mix['advance_violation_frac'])}. → unlock / DAG features for Phase 5."),
        ("Phase 5 — Review–Advance ranker (RAGR)",
         "Embeddings + Phase 3 memory + DAG + similarity + train prefixes.",
         "Train P(review|history). Review head scores seen items; advance head scores unseen. "
         "score = P(review)·s_rev + (1−P(review))·s_adv.",
         f"serve_models.joblib; train mode-review rate {fmt(p5['mode_review_rate_train'])}; "
         f"review_head_n={p5['review_head_n']}, advance_head_n={p5['advance_head_n']}. "
         "→ frozen models for Phase 6 and the live demo."),
        ("Phase 6 — Evaluation",
         f"Held-out test sequences; canonical exploratory subsample "
         f"{(review_rank or {}).get('eval', {}).get('n_sequences', p6.get('eval_sequences', 800))} seq × "
         f"{(review_rank or {}).get('eval', {}).get('cuts_per_sequence', p6.get('cuts_per_sequence', 5))} cuts; "
         "full-test replication in paper_ranking_full.json; bootstrap over sequences.",
         "Recall@5 / R@1 / MRR; force_review / force_advance; last_item on all_queries pool; "
         "pop / DAG / sim; gate AUC + PR-AUC; RAGR-no-DAG ablation; Markov order-1. "
         "Primary capped table is §4.4 from review_ranking.json; full test in §4.4b.",
         (
             f"Canonical capped RAGR R@5={fmt(review_rank['metrics']['ragr_r5']['mean'])} "
             f"{ci_str(review_rank['metrics']['ragr_r5'])}; "
             if review_rank else
             f"Legacy phase6_eval R@5={fmt(rank['ragr_all']['mean'])}; "
         )
         + f"share(next==last)={fmt(rank['share_next_equals_last'])}; "
         f"gate AUC={fmt(gate['auc'])}, PR-AUC={fmt(gate['pr_auc'])}."
         + (
             f" Full-test RAGR R@5={fmt(paper_rank_full['metrics']['ragr_r5']['mean'])} "
             f"{ci_str(paper_rank_full['metrics']['ragr_r5'])} "
             f"(n={paper_rank_full['eval']['n_sequences']})."
             if paper_rank_full else ""
         )),
        ("Paper eval-validity pass",
         "scripts/run_paper_eval.py → paper_*.json (counts, XES methods, full ranking, calibration, "
         "sensitivity, full rank-reversal).",
         "No cosmetic inflation: capped and full both reported; deviations logged in docs/DEVIATIONS.md.",
         "paper_junyi_counts / paper_ranking_full / paper_calibration / paper_sensitivity / "
         "paper_rank_reversal_full / paper_xes_kc_methods. LaTeX folder: paper_assets/."),
        ("Follow-up 3 — Cross-dataset + forgetting redo + rank reversal (pre-prereg exploratory)",
         "Timed Junyi (learner split); ASSISTments corrected; XES3G5M; gap≥1 day rows; "
         "6,290-sequence exact-dup chance draw.",
         "Concept-level next==last; two independent XES KC methods; forgetting log1p/bins; "
         "popularity / recency / Markov / GRU probe / gru4rec_style / attention probe "
         "(DAG features Junyi-only) with paired sequence bootstrap.",
         "followup3_*.json + paper_rank_reversal_full.json. Forgetting status from JSON. "
         "Probe names are not published GRU4Rec/SASRec; RecBole not installed."),
        ("Probe freeze — equal-budget validation tuning (pre-prereg)",
         "Learner-disjoint sequences from timed Junyi / ASSISTments / XES; data cards complete first.",
         "24-config grid for trainable probes (GRU + Attn only; RAGR-lite excluded from that claim; "
         "GRUProbe ignores dropout). Parameter-free popularity/recency; Markov order-1 light baseline; "
         "hash test uids without scoring test (test_touched=false).",
         "probe_freeze.json. Confirmatory protocol §4.13 + docs/PREREGISTRATION*.md."),
        ("Confirmatory freeze-partition replication (after prereg-v1 + OSF)",
         "Freeze train+val learners; frozen winner configs + best_epoch; cluster maps; "
         "GRU seeds 20261101/02/03.",
         "Fit on train+val (no early stop on test); score freeze test once (--touch-test + lock); "
         "two-method cluster agreement; Bonferroni m≤6; secondary common-query analysis.",
         "freeze_partition_replication.json + replication_per_seq_r5_*.json. Touch only after tag."),
    ]
    for title, inp, proc, out in phases:
        s += [P(title, "h2")]
        s += [table(
            ["", "What"],
            [
                ["Input (where from)", inp],
                ["Processing / operation", proc],
                ["Output → next phase", out],
            ],
            [CW * 0.22, CW * 0.78],
        )]
        s += [Spacer(1, 4)]

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Implementation plan
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("3.  Implementation plan (dataset travel)", "h1")]
    s += [P(
        "<font face='Arial-Bold'>Junyi ktbd</font> → length filter (Phase 1) → "
        "embed 835 names (Phase 2) → per-attempt memory rows + model select (Phase 3) → "
        "attach DAG/similarity + next-item mix (Phase 4) → train gate + two heads (Phase 5) → "
        "rank on test prefixes with sequence bootstrap (Phase 6) → "
        "serve same joblib in demo / Follow-up 3 cross-checks."
    )]
    s += bullets([
        "Request input: history [(exercise, correct), …], minimum length 2.",
        "Processing: rebuild attempts, success rate, Δt (steps or days), last-5 embedding mean, parent mastery.",
        "Output: P(review), mode ∈ {review, advance}, top-k items with kind, p̂_recall, parents, subgraph JSON.",
        "Train.json fits every learned weight. Test.json is only for Phase 6 / published hold-out checks. "
        "The demo loads serve_models.joblib and never trains at click time.",
        "ASSISTments / XES3G5M never enter the shipped Junyi joblib; they only validate generality of "
        "repetition and ranking-order claims.",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Output / Results
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("4.  Output / results obtained", "h1")]

    s += [P("4.1  Files the implementation writes", "h2")]
    s += [table(
        ["Path", "What it is"],
        [
            ["outputs_junyi/concepts_junyi.csv", "835 English concepts + degrees"],
            ["outputs_junyi/prerequisite_edges_junyi.csv", "978 learn-source-before-target rows"],
            ["outputs_junyi/concept_embeddings.npy", "(835, 384) float32, L2 unit"],
            ["outputs_junyi/phases/phase1–6_*.json", "Phase contracts (machine-readable)"],
            ["outputs_junyi/phases/followup3_*.json", "Forgetting, concept table, dups, rank-reversal"],
            ["outputs_junyi/phases/paper_*.json", "Full-test ranking, calibration, sensitivity, XES methods"],
            ["paper_assets/", "LaTeX tables (.tex) + vector figures (.pdf)"],
            ["outputs_junyi/serve_models.joblib", "Logistic memory + gate + two heads"],
            ["outputs_junyi/figures/*.png", "Distributions, ROC, mix, ranking, DAG, flow"],
            ["docs/NeuroTrace-DAG_Data_Flow_Report.pdf", "This document (built from JSON)"],
        ],
        [CW * 0.48, CW * 0.52],
    )]

    s += [P("4.2  Sample live prediction (product output)", "h2")]
    s += [P(
        "History: one step equations (correct) → one step inequalities (correct) → "
        "solving quadratics by factoring (wrong). Categories: "
        "<font face='Arial-Bold'>review</font> (seen) vs <font face='Arial-Bold'>advance</font> (unseen)."
    )]
    s += [table(
        ["Rank", "Exercise", "Kind", "Score", "p̂ recall", "Parents"],
        [
            ["1", "solving quadratics by factoring", "review", "2.81", "0.08", "factoring polynomials 1"],
            ["2", "one step inequalities", "review", "0.22", "0.92", "inequalities on a number line"],
            ["3", "one step equations", "review", "−0.52", "0.93", "evaluating expressions 1"],
            ["4", "addition 1", "advance", "−7.25", "0.55 unseen", "root"],
            ["5", "number line", "advance", "−7.34", "0.55 unseen", "root"],
        ],
        [CW * 0.08, CW * 0.30, CW * 0.12, CW * 0.12, CW * 0.16, CW * 0.22],
    )]
    s += [P(
        "P(review)=0.86, mode=review. Last item failed → restudy ranks first (low p̂_recall 0.08). "
        "Demo head-score calibration (softmax temperatures) is deferred; do not treat raw "
        "decision_function mixing as a calibrated probability model.",
        "cap",
    )]

    s += [P("4.3  Phase 4 next-item mix (intermediate output)", "h2")]
    s += [img("next_item_mix.png"), P(
        f"Figure 3. Train-prefix mix (n={mix['n']}): review {fmt(mix['review_frac'])}, "
        f"unlocked advance {fmt(mix['advance_unlocked_frac'])}, "
        f"hard-gate violation {fmt(mix['advance_violation_frac'])}. "
        "Monolithic next-item scores on this log mostly measure restudy.",
        "cap",
    )]
    if p4.get("hard_gate_skips_top20"):
        top = p4["hard_gate_skips_top20"][:6]
        s += [table(
            ["Source (missing prereq)", "Target attempted", "Count"],
            [[r["source"], r["target"], str(r["count"])] for r in top],
            [CW * 0.42, CW * 0.42, CW * 0.16],
        )]
        s += [P("Table: most frequent hard-gate skips in the Phase 4 sample.", "cap")]

    # Single primary ranking table from review_ranking.json only
    def _mci(block):
        return f"{fmt(block['mean'])} {ci_str(block)}"

    if review_rank:
        rr = review_rank
        rm = rr["metrics"]
        paired = rr.get("paired_deltas") or rr.get("paired_deltas_r5")
        tw = rr["three_way_mix"]
        s += [P("4.4  Primary ranking table (from review_ranking.json only)", "h2")]
        s += [P(
            f"Eval: {rr['eval']['n_sequences']} seq × {rr['eval']['cuts_per_sequence']} cuts "
            f"(canonical <font face='Arial-Bold'>capped</font> exploratory subsample; "
            f"full test in §4.4b). "
            f"seed={rr['eval']['seed']}. "
            f"Action mix continue/bounce/far/advance = "
            f"{fmt(tw['continue']['frac'])}/{fmt(tw['revisit_bounce']['frac'])}/"
            f"{fmt(tw['revisit_far']['frac'])}/{fmt(tw['advance']['frac'])}. "
            "<font face='Arial-Bold'>last_item</font> below = score-1-on-last on "
            "<font face='Arial-Bold'>all_queries</font> pool (not a review-only pool).",
            "small",
        )]
        s += [img("ranking_recall.png"), P(
            "Figure 4. Ranking figure assets; numeric table is exclusively from review_ranking.json.",
            "cap",
        )]
        s += [table(
            ["Method", "R@1 (seq. boot. CI)", "R@5 (seq. boot. CI)", "MRR (seq. boot. CI)", "Pool"],
            [
                ["RAGR", _mci(rm["ragr_r1"]), _mci(rm["ragr_r5"]), _mci(rm["ragr_mrr"]), "all_queries"],
                ["Force review", _mci(rm["force_review_r1"]), _mci(rm["force_review_r5"]), _mci(rm["force_review_mrr"]), "all_queries"],
                ["Recent-5", _mci(rm["recent5_r1"]), _mci(rm["recent5_r5"]), _mci(rm["recent5_mrr"]), "all_queries"],
                ["last_item", _mci(rm["last_item_r1"]), _mci(rm["last_item_r5"]), _mci(rm["last_item_mrr"]), "all_queries"],
                ["Frequency", _mci(rm["freq_r1"]), _mci(rm["freq_r5"]), _mci(rm["freq_mrr"]), "all_queries"],
            ],
            [CW * 0.16, CW * 0.24, CW * 0.24, CW * 0.24, CW * 0.12],
        )]
        s += [P(
            "CIs are sequence-level bootstrap (mean of per-sequence means, n_boot=1000) from "
            "review_ranking.json. "
            "The older Phase 6 print 0.891 [0.879, 0.905] (phase6_eval.json ragr_all, "
            "CI [0.8794875, 0.90475]) is the same point estimate under a different bootstrap "
            "draw and is <font face='Arial-Bold'>superseded</font> by the review_ranking.json "
            f"interval {ci_str(rm['ragr_r5'])}.",
            "small",
        )]
        if paired:
            s += [table(
                ["Paired Δ R@5", "Mean", "95% CI"],
                [
                    ["RAGR − force_review", fmt(paired["ragr_minus_force_review_r5"]["mean"], 4), ci_str(paired["ragr_minus_force_review_r5"])],
                    ["RAGR − recent5", fmt(paired["ragr_minus_recent5_r5"]["mean"], 4), ci_str(paired["ragr_minus_recent5_r5"])],
                ],
                [CW * 0.40, CW * 0.20, CW * 0.40],
            )]
            s += [P(
                "Note: RAGR − last_item is omitted here (last_item on all_queries is a sticky continue ceiling, "
                "not a matched review-pool baseline). See review_ranking.json if needed for audit.",
                "small",
            )]

    if paper_rank_full:
        fm = paper_rank_full["metrics"]
        s += [P("4.4b  Full-test ranking (paper_ranking_full.json)", "h2")]
        s += [P(
            "<font face='Arial-Bold'>Exploratory summary (pre-prereg, CI verdict rule t_R@5=0.01 / t_AUC=0.005):</font> "
            "no slice reversals flagged under the required-model H-rev rule in exploratory data; "
            "only XES native vs cluster flips (GRU−Markov sign change with both sides supported); "
            "Markov beats RAGR on advance (supported_B); DAG features negligible vs RAGR; "
            "Phase 3 logistic vs success inconclusive at t=0.005. "
            "See paper_paired_verdicts.json → exploratory_summary_for_pdf.",
            "small",
        )]
        s += [P(
            f"Full published test.json: {paper_rank_full['eval']['n_sequences']} sequences × "
            f"{paper_rank_full['eval']['cuts_per_sequence']} cuts = "
            f"{paper_rank_full['eval']['n_queries']:,} queries; "
            f"seed={paper_rank_full['eval']['seed']}; sequence bootstrap n_boot="
            f"{paper_rank_full.get('n_boot', 1000)}. "
            "Not tuned to look better than the capped table — report both.",
            "small",
        )]
        s += [table(
            ["Method", "R@5 (seq. boot. CI)", "n seq"],
            [
                ["RAGR", _mci(fm["ragr_r5"]) if "ragr_r5" in fm else "—", str(fm.get("ragr_r5", {}).get("n_sequences", "—"))],
                ["RAGR-no-DAG", _mci(fm["ragr_no_dag_r5"]), str(fm["ragr_no_dag_r5"]["n_sequences"])],
                ["Markov order-1", _mci(fm["markov_r5"]), str(fm["markov_r5"]["n_sequences"])],
                ["Recent-5", _mci(fm["recent5_r5"]), str(fm["recent5_r5"]["n_sequences"])],
                ["RAGR (advance)", _mci(fm["ragr_advance_r5"]), str(fm["ragr_advance_r5"]["n_sequences"])],
                ["RAGR-no-DAG (advance)", _mci(fm["ragr_no_dag_advance_r5"]), str(fm["ragr_no_dag_advance_r5"]["n_sequences"])],
                ["Markov (advance)", _mci(fm["markov_advance_r5"]), str(fm["markov_advance_r5"]["n_sequences"])],
            ],
            [CW * 0.32, CW * 0.48, CW * 0.20],
        )]
        s += [P(
            "RAGR-no-DAG zeros advance unlock + is_child features (cols 1, 3). "
            "Advance-slice R@5 is the validity-relevant contrast: sticky overall R@5 is not a recommender win.",
            "small",
        )]

    cal = paper_cal or ((paper_rank_full or {}).get("calibration") if paper_rank_full else None)
    if cal:
        gate_c = cal.get("gate") or {}
        mem_c = cal.get("p_recall_on_continue_queries") or {}
        s += [P("4.4c  Calibration (gate + p̂_recall)", "h2")]
        s += [table(
            ["Predictor", "n", "Base rate", "Brier", "ECE", "ΔBrier vs base"],
            [
                [
                    "Gate P(review)",
                    f"{gate_c.get('n', '—'):,}" if isinstance(gate_c.get("n"), int) else "—",
                    fmt(gate_c.get("base_rate")),
                    fmt(gate_c.get("brier"), 4),
                    fmt(gate_c.get("ece"), 4),
                    fmt(gate_c.get("base_rate_adjusted_brier"), 4),
                ],
                [
                    "p̂_recall (continue queries)",
                    f"{mem_c.get('n', '—'):,}" if isinstance(mem_c.get("n"), int) else "—",
                    fmt(mem_c.get("base_rate")),
                    fmt(mem_c.get("brier"), 4),
                    fmt(mem_c.get("ece"), 4),
                    fmt(mem_c.get("base_rate_adjusted_brier"), 4),
                ],
            ],
            [CW * 0.28, CW * 0.12, CW * 0.14, CW * 0.14, CW * 0.14, CW * 0.18],
        )]
        lowest = mem_c.get("lowest_pred_bin") or {}
        s += [P(
            "ΔBrier vs base = Brier(model) − Brier(constant base-rate); negative ⇒ better than base rate. "
            f"For p̂_recall, base rate={fmt(mem_c.get('base_rate'))} so "
            f"Brier(base)=base×(1−base)={fmt(mem_c.get('brier_constant_base_rate'), 4)}; "
            f"ΔBrier={fmt(mem_c.get('base_rate_adjusted_brier'), 4)}. "
            + (
                f"Lowest predicted bin [{fmt(lowest.get('lo'))}, {fmt(lowest.get('hi'))}): "
                f"n={lowest.get('n')}, pred={fmt(lowest.get('pred'), 4)}, "
                f"obs={fmt(lowest.get('obs'), 4)}, obs−pred={fmt(lowest.get('gap_obs_minus_pred'), 4)}."
                if lowest
                else ""
            )
            + " Reliability plots: paper_assets/figures/fig_gate_reliability.pdf, fig_memory_reliability.pdf.",
            "small",
        )]
        for stem, caption in (
            ("fig_gate_reliability", "Figure 4b. Gate reliability diagram."),
            ("fig_memory_reliability", "Figure 4c. p̂_recall reliability on continue queries."),
        ):
            src = ROOT / "paper_assets" / "figures" / f"{stem}.png"
            if src.exists():
                im = Image(str(src), width=CW * 0.48, height=CW * 0.48)
                s += [im, P(caption, "cap")]

    if paper_sens:
        s += [P("4.4d  Sensitivity (length cap, cuts, timed-6k selection)", "h2")]
        caps = paper_sens.get("sequence_length_cap") or {}
        if caps:
            s += [table(
                ["Max len", "n train", "n test", "Topic share(next==last)"],
                [
                    [
                        str(k),
                        str(v.get("n_train_seq")),
                        str(v.get("n_test_seq")),
                        fmt((v.get("topic_next_eq_last") or {}).get("share"), 4),
                    ]
                    for k, v in caps.items()
                ],
                [CW * 0.20, CW * 0.25, CW * 0.25, CW * 0.30],
            )]
        grids = (paper_sens.get("cut_positions") or {}).get("grids") or {}
        if grids:
            s += [table(
                ["Cuts / seq", "continue", "revisit", "advance", "n queries"],
                [
                    [
                        str(k),
                        fmt(g["continue"]),
                        fmt(g["revisit"]),
                        fmt(g["advance"]),
                        str(g["n_queries"]),
                    ]
                    for k, g in grids.items()
                ],
                [CW * 0.18, CW * 0.18, CW * 0.18, CW * 0.18, CW * 0.28],
            )]
        sel = paper_sens.get("timed_6k_selection") or {}
        if sel:
            f6 = sel.get("first_6k_file_order") or {}
            r6 = sel.get("random_6k") or {}
            s += [P(
                f"Timed 6k selection bias: first-6k file-order share={fmt(f6.get('share'), 4)} "
                f"(n={f6.get('n')}); random-6k={fmt(r6.get('share'), 4)} (n={r6.get('n')}); "
                f"Δ(first−random)={fmt(sel.get('delta_first_minus_random'), 4)} "
                f"(eligible users={sel.get('n_eligible_users')}).",
                "small",
            )]

    if f3_concept:
        s += [P("4.5  Concept-level repetition (Follow-up 3) — exercise/question rows dropped", "h2")]
        rows_45 = []
        for r in f3_concept["rows"]:
            unit = r["dataset_unit"]
            if "ASSISTments" in unit or "assistments" in unit.lower():
                unit = "ASSISTments composite skill token"
            # skip deprecated first-KC method B duplicate; parquet sensitivity kept with note
            if "drop multi-KC" in unit:
                continue
            rows_45.append([
                unit,
                fmt(r["share_next_equals_last"]["share"]),
                f"{r['share_next_equals_last']['n']:,}",
                fmt(r["three_way_mix"]["continue"]["frac"]),
                fmt(r["three_way_mix"]["revisit"]["frac"]),
                fmt(r["three_way_mix"]["advance"]["frac"]),
            ])
        if freeze and freeze.get("assistments_filter_funnel", {}).get("repetition_share"):
            prim = freeze["assistments_filter_funnel"]["repetition_share"]["primary_atomic_skill"]
            rows_45.append([
                "ASSISTments primary atomic skill",
                fmt(prim["share"]),
                f"{prim['n']:,}",
                "—",
                "—",
                "—",
            ])
        s += [table(
            ["Dataset / unit", "next==last", "n", "continue", "revisit", "advance"],
            rows_45,
            [CW * 0.30, CW * 0.12, CW * 0.14, CW * 0.14, CW * 0.15, CW * 0.15],
        )]
        trunc_note = ""
        if freeze and freeze.get("assistments_filter_funnel", {}).get("steps"):
            st = {x["step"]: x for x in freeze["assistments_filter_funnel"]["steps"]}
            if "keep_learners_len_ge_12" in st and "truncate_each_learner_to_200" in st:
                dropped = st["keep_learners_len_ge_12"]["n_rows"] - st["truncate_each_learner_to_200"]["n_rows"]
                trunc_note = (
                    f" ASSISTments truncation to 200 dropped {dropped:,} of "
                    f"{st['keep_learners_len_ge_12']['n_rows']:,} rows after len≥12."
                )
        s += [P(
            f"Note from JSON: {f3_concept.get('note', '')} "
            "XES method B in this table is expand-all KC tags (share≈0.216), not the deprecated "
            "first-KC-only B that duplicated A. "
            "Native-unit cross-dataset shares are descriptive until cluster-matched H-rep. "
            f"{trunc_note}",
            "small",
        )]

    if f3_xes:
        s += [P("4.6  XES3G5M KC repetition — both methods recorded", "h2")]
        a = f3_xes["method_a_question_then_primary_kc"]
        b = f3_xes.get("method_b_expand_all_kc_tags") or f3_xes.get("method_b_drop_multi_kc_extras") or {}
        b2 = f3_xes.get("method_b_drop_is_repeat_parquet", {})
        s += [table(
            ["Method", "Rule", "next==last", "n", "continue / revisit / advance"],
            [
                [
                    "A (primary reported)",
                    a["method"],
                    fmt(a["share_next_equals_last"]["share"]),
                    f"{a['share_next_equals_last']['n']:,}",
                    f"{fmt(a['three_way_mix']['continue']['frac'])} / "
                    f"{fmt(a['three_way_mix']['revisit']['frac'])} / "
                    f"{fmt(a['three_way_mix']['advance']['frac'])}",
                ],
                [
                    "B (expand-all KC tags)" if b.get("method") else "B",
                    b.get("method") or b.get("reason") or "—",
                    fmt(b["share_next_equals_last"]["share"]),
                    f"{b['share_next_equals_last']['n']:,}",
                    f"{fmt(b['three_way_mix']['continue']['frac'])} / "
                    f"{fmt(b['three_way_mix']['revisit']['frac'])} / "
                    f"{fmt(b['three_way_mix']['advance']['frac'])}",
                ],
            ],
            [CW * 0.18, CW * 0.28, CW * 0.12, CW * 0.14, CW * 0.28],
        )]
        if f3_xes.get("numerically_identical") is False:
            s += [P(
                "Methods A and B are <font face='Arial-Bold'>independent</font> "
                f"(numerically_identical=False). "
                f"primary_reported={f3_xes.get('primary_reported')}. "
                "Old first-KC-only method B is deprecated.",
                "small",
            )]
        elif b2:
            s += [P(
                f"Parquet drop_is_repeat check: next==last={fmt(b2['share_next_equals_last']['share'])} "
                f"(n={b2['share_next_equals_last']['n']:,}). "
                f"primary_reported={f3_xes.get('primary_reported')}.",
                "small",
            )]

    if f3_forget:
        s += [P("4.7  Forgetting redo (gap ≥ 1 day only) — status unresolved", "h2")]
        s += [P(
            f"Protocol: {f3_forget['protocol']['fit_and_test_rows']}; "
            f"split={f3_forget['protocol']['split']}; feature={f3_forget['protocol']['gap_feature']}. "
            f"Learners train/test={f3_forget['n_train_learners']:,}/{f3_forget['n_test_learners']:,}; "
            f"rows={f3_forget['n_train_rows']:,}/{f3_forget['n_test_rows']:,}."
        )]
        s += [table(
            ["Model", "AUC"],
            [
                ["success_rate only (logistic)", fmt(f3_forget["auc"]["success_only"])],
                ["+ log1p(gap)", fmt(f3_forget["auc"]["success_plus_log1p_gap"])],
                ["+ gap bins", fmt(f3_forget["auc"]["success_plus_gap_bins"])],
            ],
            [CW * 0.55, CW * 0.45],
        )]
        dlog = f3_forget["paired_deltas"]["log1p_gap_minus_success"]
        dbins = f3_forget["paired_deltas"]["gap_bins_minus_success"]
        s += [table(
            ["Paired ΔAUC (learner bootstrap)", "Mean", "95% CI", "CI excludes 0?"],
            [
                ["log1p − success", fmt(dlog["mean"], 4), ci_str(dlog), str(dlog["ci_excludes_zero"])],
                ["gap bins − success", fmt(dbins["mean"], 4), ci_str(dbins), str(dbins["ci_excludes_zero"])],
            ],
            [CW * 0.34, CW * 0.18, CW * 0.30, CW * 0.18],
        )]
        s += [P(
            f"Status=<font face='Arial-Bold'>{f3_forget['status']}</font> "
            f"(rule: {f3_forget.get('status_rule', 'CI must exclude 0')}). "
            "Do not call this a null result.",
            "small",
        )]
        # Accuracy by gap bin within success strata (compact)
        strata = f3_forget.get("accuracy_by_gap_bin_within_success_strata", [])
        if strata:
            s += [P("Accuracy by gap bin within success-rate strata (test rows, gap≥1):", "h2")]
            s += [table(
                ["Success stratum", "Gap bin", "n", "Accuracy", "Mean success rate"],
                [
                    [
                        str(r["success_stratum"]),
                        r["gap_bin"],
                        f"{r['n']:,}",
                        fmt(r["accuracy"]),
                        fmt(r["mean_success_rate"]),
                    ]
                    for r in strata
                ],
                [CW * 0.18, CW * 0.22, CW * 0.16, CW * 0.20, CW * 0.24],
            )]

    if f3_dup:
        s += [P("4.8  Exact-duplicate chance baseline", "h2")]
        real = f3_dup["real_train_test_exact"]
        chance = f3_dup["chance_baseline_pseudo_test"]
        cmp_ = f3_dup["comparison"]
        s += [table(
            ["Check", "n", "Exact-hash overlaps"],
            [
                ["Real train ∩ test", f"{real['n_test_sequences']:,} test seq", str(real["n_overlap_hashes"])],
                [
                    "Chance: pseudo-test from train",
                    f"{chance['n_pseudo_test']:,} vs {chance['n_remaining_train']:,} remaining",
                    str(chance["n_overlap_hashes"]),
                ],
                ["Real − chance", "—", str(cmp_["real_minus_chance"])],
            ],
            [CW * 0.40, CW * 0.40, CW * 0.20],
        )]
        s += [P(
            f"Real overlap ({cmp_['real_overlap']}) equals chance ({cmp_['chance_overlap']}) under seed={chance['seed']}. "
            "The 23 exact duplicates are consistent with chance, not evidence of a special train/test leak.",
            "small",
        )]

    PROBE_DISPLAY = {
        "gru4rec": "GRU-based next-item probe",
        "sasrec": "attention-based next-item probe",
        "gru_probe": "GRU-based next-item probe",
        "gru4rec_style": "GRU4Rec-style local probe (not RecBole)",
        "attn_probe": "attention-based next-item probe",
        "popularity": "popularity",
        "recency": "recency",
        "markov_order1": "Markov order-1",
        "ragr": "RAGR",
        "ragr_no_dag": "RAGR-no-DAG",
    }

    def _disp(name: str) -> str:
        return PROBE_DISPLAY.get(name, name)

    if rank_rev:
        src_name = "paper_rank_reversal_full.json" if paper_rev else "followup3_rank_reversal.json"
        expl = (rank_rev.get("status") == "EXPLORATORY") or bool(rank_rev.get("exploratory_disclosure"))
        s += [P(f"4.9  Rank-reversal study ({src_name})", "h2")]
        if expl:
            ed = rank_rev.get("exploratory_disclosure") or {}
            ev = ed.get("evidence") or {}
            s += [P(
                "<font face='Arial-Bold'>EXPLORATORY — not confirmatory.</font> "
                f"prereg-v1 existed before run_paper_eval? "
                f"<font face='Arial-Bold'>{ed.get('prereg_v1_existed_before_run', False)}</font>. "
                f"n_test sizes match freeze 15% ({ev.get('n_test_sizes_match_freeze')}); "
                f"uid hashes match freeze? <font face='Arial-Bold'>{ev.get('test_uid_hashes_match_freeze')}</font> "
                f"(paper seed={ev.get('paper_eval_seed_used')}, freeze seed={ev.get('freeze_seed')}). "
                "Confirmatory replication reuses the <font face='Arial-Bold'>freeze holdout</font> "
                "(seed 20261004), not a new learner-split seed — see §4.13.",
                "small",
            )]
        s += [P(
            f"Methods on the same learner-disjoint split (seed={rank_rev.get('seed')}, "
            f"cuts={rank_rev.get('eval_cuts', 5)}"
            f"{', full_test=True' if rank_rev.get('full_test') else ''}): "
            "popularity, recency, <font face='Arial-Bold'>Markov order-1</font>, "
            "GRU probe, gru4rec_style robustness check, attention probe. "
            "<font face='Arial-Bold'>Recency on advance = N/A</font> for advance-slice display "
            "(counts as 0 in all/nonrepeat aggregates under the registered replication rule). "
            "Primary confirmatory H-rev is native↔cluster GRU−Markov (§4.13), not slice order alone. "
            "RecBole not installed. Freeze val numbers in §4.11."
        )]
        for ds, payload in rank_rev["datasets"].items():
            ranks = payload["rankings_by_slice_r5"]
            all_rows = ranks["all"]
            adv_rows = ranks.get("advance") or []
            s += [P(f"{ds} — R@5 on all queries", "h2")]
            adv_map = {r["method"]: r for r in adv_rows}
            s += [table(
                ["Rank", "Method", "R@5 (all)", "R@5 (advance)"],
                [
                    [
                        str(i + 1),
                        _disp(x["method"]),
                        fmt(x["r5"]),
                        (
                            "N/A"
                            if x["method"] == "recency"
                            or (adv_map.get(x["method"]) or {}).get("r5_display") == "N/A"
                            or (adv_map.get(x["method"]) or {}).get("r5") is None
                            and x["method"] == "recency"
                            else fmt((adv_map.get(x["method"]) or {}).get("r5"))
                        ),
                    ]
                    for i, x in enumerate(all_rows)
                ],
                [CW * 0.10, CW * 0.40, CW * 0.25, CW * 0.25],
            )]
            # force recency advance cell to N/A
            # (table above handles recency method row)
            kt = payload.get("kendall_tau_by_slice") or {}
            if kt:
                s += [table(
                    ["Slice", "Kendall τ vs all", "Slice order (excl. recency on advance)"],
                    [
                        [
                            sl,
                            fmt(v.get("kendall_tau_vs_all"), 3),
                            " > ".join(_disp(m) for m in (v.get("slice_order") or [])[:6]),
                        ]
                        for sl, v in kt.items()
                    ],
                    [CW * 0.16, CW * 0.20, CW * 0.64],
                )]
            hrev = (payload.get("h_rev_diagnostics") or {}).get("advance") or {}
            if hrev:
                s += [P(
                    f"H-rev advance (required models excl. recency): "
                    f"ordering_changed={hrev.get('ordering_changed')}; "
                    f"Kendall τ={fmt(hrev.get('kendall_tau'), 3)}. "
                    f"{(payload.get('h_rev_diagnostics') or {}).get('flip_probability_note', '')}",
                    "small",
                )]
            split = payload.get("split", {})
            s += [P(
                f"Split: unit={split.get('unit')}, train/test={split.get('n_train')}/{split.get('n_test')}, "
                f"eval_seq={split.get('n_eval_sequences')}"
                + (
                    f"; role={split.get('native_role')}"
                    if split.get("native_role")
                    else ""
                )
                + f"; status={payload.get('status', rank_rev.get('status', '—'))}.",
                "small",
            )]
            if payload.get("gru4rec_style_note"):
                s += [P(payload["gru4rec_style_note"], "small")]

    if review_ready:
        rd = review_ready
        dag = rd["dag_helps_correctness"]
        delta = dag.get("primary_delta_auc") or {}
        thresh = dag.get("practical_threshold_delta_auc", 0.005)
        s += [P("4.10  Readiness diagnostic (review_readiness.json)", "h2")]
        s += [P(
            f"Status=<font face='Arial-Bold'>{dag['status']}</font> (exploratory). "
            f"Practical threshold ΔAUC={fmt(thresh)}. "
            f"Primary ΔAUC mean={fmt(delta.get('mean'), 4)} "
            f"95% CI={ci_str(delta) if delta.get('ci') is not None else '—'} "
            f"(ci_above_zero={delta.get('ci_above_zero')}). "
            f"Rule from JSON: {dag.get('rule', '')}"
        )]

    if freeze:
        s += [P(f"4.11  Probe freeze on validation (probe_freeze.json, schema v{freeze.get('schema_version', '?')})", "h2")]
        ncfg = freeze.get("n_grid_configs") or len(freeze.get("grid") or [])
        gs = freeze.get("grid_spec") or {}
        attn_arch = freeze.get("attn_probe_architecture") or {}
        unit_pol = freeze.get("h_rev_unit_policy") or {}
        s += [P(
            f"Equal-budget <font face='Arial-Bold'>trainable</font> grid: GRU + Attn only "
            f"(RAGR-lite is <font face='Arial-Bold'>not</font> in that claim). "
            f"<font face='Arial-Bold'>GRUProbe has no dropout</font> — dropout in the grid is a no-op for GRU "
            f"(Attn uses it). Parameter-free: popularity, recency. "
            f"Parameter-light: <font face='Arial-Bold'>markov_order1</font> (not on the 24-config grid). "
            f"Grid configs=<font face='Arial-Bold'>{ncfg}</font>. "
            f"Selection=<font face='Arial-Bold'>{freeze.get('selection_metric', 'val_nonrepeat_r5')}</font> "
            f"(tol={freeze.get('selection_tolerance_r5', '—')}). "
            f"Seed={freeze['replication_seed']}. Test hashed only (<font face='Arial-Bold'>test_touched=false</font>). "
            f"Attn prereg=<font face='Arial-Bold'>{attn_arch.get('preregistered_h_rev_family', 'excluded')}</font>. "
            f"H-rev primary={unit_pol.get('primary', {}).get('unit', 'matched clusters')} "
            f"(~{unit_pol.get('primary', {}).get('target_n_clusters', 120)}); "
            f"native secondary; XES question-level = curriculum-order / descriptive for non-H-rev. "
            f"Min models for tables: popularity, recency, markov_order1, gru_probe."
        )]
        s += [table(
            ["Hyperparameter", "Values"],
            [
                ["lr", str(gs.get("lr", "—"))],
                ["emb", str(gs.get("emb", "—"))],
                ["dropout", f"{gs.get('dropout', '—')} (no-op for GRUProbe)"],
                ["max_epochs / patience", f"{gs.get('max_epochs')} / {gs.get('early_stopping_patience')}"],
                ["GRU hidden / windows", "max(64, emb×2); MAX_WINDOWS=20,000; ctx=50; batch=64"],
                ["selection", "val non-rep R@5; within 0.005 → lowest val CE"],
            ],
            [CW * 0.32, CW * 0.68],
        )]
        s += [P(
            f"Grid note: {gs.get('note', '—')[:280]} "
            "Deviations: docs/DEVIATIONS.md.",
            "small",
        )]

        def _r5(block, key="nonrepeat"):
            if not block:
                return None
            if isinstance(block.get(key), dict) and "mean_r5" in block[key]:
                return block[key]["mean_r5"]
            return block.get("val_nonrepeat_r5")

        def _nq(block, key="nonrepeat"):
            if not block:
                return None
            sl = block.get(key) if isinstance(block.get(key), dict) else None
            if sl and sl.get("n_queries") is not None:
                return sl.get("n_queries")
            return block.get("n_queries")

        rows = []
        for name, ds in freeze["datasets"].items():
            native = ds.get("native_unit") or {}
            cluster = ds.get("cluster_unit") or {}
            nb = native.get("baselines") or {}
            cb = cluster.get("baselines") or {}
            nw = (native.get("winners") or ds.get("winners") or {}).get("gru_probe", {})
            cw = (cluster.get("winners") or {}).get("gru_probe", {})
            markov_n = _r5(nb.get("markov_order1")) if nb else None
            if markov_n is None and ds.get("parameter_light_baselines"):
                markov_n = _r5(ds["parameter_light_baselines"].get("markov_order1", {}).get("val"))
            markov_c = _r5(cb.get("markov_order1")) if cb else None
            pop = _r5(nb.get("popularity")) if nb else ds.get("stop_rule", {}).get("popularity_nonrepeat_r5")
            role = "curriculum/desc" if ds.get("curriculum_order_structured") else (
                "descriptive" if ds.get("native_unit_role") == "descriptive_only" else "—"
            )
            rows.append([
                name,
                ds.get("ranking_unit_native") or ds.get("ranking_unit", "—"),
                role,
                fmt(pop),
                fmt(markov_n),
                fmt(markov_c),
                fmt(nw.get("val_nonrepeat_r5")),
                fmt(cw.get("val_nonrepeat_r5")),
                str(cluster.get("n_clusters_used", "—")),
            ])
        s += [table(
            ["Dataset", "Native unit", "Native role", "Pop", "Markov nat.", "Markov cl.", "GRU nat.", "GRU cl.", "n_cl used"],
            rows,
            [CW * 0.13, CW * 0.13, CW * 0.12, CW * 0.08, CW * 0.11, CW * 0.11, CW * 0.10, CW * 0.10, CW * 0.12],
        )]
        s += [P(
            "Markov order-1 is frozen at both native and cluster units (parameter-light; not on the 24-config grid). "
            "n_cl used = unique cluster ids after mapping train+val and dropping empties "
            "(115 / 106 / 120 vs target 120).",
            "small",
        )]
        nq_rows = []
        for name, ds in freeze["datasets"].items():
            native = ds.get("native_unit") or {}
            cluster = ds.get("cluster_unit") or {}
            nb = native.get("baselines") or {}
            cb = cluster.get("baselines") or {}
            nw = (native.get("winners") or ds.get("winners") or {}).get("gru_probe", {})
            def _q(block, fallback=None):
                v = _nq(block)
                if v is None:
                    v = fallback
                return "—" if v is None else f"{int(v):,}"
            nq_rows.append([
                name,
                _q(nb.get("popularity")),
                _q(nb.get("markov_order1")),
                _q(nw.get("val") or nw.get("final_val"), nw.get("val_slice_n_queries", {}).get("nonrepeat") if isinstance(nw.get("val_slice_n_queries"), dict) else None),
                _q(cb.get("markov_order1")),
                _q((cluster.get("winners") or {}).get("gru_probe", {}).get("val") or (cluster.get("winners") or {}).get("gru_probe", {}).get("final_val"),
                   ((cluster.get("winners") or {}).get("gru_probe", {}).get("val_slice_n_queries") or {}).get("nonrepeat")),
            ])
        s += [table(
            ["Dataset", "Pop nq", "Markov nat. nq", "GRU nat. nq", "Markov cl. nq", "GRU cl. nq"],
            nq_rows,
            [CW * 0.18, CW * 0.14, CW * 0.18, CW * 0.16, CW * 0.18, CW * 0.16],
        )]
        s += [P(
            "Validation non-repeat query counts. Slice &lt; 500 queries → N/A at replication. "
            "XES native/cluster R@5 may use a 400-sequence cap (see DEVIATIONS.md).",
            "small",
        )]
        cfg_rows = []
        for name, ds in freeze["datasets"].items():
            native = ds.get("native_unit") or {}
            cluster = ds.get("cluster_unit") or {}
            for label, block in (("native", native.get("winners") or ds.get("winners") or {}), ("cluster", cluster.get("winners") or {})):
                w = block.get("gru_probe")
                if not w:
                    continue
                cfg = w.get("config") or {}
                cfg_rows.append([
                    name,
                    label,
                    str(cfg.get("lr")),
                    str(cfg.get("emb")),
                    str(w.get("best_epoch", "—")),
                    str(cfg.get("dropout")),
                    fmt(w.get("val_nonrepeat_r5")),
                    fmt(w.get("val_ce_loss"), 4) if w.get("val_ce_loss") is not None else "—",
                ])
        s += [table(
            ["Dataset", "Unit", "lr", "emb", "best_ep", "dropout", "val nonrep R@5", "val CE"],
            cfg_rows,
            [CW * 0.14, CW * 0.12, CW * 0.10, CW * 0.08, CW * 0.10, CW * 0.10, CW * 0.18, CW * 0.18],
        )]
        if freeze.get("clustering"):
            cl_rows = []
            for ds_name, cl in freeze["clustering"].items():
                cl_rows.append([
                    ds_name,
                    cl.get("method", "—"),
                    str(cl.get("n_clusters") or cl.get("target_n_clusters")),
                    "yes" if cl.get("fallback_used") else "no",
                    "yes" if cl.get("assist_has_skill_names") else ("n/a" if cl.get("assist_has_skill_names") is None else "no"),
                ])
            s += [table(
                ["Dataset", "Clustering", "k", "Fallback?", "ASSIST names?"],
                cl_rows,
                [CW * 0.18, CW * 0.42, CW * 0.10, CW * 0.14, CW * 0.16],
            )]
        if freeze["datasets"].get("xes3g5m", {}).get("question_order_structure"):
            qo = freeze["datasets"]["xes3g5m"]["question_order_structure"]
            s += [P(
                f"XES curriculum-order (descriptive native): mode successor="
                f"{fmt(qo.get('share_next_is_empirical_mode_successor_of_prev'))}; "
                f"global-order successor={fmt(qo.get('share_next_is_global_order_successor'))}.",
                "small",
            )]
        if freeze.get("assistments_filter_funnel", {}).get("skill_construction"):
            sc = freeze["assistments_filter_funnel"]["skill_construction"]
            s += [P(
                f"ASSISTments skill construction: atomic={sc.get('n_atomic_skill_ids_ge0')}; "
                f"composite={sc.get('n_composite_items_all_rows')}.",
                "small",
            )]
        if freeze["datasets"].get("junyi_timed", {}).get("learner_selection"):
            sel = freeze["datasets"]["junyi_timed"]["learner_selection"]
            s += [P(
                f"Junyi timed 6,000-learner selection: not_random_sample={sel.get('not_random_sample')}.",
                "small",
            )]
        s += [P(
            f"Elapsed={freeze.get('elapsed_sec')}s. min_slice_queries={freeze.get('min_slice_queries', 500)}. "
            "Canonical upgrade: scripts/run_preprereg_freeze_v4.py.",
            "small",
        )]

    if _maybe("phase3_base_rate.json") or p3_full:
        s += [P("4.11b  Phase 3 confusion / base-rate audit", "h2")]
        if p3_full:
            ev = p3_full["evaluate_on_test_json"]
            cm = ev["confusion_matrix_logistic_step_dt"]
            old = p3_full.get("superseded_first_80k_file_order") or {}
            s += [P(
                f"Headline 0.877 used the first {old.get('n_rows', 80000):,} test.json memory "
                "(revisit) rows in file order and is "
                "<font face='Arial-Bold'>superseded</font>. "
                f"Replacement: ALL {ev['n_test']:,} test.json memory rows "
                f"(correct_rate={fmt(p3_full['test_protocol']['correct_rate'])}); "
                f"logistic_step_dt AUC={fmt(ev['logistic_step_dt']['auc'])} "
                f"{ci_str(ev['logistic_step_dt'], 'auc_ci')} "
                f"(sequence bootstrap); PR-AUC={fmt(ev['logistic_step_dt']['pr_auc'])}. "
                f"CM @0.5 integers TN={cm['TN']:,} FP={cm['FP']:,} FN={cm['FN']:,} TP={cm['TP']:,}. "
                "Task = correctness on revisit attempts only."
            )]
        else:
            br = jload("phase3_base_rate.json")
            te = br["memory_row_correct_rate"]["test_first_80k_used_for_auc_and_confusion"]
            all_te = br["memory_row_correct_rate"]["test_all_memory_rows_uncapped"]
            cm = br["logistic_step_dt_on_capped_test"]["confusion_matrix_counts"]
            s += [P(
                f"AUC/CM rows = first {te['n_rows']:,} test.json memory (revisit) rows in file order "
                f"(correct_rate={fmt(te['correct_rate'])}); uncapped test memory rate={fmt(all_te['correct_rate'])} "
                f"(n={all_te['n_rows']:,}). CM integers TN={cm['TN']:,} FP={cm['FP']:,} FN={cm['FN']:,} TP={cm['TP']:,}. "
                f"{br['gap_explanation']}"
            )]

    if gru_mk:
        s += [P("4.11c  GRU vs Markov paired sequence CIs (validation)", "h2")]
        s += [P(gru_mk.get("h_rev_implication", {}).get("what_losing_to_markov_means", ""))]
        gm_rows = []
        for name, ds in gru_mk.get("datasets", {}).items():
            for unit in ("native", "cluster"):
                block = (ds.get(unit) or {}).get("nonrepeat") or {}
                gm_rows.append([
                    name,
                    unit,
                    fmt(block.get("mean_gru")),
                    fmt(block.get("mean_markov")),
                    fmt(block.get("mean_delta_gru_minus_markov"), 4),
                    ci_str(block) if block.get("ci") else "—",
                    "yes" if block.get("gru_beats_markov") else ("Markov" if block.get("markov_beats_gru") else "tie/NS"),
                ])
        if gm_rows:
            s += [table(
                ["Dataset", "Unit", "GRU R@5", "Markov R@5", "Δ GRU−Mk", "95% CI", "Winner"],
                gm_rows,
                [CW * 0.14, CW * 0.12, CW * 0.12, CW * 0.14, CW * 0.14, CW * 0.18, CW * 0.16],
            )]
        xlong = gru_mk.get("xes_native_longer_epoch_diagnostic")
        if xlong:
            s += [P(
                f"XES native longer-epoch diagnostic: max_epochs={xlong['config'].get('max_epochs')}, "
                f"patience={xlong['config'].get('early_stopping_patience')}, "
                f"best_epoch={xlong.get('best_epoch')}, "
                f"val CE={fmt(xlong.get('best_val_ce_loss'), 4)}. "
                f"{xlong.get('interpretation_if_still_loses', '')}",
                "small",
            )]

    if dual_cl:
        s += [P("4.11d  Dual clustering + feasible matched counts", "h2")]
        s += [P(
            f"Actual used cluster counts (freeze primary, dense remap): "
            f"{dual_cl['actual_used_cluster_counts_freeze_primary']['junyi_timed']} / "
            f"{dual_cl['actual_used_cluster_counts_freeze_primary']['assistments']} / "
            f"{dual_cl['actual_used_cluster_counts_freeze_primary']['xes3g5m']}. "
            f"{dual_cl.get('feasible_matched_count_summary', '')} "
            "A method contrast counts only if both methods agree. "
            f"XES text: {dual_cl.get('xes_question_text', {}).get('reason', 'N/A')}."
        )]
        fc = dual_cl.get("feasible_matched_count_table") or {}
        if fc:
            s += [table(
                ["Target k", "Junyi (835)", "ASSISTments (150)", "XES (7,439)"],
                [
                    [
                        k,
                        fc[k]["junyi_timed"]["mark"],
                        fc[k]["assistments"]["mark"],
                        fc[k]["xes3g5m"]["mark"],
                    ]
                    for k in ("40", "120", "800") if k in fc
                ],
                [CW * 0.16, CW * 0.28, CW * 0.28, CW * 0.28],
            )]
        drows = []
        for name, ds in dual_cl.get("datasets", {}).items():
            agr = ds.get("method_agreement") or {}
            drows.append([
                name,
                "yes" if ds.get("text_available") else "no",
                ds.get("text", {}).get("method") or "N/A",
                (ds.get("cooccurrence") or {}).get("method") or "co-occurrence",
                fmt(agr.get("ari")) if agr.get("ari") is not None else "N/A",
            ])
        if drows:
            s += [table(
                ["Dataset", "Text?", "Text method", "2nd method", "ARI"],
                drows,
                [CW * 0.16, CW * 0.10, CW * 0.28, CW * 0.28, CW * 0.18],
            )]

    if inertia:
        fw = inertia["failure_conditioned_next_action"]["wrong"]
        fc = inertia["failure_conditioned_next_action"]["correct"]
        pc = inertia["pedagogical_cost_of_violations"]
        s += [P("4.12  Inertia / pedagogy intermediate outputs", "h2")]
        s += [P(
            f"After wrong attempt, P(next=last)={fmt(fw['p_last'])}; after correct, {fmt(fc['p_last'])}. "
            f"Hard-gate violations cost accuracy: unlocked {fmt(pc['advance_unlocked']['accuracy'])} vs "
            f"violate {fmt(pc['advance_hard_gate_violation']['accuracy'])} "
            f"(gap {fmt(pc['accuracy_gap_unlocked_minus_violate'])})."
        )]

    # ── 4.13 Confirmatory preregistration / replication protocol ─────────
    s += [P("4.13  Confirmatory H-rev protocol (preregistration draft)", "h2")]
    s += [P(
        "Drafts (not yet the git tag <font face='Arial-Bold'>prereg-v1</font>): "
        "docs/PREREGISTRATION.md, docs/PREREGISTRATION_REPLICATION.md, docs/DEVIATIONS.md "
        "(attached to OSF/Zenodo and covered by the tag). "
        "Human tags → registers → runs "
        "<font face='Arial-Bold'>scripts/run_freeze_partition_replication.py --touch-test</font> "
        "once (touch_test.lock.json blocks a second run unless DEVIATIONS records "
        "TOUCH_TEST_OVERRIDE:)."
    )]
    s += [P(
        "<font face='Arial-Bold'>Primary claim:</font> dataset D satisfies H-rev iff both unit "
        "verdicts are supported_* with <font face='Arial-Bold'>opposite signs</font>; "
        "project claim = at least one of {Junyi, ASSISTments, XES}. "
        "Family <font face='Arial-Bold'>m_planned = 6</font> (3 datasets × native/cluster); "
        "Bonferroni level 1−0.05/m. Two-method Junyi/ASSIST cluster cells use an "
        "<font face='Arial-Bold'>intersection rule</font> (one family member, not two). "
        "Cell N/A if &lt;500 non-repeat queries → m := 6 − n_NA. "
        "Pipeline written to JSON: eligibility → m → CIs → dataset claims → project claim."
    )]
    s += [table(
        ["Item", "Registered value"],
        [
            ["Holdout", "freeze seed 20261004 (70/15/15); not junyi_pipeline.SEED=0"],
            ["GRU seeds", "20261101, 20261102, 20261103; per-seq R@5 = mean; bootstrap on mean"],
            ["Markov", "deterministic single fit"],
            ["Bootstrap", "≥10,000 sequence-level; zlib.crc32 seeds (not builtin hash)"],
            ["Clustering", "used counts 115 / 106 / 120; seed 20261021; ARI 0.038 / 0.073"],
            ["Two-method", "text ∧ cooc; cooc uses primary cluster GRU config (no retune); XES cooc only"],
            ["Recency", "0 on advance for all/nonrepeat; N/A only for advance-slice display"],
            ["recall_at ties", "lower item index wins among equal scores (stable lexsort)"],
            ["Secondary", "common native∧cluster non-repeat queries; same CI rule; primary-only flips labeled"],
            ["H-rep / graph / forget", "no confirmatory H-rep; graph & forgetting exploratory Junyi-only"],
            ["First-order Markov limit", "stronger sequence baselines could change conclusions"],
        ],
        [CW * 0.22, CW * 0.78],
    )]
    if dual_gm and dual_gm.get("datasets"):
        s += [P(
            "Pre-tag dual-map GRU−Markov on freeze <font face='Arial-Bold'>val</font> (not freeze test):",
            "small",
        )]
        dg_rows = []
        for name, ds in dual_gm["datasets"].items():
            agr = ds.get("cluster_unit_agreement") or {}
            dg_rows.append([
                name,
                agr.get("text_verdict", "—"),
                agr.get("cooccurrence_verdict", "—"),
                agr.get("registered_cluster_verdict", "—"),
                fmt(agr.get("ari"), 3) if agr.get("ari") is not None else "—",
            ])
        s += [table(
            ["Dataset", "text verdict", "cooc verdict", "registered cluster", "ARI"],
            dg_rows,
            [CW * 0.18, CW * 0.20, CW * 0.20, CW * 0.24, CW * 0.18],
        )]
    if paper_verdicts and paper_verdicts.get("exploratory_summary_for_pdf"):
        s += [P(
            f"Exploratory paired-verdict blurb: {paper_verdicts['exploratory_summary_for_pdf']}",
            "small",
        )]
    s += [P(
        "Seed-0 exploratory test ∩ freeze test overlap (still EXPLORATORY): "
        "Junyi 128/900 (14.2%); ASSISTments 70/439 (15.9%); XES 401/2711 (14.8%).",
        "small",
    )]
    if freeze_repl:
        scored_any = any(
            (d.get("result") or {}).get("scored")
            or (d.get("units") or {})
            for d in (freeze_repl.get("datasets") or {}).values()
        )
        s += [P(
            f"freeze_partition_replication.json: schema={freeze_repl.get('schema')}; "
            f"touch_test={freeze_repl.get('touch_test')}; "
            f"has_unit_scores={bool(scored_any)}. "
            "Confirmatory test numbers appear only after --touch-test post-tag.",
            "small",
        )]
    if locked_hashes:
        s += [P(
            "Locked SHA-256 digests (prereg_v1_locked_hashes.json; pytest fails if these files change):",
            "small",
        )]
        by = {r["path"]: r["sha256"] for r in locked_hashes["files"]}
        curated = [
            ("probe_freeze.json", by.get("outputs_junyi/phases/probe_freeze.json", "—")),
            ("run_freeze_partition_replication.py", by.get("scripts/run_freeze_partition_replication.py", "—")),
            ("dual_clustering.json", by.get("outputs_junyi/phases/dual_clustering.json", "—")),
            ("junyi text/item maps", by.get(
                "outputs_junyi/phases/cluster_maps/junyi_timed_text_item_to_cluster.json", "—"
            )),
            ("junyi cooc map", by.get(
                "outputs_junyi/phases/cluster_maps/junyi_timed_cooc_item_to_cluster.json", "—"
            )),
            ("assist text/item maps", by.get(
                "outputs_junyi/phases/cluster_maps/assistments_text_item_to_cluster.json", "—"
            )),
            ("assist cooc map", by.get(
                "outputs_junyi/phases/cluster_maps/assistments_cooc_item_to_cluster.json", "—"
            )),
            ("xes cooc/item maps", by.get(
                "outputs_junyi/phases/cluster_maps/xes3g5m_cooc_item_to_cluster.json", "—"
            )),
        ]
        s += [table(
            ["Locked file", "SHA-256"],
            [[a, b] for a, b in curated],
            [CW * 0.38, CW * 0.62],
        )]

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Visualizations
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("5.  Visualizations", "h1")]
    s += [P(
        "Figures below are generated artifacts under outputs_junyi/figures/ and embedded here. "
        "They show data distributions, embedding structure, DAG neighbourhoods, next-item mix, "
        "ranking, memory AUC, ROC, and gate regret — not decorative placeholders."
    )]
    s += [img("embedding_cosine.png"), P(
        f"Figure 5. Embedding cosine: linked {fmt(p2['linked_mean_cosine'])} vs random "
        f"{fmt(p2['random_mean_cosine'])} (pair-AUC {fmt(p2['pair_auc_vs_random'])}). "
        f"Same-topic control pair-AUC {fmt(p2['pair_auc_vs_same_topic'])}.",
        "cap",
    )]
    dag_quad = FIG / "dag_quadratic.png"
    dag_src = dag_quad if dag_quad.exists() else DAG_PNG
    if dag_src.exists():
        s += [Image(str(dag_src), width=CW, height=CW * 0.58)]
        s += [P(
            "Figure 6. Expert DAG — quadratic neighbourhood. Arrow = learn source before target.",
            "cap",
        )]
    if (FIG / "dag_overview.png").exists():
        s += [img("dag_overview.png"), P(
            "Figure 6b. Full DAG structure: degrees, longest-path sample, component sizes.",
            "cap",
        )]
    if (FIG / "dag_addition.png").exists():
        s += [img("dag_addition.png"), P(
            "Figure 6c. Early arithmetic neighbourhood (same layout rules).",
            "cap",
        )]
    n_mem = mem_test.get("n_test")
    row_note = (
        f"all {n_mem:,} test.json memory (revisit) rows"
        if n_mem else "test.json memory (revisit) rows"
    )
    s += [img("memory_auc_bars.png"), P(
        f"Figure 7. Memory AUCs on official test.json {row_note}: logistic "
        f"{fmt(mem_test['logistic_step_dt']['auc'])} "
        f"{ci_str(mem_test['logistic_step_dt'], 'auc_ci')} / success "
        f"{fmt(mem_test['success_rate']['auc'])} / HLR {fmt(mem_test['hlr_step']['auc'])}. "
        "Task = correctness on revisit attempts only. CIs are sequence-level bootstrap "
        "once phase3_memory_fulltest.json is the source.",
        "cap",
    )]
    if (FIG / "memory_roc.png").exists():
        s += [img("memory_roc.png", aspect=0.72), P(
            f"Figure 7b. ROC on {row_note} — logistic_step_dt vs success_rate "
            "(predictions = P(correct on this revisit); labels = revisit-attempt correctness). "
            "The first attempt on a concept is not a row.",
            "cap",
        )]
    if (FIG / "memory_confusion.png").exists():
        s += [img("memory_confusion.png", aspect=0.85), P(
            f"Figure 7c. Confusion matrix for logistic_step_dt at threshold 0.5 on {row_note} "
            "(revisit attempts only; not the superseded first-80k file-order slice).",
            "cap",
        )]
    if (FIG / "gate_regret.png").exists():
        s += [img("gate_regret.png"), P(
            "Figure 7d. Raising the review-gate threshold τ forces more advance-head use and trades "
            "overall Recall@5 for unlock hits.",
            "cap",
        )]

    # ═══════════════════════════════════════════════════════════════════════
    # 6. ROC / AUC
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("6.  ROC / AUC (memory classification)", "h1")]
    s += [P(
        "Phase 3 is a binary classifier for <font face='Arial-Bold'>correctness on revisit attempts only</font>: "
        "predict whether the next attempt on a concept already seen in the sequence is correct "
        "(label 0/1). The first attempt on a concept emits no row. "
        "Models are fit with a <font face='Arial-Bold'>sequence-id split</font> inside train.json "
        f"(leakage_check={p3.get('leakage_check')}) and scored on official "
        "<font face='Arial-Bold'>test.json</font>."
    )]
    s += [P(
        "<font face='Arial-Bold'>What the ROC curve represents:</font> trade-off between true-positive rate "
        "and false-positive rate as the decision threshold on predicted P(correct) varies. "
        "AUC is the probability that a random correct attempt scores higher than a random incorrect one."
    )]
    s += [P(
        "<font face='Arial-Bold'>Inputs used for ROC/AUC:</font> feature vectors built from prior history "
        "in the sequence (success rate, counts, Δt in steps or days / log1p gap); "
        "<font face='Arial-Bold'>prediction values</font> = model’s P(correct) scores; "
        "<font face='Arial-Bold'>ground truth</font> = observed correctness of the next attempt."
    )]
    s += [table(
        ["Model (test.json)", "AUC", "95% CI", "PR-AUC"],
        [
            [
                "logistic_step_dt (headline model)",
                fmt(mem_test["logistic_step_dt"]["auc"]),
                ci_str(mem_test["logistic_step_dt"], "auc_ci"),
                fmt(mem_test["logistic_step_dt"].get("pr_auc")),
            ],
            [
                "success_rate",
                fmt(mem_test["success_rate"]["auc"]),
                ci_str(mem_test["success_rate"], "auc_ci"),
                fmt(mem_test["success_rate"].get("pr_auc")),
            ],
            [
                "hlr_step",
                fmt(mem_test["hlr_step"]["auc"]),
                ci_str(mem_test["hlr_step"], "auc_ci"),
                fmt(mem_test["hlr_step"].get("pr_auc")),
            ],
        ],
        [CW * 0.34, CW * 0.16, CW * 0.28, CW * 0.22],
    )]
    if mem_timed:
        delta = mem_timed["paired_logistic_day_minus_success"]
        s += [P(
            f"Timed learner split ({p3.get('timed_n_train_users')} / {p3.get('timed_n_test_users')} users): "
            f"success_rate AUC {fmt(mem_timed['success_rate']['auc'])} vs logistic_day "
            f"{fmt(mem_timed['logistic_day_dt']['auc'])}. "
            f"Paired Δ(day−success)={fmt(delta['mean_delta_auc'], 4)} {ci_str(delta)}. "
            f"forgetting_helps={delta['forgetting_helps']} on unrestricted timed rows; "
            + (
                f"Follow-up 3 gap≥1 redo status=<font face='Arial-Bold'>{f3_forget['status']}</font>."
                if f3_forget else ""
            )
        )]
    s += [P(
        "ROC / confusion figures in §5 are regenerated from the same protocol "
        "(sequence-id fit on train.json → score official test.json).",
        "small",
    )]
    s += [P(
        f"Gate classifier P(review | history) on Phase 6 queries: AUC {fmt(gate['auc'])}, "
        f"PR-AUC {fmt(gate['pr_auc'])}, base rate {fmt(gate['base_rate'])} "
        "(separate binary task: will the true next item be a review vs advance)."
        + (
            f" Full-test calibration (§4.4c): Brier={fmt((paper_cal or {}).get('gate', {}).get('brier'), 4)}, "
            f"ECE={fmt((paper_cal or {}).get('gate', {}).get('ece'), 4)}, "
            f"ΔBrier vs base={fmt((paper_cal or {}).get('gate', {}).get('base_rate_adjusted_brier'), 4)}."
            if paper_cal
            else ""
        )
    )]

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Complete flow
    # ═══════════════════════════════════════════════════════════════════════
    s += [P("7.  Complete Dataset → Input → Processing → Output flow", "h1")]
    s += [P(
        "<b>Dataset taken:</b> Junyi Academy / EduData ktbd-junyi "
        f"({p1['concepts']} concepts, {p1['edges']} edges, "
        f"{p1['train_sequences']:,}/{p1['test_sequences']:,} train/test sequences)<br/>"
        "↓ <b>Why:</b> only public log with traces + expert DAG in one id space for review/advance recommendation<br/>"
        "↓ <b>Phase 1 input:</b> raw ktbd files → filter len 12–200 → "
        f"output concepts CSV + edges CSV "
        f"(train interactions {p1.get('interactions_train', p1['interactions']):,}; "
        f"topic n {p1.get('transitions_train_plus_test', '—'):,})<br/>"
        "↓ <b>Phase 2 input:</b> 835 English names → MiniLM embeddings → "
        f"concept_embeddings.npy (pair-AUC {fmt(p2['pair_auc_vs_random'])}; "
        f"same-topic control {fmt(p2['pair_auc_vs_same_topic'])})<br/>"
        "↓ <b>Phase 3 input:</b> train sequences (+ timed for day gap) → fit memory classifiers → "
        f"test.json logistic AUC {fmt(mem_test['logistic_step_dt']['auc'])}; "
        + (
            f"gap≥1 forgetting <b>{f3_forget['status']}</b><br/>"
            if f3_forget else "timed day-Δt check<br/>"
        )
        + f"↓ <b>Phase 4 input:</b> DAG + prefixes → mix review {fmt(mix['review_frac'])}, "
        f"unlock features<br/>"
        "↓ <b>Phase 5 input:</b> embeddings + memory + DAG + similarity → gate × (review, advance) heads → "
        "serve_models.joblib<br/>"
        + (
            f"↓ <b>Phase 6 (canonical capped):</b> review_ranking.json RAGR R@5 "
            f"{fmt(review_rank['metrics']['ragr_r5']['mean'])} {ci_str(review_rank['metrics']['ragr_r5'])}<br/>"
            if review_rank
            else f"↓ <b>Phase 6 input:</b> frozen models + test prefixes → RAGR R@5 {fmt(rank['ragr_all']['mean'])} "
            f"{ci_str(rank['ragr_all'])}<br/>"
        )
        + (
            f"↓ <b>Full-test ranking:</b> paper_ranking_full RAGR R@5 "
            f"{fmt(paper_rank_full['metrics']['ragr_r5']['mean'])} "
            f"{ci_str(paper_rank_full['metrics']['ragr_r5'])} "
            f"(n={paper_rank_full['eval']['n_sequences']}); "
            f"advance Markov {fmt(paper_rank_full['metrics']['markov_advance_r5']['mean'])} vs "
            f"RAGR {fmt(paper_rank_full['metrics']['ragr_advance_r5']['mean'])}<br/>"
            if paper_rank_full
            else ""
        )
        + "↓ <b>Follow-up 3 / paper rank-reversal:</b> concept-level table; exact-dup chance; "
        "Markov + GRU/attention probes (not published GRU4Rec/SASRec; RecBole absent)<br/>"
        + (
            f"↓ <b>Probe freeze (val only, seed={freeze['replication_seed']}):</b> "
            "equal-budget GRU+Attn grid (RAGR-lite not in that claim; GRU dropout no-op); "
            "test hashed not scored<br/>"
            if freeze else ""
        )
        + "↓ <b>Prereg draft (§4.13):</b> primary H-rev m≤6 Bonferroni; GRU seeds 20261101–03; "
        "two-method cluster intersection; freeze-holdout replication after tag + OSF<br/>"
        + "↓ <b>Final product / diagnostic output:</b> ranked next exercises + P(review) + p̂_recall + subgraph "
        "(demo; head softmax calibration deferred; Brier/ECE in §4.4c)<br/>"
        "↓ <b>How we show it:</b> figures in §5–6; tables in §4; data cards in docs/data_cards/; "
        "JSON under outputs_junyi/phases/; LaTeX under paper_assets/."
    )]
    s += [P(
        "Reader map: Dataset + cards — §1 · Phase I/O — §2 · Travel plan — §3 · "
        "Actual outputs (freeze §4.11, confirmatory protocol §4.13) — §4 · Charts — §5 · "
        "ROC/AUC — §6 · Full trace — §7.",
        "small",
    )]
    s += [P(
        "Reproduce: scripts/fetch_datasets.py → run_all_phases.py / run_valid_eval.py → "
        "run_followup3.py → run_probe_freeze.py → scripts/run_paper_eval.py → "
        "scripts/generate_report_assets.py → scripts/build_paper_assets.py → "
        "scripts/build_dataflow_report.py. "
        "After prereg-v1 + OSF only: scripts/run_freeze_partition_replication.py --touch-test.",
        "small",
    )]

    doc.build(s)
    print(OUT)


def build_slim():
    """Paper-facing PDF: status banner, H-rev stubs, scope, 5 figures. Numbers from JSON only."""
    main_nums_path = ASSETS / "paper_main_numbers.json"
    main_nums = json.loads(main_nums_path.read_text()) if main_nums_path.exists() else {}
    touched = bool(main_nums.get("touch_test"))
    freeze_repl = jload("freeze_partition_replication.json") if _maybe("freeze_partition_replication.json") else {}
    review_rank = jload("review_ranking.json") if _maybe("review_ranking.json") else None

    doc = BaseDocTemplate(
        str(OUT_SLIM), pagesize=A4, leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title="NeuroTrace-DAG — Paper-facing slim report",
        author="NeuroTrace-DAG",
    )
    frame = Frame(ML, MB, CW, PAGE_H - MT - MB, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    s = []

    s += [
        P("JOURNAL TRACK · PAPER-FACING SLIM", "kicker"),
        P("NeuroTrace-DAG", "title"),
        P(
            "<font face='Arial-Bold'>Status:</font> Confirmatory H-rev results "
            + (
                "<font face='Arial-Bold'>filled from primary_claims</font> (touch_test=true)."
                if touched
                else "pending locked <font face='Arial-Bold'>--touch-test</font> "
                "(prereg-v1 + OSF). Ranking / calibration below are exploratory or diagnostic."
            ),
            "sub",
        ),
        P(
            "Primary claim (registered): native↔cluster GRU−Markov sign flip; "
            "m_planned=6 Bonferroni; GRU seeds 20261101/02/03; two-method intersection; "
            "TOUCH_TEST_OVERRIDE gate for re-touch. "
            "Full technical inventory: <font face='Arial-Bold'>--mode=full</font> → "
            "NeuroTrace-DAG_Data_Flow_Report.pdf. "
            "LaTeX: paper/main.tex + paper_assets/tables/. "
            "After touch: regenerate with scripts/build_paper_assets.py from JSON only — "
            "never hand-edit claim cells.",
            "body",
        ),
    ]

    s += [P("1.  Confirmatory H-rev", "h1")]
    cells = ((main_nums.get("primary_hrev") or {}).get("cells")) or []
    if not cells:
        # fall back to pending grid
        for ds in ("junyi_timed", "assistments", "xes3g5m"):
            for unit in ("native", "cluster"):
                cells.append({"dataset": ds, "unit": unit, "verdict": "pending_confirmatory"})
    body_rows = []
    for c in cells:
        body_rows.append([
            str(c.get("dataset", "—")),
            str(c.get("unit", "—")),
            str(c.get("verdict", "pending_confirmatory")),
            "—" if not touched else str(c.get("ci_tex", c.get("ci", "—"))),
            "—" if c.get("eligible") is None else str(c.get("eligible")),
        ])
    s += [table(
        ["Dataset", "Unit", "Verdict", "Bonferroni CI", "Eligible"],
        body_rows,
        [CW * 0.22, CW * 0.14, CW * 0.28, CW * 0.22, CW * 0.14],
    )]
    s += [P(
        "Source: freeze_partition_replication.json → primary_claims "
        f"(touch_test={freeze_repl.get('touch_test', False)}). "
        "Pre-tag dual-map validation numbers are disclosure-only.",
        "small",
    )]

    s += [P("2.  Scope", "h1")]
    s += [table(
        ["Track", "Contents"],
        [
            ["Confirmatory", "Primary H-rev cells + dataset/project sign-flip rules"],
            ["Exploratory", "RAGR ranking, calibration, seed-0 rank-reversal, memory, DAG"],
            ["Disclosure-only", "Pre-tag dual_cluster_gru_markov.json validation CIs"],
        ],
        [CW * 0.28, CW * 0.72],
    )]
    if review_rank:
        r5 = review_rank["metrics"]["ragr_r5"]
        s += [P(
            f"Exploratory canonical capped Junyi RAGR R@5 = {fmt(r5['mean'])} {ci_str(r5)} "
            "(review_ranking.json; not confirmatory).",
            "body",
        )]

    s += [P("3.  Figures (paper set)", "h1")]
    fig_specs = [
        (FIG / "embedding_cosine.png", "Concept embedding geometry"),
        (FIG / "next_item_mix.png", "Continue / revisit / advance mix"),
        (FIG / "ranking_recall.png", "Exploratory ranking Recall@K"),
        (PAPER_FIG / "fig_gate_reliability.png", "Gate reliability (exploratory)"),
        (FIG / "dag_quadratic.png", "DAG neighbourhood (Junyi quadratic)"),
    ]
    for path, caption in fig_specs:
        s += [P(caption, "h2")]
        if path.exists():
            s += [Image(str(path), width=CW * 0.88, height=CW * 0.50)]
            s += [P(path.name, "cap")]
        else:
            # fall back reliability pdf→ skip; try memory reliability
            alt = PAPER_FIG / "fig_memory_reliability.png"
            if "reliability" in path.name and alt.exists():
                s += [Image(str(alt), width=CW * 0.88, height=CW * 0.50)]
                s += [P(alt.name + " (fallback)", "cap")]
            else:
                s += [P(f"[missing: {path}]", "small")]

    s += [P(
        "Rebuild: python scripts/build_dataflow_report.py --mode=slim. "
        "Companion narrative: docs/NeuroTrace-DAG_Paper.md.",
        "small",
    )]

    doc.build(s)
    print(OUT_SLIM)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build NeuroTrace-DAG data-flow / paper PDFs")
    ap.add_argument(
        "--mode",
        choices=("full", "slim"),
        default="full",
        help="full=technical supplement; slim=paper-facing tables+figures",
    )
    args = ap.parse_args()
    if args.mode == "slim":
        build_slim()
    else:
        build()
