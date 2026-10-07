"""Render the final research-facing docs from research_integrity_audit.json.

No numerical claim is typed into these documents. Every metric is read from the
audit JSON written by ``build_research_integrity_audit.py``.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs_junyi" / "phases"
DOCS = ROOT / "docs"


def fmt(value, digits=3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def ci(metric: dict, digits=3) -> str:
    lo, hi = metric.get("ci", [None, None])
    return "—" if lo is None else f"[{fmt(lo, digits)}, {fmt(hi, digits)}]"


def main() -> None:
    audit = json.loads((PHASE / "research_integrity_audit.json").read_text())
    findings = audit["validated_findings"]
    rec = findings["ragr_vs_recent5"]
    ready = findings["readiness_correctness"]
    forget = findings["forgetting_gap_ge_1_day"]
    concept = findings["cross_dataset_concept_repetition"]
    probe = audit["exploratory_sequence_probes"]

    rows = []
    for r in concept["rows"]:
        mix = r["three_way_mix"]
        rows.append(
            f"| {r['dataset_unit']} | {fmt(r['share_next_equals_last']['share'])} | "
            f"{r['share_next_equals_last']['n']} | {fmt(mix['continue']['frac'])} | "
            f"{fmt(mix['revisit']['frac'])} | {fmt(mix['advance']['frac'])} |"
        )

    report = [
        "# NeuroTrace-DAG — Research Integrity Report",
        "",
        "_Generated from `outputs_junyi/phases/research_integrity_audit.json`. Do not hand-edit metrics._",
        "",
        "## What this project is",
        "",
        f"**{audit['positioning']['final_project_type']}**.",
        "",
        "The project studies whether a standard next-item benchmark actually measures useful educational progression. "
        "It does not claim to have created a superior learning recommender.",
        "",
        "## Findings that survive the audit",
        "",
        "### 1. Action granularity changes what ‘repetition’ means",
        "",
        "| Dataset / concept unit | next == last | transitions | continue | revisit | advance |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "This is the main contribution: conclusions based on item-level repeat rates do not automatically transfer across topic, skill, and knowledge-component units.",
        "",
        "### 2. The prototype is not practically superior to recency",
        "",
        f"RAGR Recall@5 = **{fmt(rec['ragr_r5'])}**; Recent-5 = **{fmt(rec['recent5_r5'])}**. "
        f"The paired difference is **{fmt(rec['paired_delta_r5']['mean'], 4)}** {ci(rec['paired_delta_r5'], 4)}, "
        f"below the predeclared practical bar of **{fmt(rec['practical_threshold_r5'], 2)}**. "
        f"Status: **{rec['status']}**.",
        "",
        "### 3. Parent readiness is detectable but negligible",
        "",
        f"Primary readiness ΔAUC = **{fmt(ready['primary']['mean'], 4)}** {ci(ready['primary'], 4)}. "
        f"The practical threshold was **{fmt(ready['practical_threshold_delta_auc'], 3)}**. "
        f"Status: **{ready['status']}**.",
        "",
        "### 4. The timed forgetting question remains unresolved",
        "",
        f"On {forget['n_test_rows']} held-out gap≥1-day return rows from {forget['n_test_learners']} learners, "
        f"the success-only AUC is **{fmt(forget['auc']['success_only'])}** and the success+log-gap AUC is "
        f"**{fmt(forget['auc']['success_plus_log1p_gap'])}**. The paired ΔAUC is "
        f"**{fmt(forget['delta_log_gap_minus_success']['mean'], 4)}** {ci(forget['delta_log_gap_minus_success'], 4)}. "
        f"Status: **{forget['status']}**.",
        "",
        "## Method labels that are accurate",
        "",
        f"The rank-reversal file is **{probe['status']}**. The local neural probes trained for "
        f"{probe['reason']['epochs']} epochs on at most {probe['reason']['training_pair_cap']} windows using "
        f"{probe['reason']['loss']}; they are not faithful published GRU4Rec or SASRec implementations. "
        f"Use only the label **{probe['allowed_label']}** in exploratory notes.",
        "",
        "## What to say to a teacher or interviewer",
        "",
        "“We discovered that next-item performance can mostly reflect a dataset’s repetition structure. "
        "We introduced a reproducible protocol that separates continue, revisit, and advance actions; reports "
        "simple recency baselines; and checks the result at topic, skill, and KC levels across three educational logs. "
        "We deliberately report where the prototype does not show a practically meaningful gain.”",
        "",
        "## What we do not claim",
        "",
        *[f"- {x}" for x in audit['positioning']['prohibited_method_claims']],
        "",
        "## Research references",
        "",
        *[f"- [{x['citation']}]({x['url']}) — {x['use']}" for x in audit['references']],
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/build_research_integrity_audit.py",
        "python3 scripts/render_research_integrity_docs.py",
        "python3 -m pytest tests/test_leakage_guards.py -q",
        "```",
        "",
    ]
    (DOCS / "RESEARCH_INTEGRITY_REPORT.md").write_text("\n".join(report))

    paper = [
        "# When Next-Item Metrics Mislead: A Multi-Dataset Evaluation Study in Educational Logs",
        "",
        "_Canonical project paper. Generated from `research_integrity_audit.json`._",
        "",
        "## Abstract",
        "",
        "Educational next-item recommendation is commonly evaluated with aggregate Recall@K. We audit whether "
        "that metric reflects educational progression or the action structure induced by the platform. Across "
        "three public educational logs, we normalize the action unit to Junyi topics, ASSISTments skills, and "
        "XES3G5M knowledge components, and partition next actions into continue, revisit, and advance. "
        "We find materially different repetition structures across datasets. On Junyi, a prototype gated ranker "
        f"has Recall@5 {fmt(rec['ragr_r5'])}, but its advantage over Recent-5 ({fmt(rec['recent5_r5'])}) is "
        f"below a {fmt(rec['practical_threshold_r5'], 2)} practical threshold. We therefore present it as an "
        "exploratory diagnostic, not a new recommendation method. The contribution is a reproducible "
        "evaluation protocol that exposes action-granularity effects, duplicate sensitivity, and slice-dependent "
        "rankings while reporting null and negligible results honestly.",
        "",
        "## Contribution",
        "",
        "1. A unit-aware continue/revisit/advance protocol across three public logs.",
        "2. A practical-effect threshold alongside confidence intervals, rather than significance alone.",
        "3. A reproducibility pack that records data limitations, duplicate controls, and the distinction between "
        "published baselines and exploratory local probes.",
        "",
        "For detailed results, see `RESEARCH_INTEGRITY_REPORT.md`.",
        "",
    ]
    (DOCS / "NeuroTrace-DAG_Paper.md").write_text("\n".join(paper))
    (ROOT / "NeuroTrace-DAG_Paper.md").write_text("\n".join(paper))

    readme = [
        "# NeuroTrace-DAG documentation",
        "",
        "## Read first",
        "",
        "- [Research Integrity Report](RESEARCH_INTEGRITY_REPORT.md) — canonical scope, defensible findings, and non-claims.",
        "- [Team Findings Report](TEAM_FINDINGS_REPORT.md) — plain-language explanation for teammates and teachers.",
        "- [Data Card](DATA_CARD.md) — dataset limitations and provenance.",
        "- [Shareable PDF report](../output/pdf/NeuroTrace-DAG_Research_Integrity_Report.pdf) — polished project brief generated from the audited JSON.",
        "",
        "## Historical material",
        "",
        "The earlier `NeuroTrace-DAG_Pipeline.pdf`, `VALIDATED_RESULTS.md`, and rank-probe tables are retained "
        "for traceability. They are not the final source for claims because they include pre-audit or exploratory outputs.",
        "",
        "## Regenerate canonical documents",
        "",
        "```bash",
        "python3 scripts/build_research_integrity_audit.py",
        "python3 scripts/render_research_integrity_docs.py",
        "```",
        "",
    ]
    (DOCS / "README.md").write_text("\n".join(readme))
    (ROOT / "README.md").write_text("\n".join(readme))

    teacher = [
        "# NeuroTrace-DAG — Teacher and Team Brief",
        "",
        "_Generated from `research_integrity_audit.json`. This is the speaking guide for the final project scope._",
        "",
        "## The honest one-sentence project description",
        "",
        "We test whether next-item metrics in educational logs measure progression or mostly repeat behavior, using a reproducible protocol across three public datasets.",
        "",
        "## The three findings to explain",
        "",
        f"- Junyi topic actions continue at {fmt(concept['rows'][0]['share_next_equals_last']['share'])}; ASSISTments skills at {fmt(concept['rows'][1]['share_next_equals_last']['share'])}; XES3G5M KCs at {fmt(concept['rows'][2]['share_next_equals_last']['share'])}.",
        f"- Our prototype differs from Recent-5 by {fmt(rec['paired_delta_r5']['mean'], 4)} Recall@5 {ci(rec['paired_delta_r5'], 4)}, which is below the {fmt(rec['practical_threshold_r5'], 2)} practical threshold.",
        f"- The parent-readiness diagnostic is {ready['status']}; the timed forgetting analysis is {forget['status']}.",
        "",
        "## Questions a teacher may ask",
        "",
        "**Why is this useful if the prototype is not better?** It prevents a misleading conclusion: aggregate next-item accuracy can reward simple repetition rather than valid instructional progression.",
        "",
        "**What is novel?** The contribution is the evaluation protocol: action types are separated into continue, revisit, and advance; concept units are aligned across datasets; effect sizes are compared with practical thresholds; and null results are preserved.",
        "",
        "**What is the DAG used for?** The DAG represents declared prerequisite relations. We audit it as an observational readiness feature, while avoiding a causal claim that it improves learning.",
        "",
        "**What would be needed for a teaching recommendation claim?** A preregistered intervention or outcome study with learning-gain measures, not next-click recall alone.",
        "",
    ]
    (DOCS / "TEAM_FINDINGS_REPORT.md").write_text("\n".join(teacher))

    historical = [
        "# Historical result file — do not cite for final claims",
        "",
        "This file is retained for traceability. It contains pre-audit numbers and/or conclusions that are superseded by the generated canonical report.",
        "",
        "Read [RESEARCH_INTEGRITY_REPORT.md](RESEARCH_INTEGRITY_REPORT.md) for final evidence and [TEAM_FINDINGS_REPORT.md](TEAM_FINDINGS_REPORT.md) for the presentation guide.",
        "",
    ]
    (DOCS / "VALIDATED_RESULTS.md").write_text("\n".join(historical))
    (DOCS / "EXTERNAL_REVIEW_RESPONSE.md").write_text("\n".join(historical))
    print("wrote canonical research docs")


if __name__ == "__main__":
    main()
