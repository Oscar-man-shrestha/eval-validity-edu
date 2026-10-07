# NeuroTrace-DAG documentation

## Read first

- [Dataset / Input / Flow / Results](DATASET_INPUT_FLOW_RESULTS.md) — **primary** document for datasets, phase-by-phase I/O, actual outputs, visualizations, and ROC/AUC.
- [Data Flow PDF](NeuroTrace-DAG_Data_Flow_Report.pdf) — same material with embedded figures (rebuild: `python3 scripts/build_dataflow_report.py`).
- [Research Integrity Report](RESEARCH_INTEGRITY_REPORT.md) — canonical scope, defensible findings, and non-claims.
- [Team Findings Report](TEAM_FINDINGS_REPORT.md) — plain-language explanation for teammates and teachers.
- [Data Card](DATA_CARD.md) — Junyi ktbd limitations and provenance (see also `docs/data_cards/`).
- [Shareable PDF report](../output/pdf/NeuroTrace-DAG_Research_Integrity_Report.pdf) — polished project brief generated from the audited JSON.

## Historical material

The earlier `NeuroTrace-DAG_Pipeline.pdf`, `VALIDATED_RESULTS.md`, and rank-probe tables are retained for traceability. They are not the final source for claims because they include pre-audit or exploratory outputs.

## Regenerate canonical documents

```bash
python3 scripts/build_research_integrity_audit.py
python3 scripts/render_research_integrity_docs.py
```
