# Documentation index

**Repo:** https://github.com/Oscar-man-shrestha/eval-validity-edu  
**OSF:** https://osf.io/24sxf  

→ **Full map:** [`DOCUMENTATION.md`](DOCUMENTATION.md)

## Status

OSF preregistration **submitted** (pending approval). Confirmatory `--touch-test` **not run**. Exploratory Junyi numbers are real but not confirmatory.

## Core reading

| Doc | What it is |
| --- | --- |
| [DOCUMENTATION.md](DOCUMENTATION.md) | Master index + rebuild commands |
| [DATASET_INPUT_FLOW_RESULTS.md](DATASET_INPUT_FLOW_RESULTS.md) | Datasets, phase flow, outputs, ROC/AUC |
| [NeuroTrace-DAG_Data_Flow_Report.pdf](NeuroTrace-DAG_Data_Flow_Report.pdf) | Illustrated PDF (`--mode=full`) |
| [NeuroTrace-DAG_Paper_Slim.pdf](NeuroTrace-DAG_Paper_Slim.pdf) | Slim paper-facing PDF |
| [NeuroTrace-DAG_Paper.md](NeuroTrace-DAG_Paper.md) | Journal write-up |
| [PREREGISTRATION.md](PREREGISTRATION.md) | Locked protocol |
| [PREREGISTRATION_REPLICATION.md](PREREGISTRATION_REPLICATION.md) | Replication steps |
| [DEVIATIONS.md](DEVIATIONS.md) | Deviations log |
| [OSF_PREREG_V1_DEPOSIT.md](OSF_PREREG_V1_DEPOSIT.md) | OSF deposit record |
| [RESEARCH_INTEGRITY_REPORT.md](RESEARCH_INTEGRITY_REPORT.md) | Scope / non-claims |
| [TEAM_FINDINGS_REPORT.md](TEAM_FINDINGS_REPORT.md) | Plain-language findings |
| [DATA_CARD.md](DATA_CARD.md) | Junyi ktbd card (`data_cards/` for others) |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | Release checklist |

## Regenerate

```bash
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=full
python3 scripts/build_dataflow_report.py --mode=slim
python3 paper/ieee/build_conference_docx.py
```

## Older material

`NeuroTrace-DAG_Pipeline.pdf`, `VALIDATED_RESULTS.md`, and early rank-probe tables are kept for traceability. Prefer this index + the integrity report for current claims.
