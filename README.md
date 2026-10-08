# eval-validity-edu

Do next-item scores on educational logs track learning-aware structure, or mostly repeats / curriculum order? And does changing the **action unit** (question vs skill/topic cluster) flip which simple model looks better—GRU or Markov?

Oscar Man Shrestha — NeuroTrace-DAG evaluation-validity track.

## Links

| | |
| --- | --- |
| **GitHub** | https://github.com/Oscar-man-shrestha/eval-validity-edu |
| **OSF prereg** | https://osf.io/24sxf (submitted 2026-10-07; pending approval) |
| **Full doc map** | [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) |
| **Tag / release** | [`prereg-v1`](https://github.com/Oscar-man-shrestha/eval-validity-edu/releases/tag/prereg-v1) |

## Status

- Protocol + freeze hashes locked under `prereg-v1`
- OSF registration **submitted** (not public until approved)
- Confirmatory holdout **`--touch-test` not run yet** — H-rev cells stay blank on purpose
- Junyi ranking / calibration are **exploratory**

## Documentation (start here)

| Doc | Use |
| --- | --- |
| [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) | **Full map** of every doc and how to rebuild |
| [`docs/DATASET_INPUT_FLOW_RESULTS.md`](docs/DATASET_INPUT_FLOW_RESULTS.md) | Datasets → phase I/O → results → ROC/AUC |
| [`docs/NeuroTrace-DAG_Data_Flow_Report.pdf`](docs/NeuroTrace-DAG_Data_Flow_Report.pdf) | Same story with figures |
| [`paper/ieee/NeuroTrace-DAG_IEEE_A4_Conference.docx`](paper/ieee/NeuroTrace-DAG_IEEE_A4_Conference.docx) | IEEE Word for advisor |
| [`docs/NeuroTrace-DAG_Paper.md`](docs/NeuroTrace-DAG_Paper.md) | Journal narrative |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Locked confirmatory protocol |
| [`docs/OSF_PREREG_V1_DEPOSIT.md`](docs/OSF_PREREG_V1_DEPOSIT.md) | OSF URL / DOI + paste helpers |
| [`docs/RESEARCH_INTEGRITY_REPORT.md`](docs/RESEARCH_INTEGRITY_REPORT.md) | Scope and non-claims |
| [`docs/TEAM_FINDINGS_REPORT.md`](docs/TEAM_FINDINGS_REPORT.md) | Plain language for teammates |

## Quick regenerate

```bash
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=full
python3 scripts/build_dataflow_report.py --mode=slim
python3 paper/ieee/build_conference_docx.py
```
