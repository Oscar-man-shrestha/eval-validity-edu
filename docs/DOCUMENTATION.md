# NeuroTrace-DAG — full documentation map

**Code:** https://github.com/Oscar-man-shrestha/eval-validity-edu  
**Tag:** `prereg-v1`  
**OSF registration:** https://osf.io/24sxf (submitted 2026-10-07; pending approval as of last check)  
**Author:** Oscar Man Shrestha  

This page is the index for everything you need to read, show an advisor, or regenerate.

---

## Current status (honest)

| Gate | Status |
| --- | --- |
| Protocol locked (`prereg_v1_locked_hashes.json`) | Done |
| Git tag `prereg-v1` + GitHub release | Done |
| OSF Preregistration | Submitted → https://osf.io/24sxf (approval / public DOI still pending) |
| Confirmatory `--touch-test` | **Not run** (`touch_test=false`) |
| H-rev paper cells | **Pending** (blank until touch-test) |
| Junyi ranking / calibration / diagnostics | Measured; labeled **exploratory** |

Do **not** treat validation dual-map numbers or Data Flow demo rankings as confirmatory H-rev.

---

## What to open for whom

| Audience | Open this |
| --- | --- |
| Advisor (Word, short) | [`paper/ieee/NeuroTrace-DAG_IEEE_A4_Conference.docx`](../paper/ieee/NeuroTrace-DAG_IEEE_A4_Conference.docx) |
| Advisor (dataset / flow / results) | [`DATASET_INPUT_FLOW_RESULTS.md`](DATASET_INPUT_FLOW_RESULTS.md) + [`NeuroTrace-DAG_Dataset_Input_Flow_Results.docx`](NeuroTrace-DAG_Dataset_Input_Flow_Results.docx) |
| Advisor / journal narrative | [`NeuroTrace-DAG_Paper.md`](NeuroTrace-DAG_Paper.md) |
| Figures + long PDF | [`NeuroTrace-DAG_Data_Flow_Report.pdf`](NeuroTrace-DAG_Data_Flow_Report.pdf) |
| Slim paper-facing PDF | [`NeuroTrace-DAG_Paper_Slim.pdf`](NeuroTrace-DAG_Paper_Slim.pdf) |
| Teachers / teammates (plain) | [`TEAM_FINDINGS_REPORT.md`](TEAM_FINDINGS_REPORT.md) |
| Integrity / non-claims | [`RESEARCH_INTEGRITY_REPORT.md`](RESEARCH_INTEGRITY_REPORT.md) |

---

## Locked protocol (do not edit after seeing test)

| Doc | Role |
| --- | --- |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Primary claim wording, seeds, Bonferroni, intersection |
| [`PREREGISTRATION_REPLICATION.md`](PREREGISTRATION_REPLICATION.md) | How to reproduce the confirmatory run |
| [`DEVIATIONS.md`](DEVIATIONS.md) | Logged deviations; `TOUCH_TEST_OVERRIDE:` if ever needed |
| [`OSF_PREREG_V1_DEPOSIT.md`](OSF_PREREG_V1_DEPOSIT.md) | OSF URL / DOI record + paste helpers |
| `outputs_junyi/phases/prereg_v1_locked_hashes.json` | SHA-256 lock set |
| `docs/osf_prereg_v1_packet.zip` | Upload packet for OSF |

---

## Data and results

| Doc | Role |
| --- | --- |
| [`DATASET_INPUT_FLOW_RESULTS.md`](DATASET_INPUT_FLOW_RESULTS.md) | **Primary** end-to-end: datasets → phases → outputs → ROC/AUC |
| [`DATA_CARD.md`](DATA_CARD.md) + [`data_cards/`](data_cards/) | Provenance and licence notes |
| [`VALIDATED_RESULTS.md`](VALIDATED_RESULTS.md) | Older validated summary (prefer integrity report for claims) |
| `outputs_junyi/phases/*.json` | Machine numbers (source of truth) |
| `paper_assets/` | TeX tables + `paper_main_numbers.json` |

---

## Paper / conference

| Path | Role |
| --- | --- |
| [`../paper/ieee/`](../paper/ieee/) | IEEE A4 Word + LaTeX |
| [`../paper/main.tex`](../paper/main.tex) | Thin journal skeleton |
| [`NeuroTrace-DAG_Paper.md`](NeuroTrace-DAG_Paper.md) | Markdown journal write-up |
| [`../paper_assets/README.md`](../paper_assets/README.md) | How tables are filled |

---

## How to regenerate docs

```bash
# Paper tables / slim numbers (after touch-test: fills H-rev from primary_claims only)
PYTHONPATH=. python3 scripts/build_paper_assets.py

# Full Data Flow PDF (figures) and slim PDF
python3 scripts/build_dataflow_report.py --mode=full
python3 scripts/build_dataflow_report.py --mode=slim

# IEEE conference Word
python3 paper/ieee/build_conference_docx.py

# Integrity report PDF (optional)
python3 scripts/build_research_integrity_audit.py
python3 scripts/render_research_integrity_docs.py
```

### After OSF is public — confirmatory once

```bash
PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q -k 'not winner_retrain'
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=full
python3 scripts/build_dataflow_report.py --mode=slim
python3 paper/ieee/build_conference_docx.py
```

Never hand-edit H-rev claim cells. Never re-touch without `TOUCH_TEST_OVERRIDE:` in `DEVIATIONS.md`.

---

## Study question (one sentence)

Do next-item metrics on educational logs measure learning-aware structure, or mostly repetition / curriculum order—and does **unit granularity** flip which model looks better (GRU vs Markov)?

Contribution is **evaluation validity**, not recommender SOTA.
