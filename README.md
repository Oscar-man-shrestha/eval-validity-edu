# Evaluation validity in educational logs

**NeuroTrace-DAG · evaluation-validity track** · Oscar Man Shrestha

This research repository asks whether next-item scores in educational activity logs reflect learning-aware structure or mainly reward repetition and curriculum order. It compares simple recency and Markov baselines with local GRU sequence probes, and examines how conclusions change when an action is represented as a question, topic, skill, or knowledge-component cluster. The contribution is an evaluation protocol and its audit, **not** a state-of-the-art recommender.

## Research status

- The protocol and input hashes are frozen in the [`prereg-v1` release](https://github.com/Oscar-man-shrestha/eval-validity-edu/releases/tag/prereg-v1). See [the preregistration](docs/PREREGISTRATION.md) and [replication instructions](docs/PREREGISTRATION_REPLICATION.md).
- The checked-in replication output records `"touch_test": false`. The confirmatory `--touch-test` holdout has **not** been run in this repository snapshot. The primary H-rev claim therefore remains pending; exploratory validation rankings must not be presented as confirmatory results.
- The repository documentation records an [OSF registration submission](https://osf.io/24sxf) on 2026-10-07. Check OSF directly for its current public status before citing a registration or DOI.
- The [research integrity report](docs/RESEARCH_INTEGRITY_REPORT.md) documents exploratory findings, practical-effect thresholds, limitations, and non-claims. In particular, the local sequence probes are not reproductions of published GRU4Rec or SASRec implementations.

## Repository guide

| Location | Purpose |
| --- | --- |
| [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) | Complete document map and regeneration instructions |
| [`docs/DATASET_INPUT_FLOW_RESULTS.md`](docs/DATASET_INPUT_FLOW_RESULTS.md) | Dataset inputs, pipeline stages, outputs, and reported metrics |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Locked confirmatory questions and analysis rules |
| [`docs/RESEARCH_INTEGRITY_REPORT.md`](docs/RESEARCH_INTEGRITY_REPORT.md) | Audited findings and limits on interpretation |
| [`docs/data_cards/`](docs/data_cards/) | Dataset provenance, licences, and download notes |
| [`scripts/`](scripts/) and [`tests/`](tests/) | Analysis, report generation, and protocol checks |
| [`outputs_junyi/phases/`](outputs_junyi/phases/) and [`paper_assets/`](paper_assets/) | Checked-in machine-readable results and generated paper assets |
| [`paper/ieee/`](paper/ieee/) | Conference paper sources and Word build script |
| [`CITATION.cff`](CITATION.cff) | Machine-readable citation for this repository's code |

## Data and setup

The project studies Junyi, ASSISTments 2009, and XES3G5M. Required bulk source data are **not** all included in Git. Obtain each dataset under its own terms and follow the [data cards](docs/data_cards/) for provenance and expected locations. The checked-in [`scripts/fetch_datasets.py`](scripts/fetch_datasets.py) **verifies local SHA-256 hashes**; despite its name, it does not download data.

Run commands below from the repository root with Python 3. The [requirements file](requirements.txt) lists direct dependencies for active analysis, tests, reports, and the optional demo. It is **not pinned** and does not guarantee an exact historical environment. Some paths download external model weights on first use. Check the [replication protocol](docs/PREREGISTRATION_REPLICATION.md) and dataset availability before attempting a full reproduction.

```bash
# Set up an isolated environment and install the listed dependencies.
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Verify the local source files and their expected hashes; missing data cause a nonzero exit.
python3 scripts/fetch_datasets.py

# Run the protocol checks without initiating the confirmatory test.
PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q -k 'not winner_retrain'

# Rebuild tables and figures from checked-in machine-readable results.
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=full
python3 paper/ieee/build_conference_docx.py
```

**Do not run `--touch-test` as a setup or smoke-test step.** It writes a one-time test lock and consumes the preregistered holdout. Read the [replication protocol](docs/PREREGISTRATION_REPLICATION.md) and [deviations log](docs/DEVIATIONS.md) before any confirmatory run.

## Interpreting and reusing the work

Start with the [full documentation map](docs/DOCUMENTATION.md) and [research integrity report](docs/RESEARCH_INTEGRITY_REPORT.md) when quoting results. The processed Junyi JSON split alone does not establish learner-disjointness; the [Junyi data card](docs/DATA_CARD.md) explains this limitation. Dataset licences and citation conditions are separate from any licence for this repository's code. **No repository-wide software licence is currently included**, so obtain permission before reusing code beyond what applicable law permits.

For proposed changes, see [CONTRIBUTING.md](CONTRIBUTING.md). Report protocol changes in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md); do not edit locked inputs or reported claim cells by hand.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the code and specify the Git tag or commit you used. Cite Junyi, ASSISTments, and XES3G5M under their own terms; see the [data cards](docs/data_cards/). The repository does not claim a paper DOI.
