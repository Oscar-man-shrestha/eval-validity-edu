# Release checklist

## Blockers (must clear before public release)

- [ ] **Licence texts stored and verified** for ASSISTments and XES3G5M under `docs/data_cards/licences/` (currently UNVERIFIED). Junyi non-commercial clause captured; confirm redistribution policy.
- [ ] No bulk dataset files committed (`git status` clean of `data/**/*.csv`, `*.zip`, `train.json`, etc.).
- [ ] `prereg-v1` tagged and pushed; preregistration registered externally (OSF or Zenodo) for a timestamp (**human task**).
- [ ] Check EDM / LAK deadlines (**human task**).

## Scientific process

- [ ] Data cards complete for all four corpora before freeze/prereg.
- [ ] `probe_freeze.json` written; test_touched=false; grid has >1 config; curves logged.
- [ ] Preregistration committed with freeze (hash/tag in `replication_*.json`, not inside the md).
- [ ] F3 prior peeking disclosed.
- [ ] Replication uses new seed/split; full test; val dry-run first; test index hash logged.
- [ ] Probe names: GRU-based / attention-based next-item probe (not published GRU4Rec/SASRec).
- [ ] Claims in docs/PDF match JSON only.
- [ ] `pytest` green; `scripts/build_dataflow_report.py` regenerates PDF.

## Claims we do NOT make

- Recommender SOTA / practical win over recency
- Discovery of repetition bias (see Li et al. TOIS 2023)
- Faithful GRU4Rec/SASRec reproduction
- Forgetting or DAG readiness as supported effects without prereg verdict
