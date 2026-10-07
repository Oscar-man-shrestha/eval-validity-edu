# When Next-Item Metrics Mislead: Evaluation Validity Across Educational Logs

> **Status**: Holdout confirmatory cells still blank until OSF registration + one `--touch-test`.  
> Junyi ranking / calibration numbers below are real, but exploratory.

_Journal-facing write-up. Confirmatory claim numbers load only from `freeze_partition_replication.json → primary_claims` via `scripts/build_paper_assets.py`—do not hand-edit those cells. Long technical inventory: `docs/NeuroTrace-DAG_Data_Flow_Report.pdf`._

## Abstract

Recall@K is the usual report card for next-item models on student logs. We are less sure it measures what people think it measures—especially once the “item” can mean a question, a skill, or a topic cluster. Across Junyi timed, ASSISTments 2009, and XES3G5M we lock a small comparison: a GRU probe versus first-order Markov at native units and at matched clusters (~120), under a freeze holdout and Bonferroni family \(m\leq 6\). The registered claim is a **native ↔ cluster sign flip** on that contrast. Holdout cells stay blank until OSF registration and one `--touch-test`. Junyi ranking and calibration stay exploratory.

## 1. Introduction

On educational logs, next-item Recall@K often rewards repeats and the curriculum’s natural order. That is not the same as measuring whether a model captured learning structure. Changing the action unit can flip which simple baseline wins.

Our goal is modest: check whether that unit choice systematically reverses a GRU-versus-Markov contrast under a locked protocol—not ship a new recommender champion.

**Registered question.** Does the sign of the paired GRU−Markov non-repeat Recall@5 contrast flip between the native unit and the registered cluster unit on at least one dataset?

## 2. Related work (brief)

Knowledge tracing and educational recommender evaluations routinely report Recall@K / NDCG on platform-native ids. Unit mismatch, duplicate next-items, and curriculum order are known confounders; we treat them as first-class design choices rather than post-hoc caveats. Faithful library baselines (e.g. RecBole GRU4Rec/SASRec) are out of confirmatory scope until installed and locked; local probes are labeled diagnostic.

## 3. Methods

### 3.1 Datasets and units

| Dataset | Native unit | Cluster unit (registered) |
| --- | --- | --- |
| Junyi timed | topic / exercise id | text MiniLM k-means (freeze primary); co-occurrence for two-method intersection |
| ASSISTments 2009 | composite skill token | text on `skill_name` (primary); co-occurrence for intersection |
| XES3G5M | question id | co-occurrence only (XES amendment; no local stem text) |

Freeze learner partition seed: `20261004`. Maps and freeze winners are locked under `prereg_v1_locked_hashes.json`.

### 3.2 Confirmatory H-rev protocol

Canonical wording (must match `docs/PREREGISTRATION.md`):

| Term | Registered wording |
| --- | --- |
| **Primary H-rev claim** | Native ↔ cluster **GRU−Markov sign flip**: dataset \(D\) satisfies it iff both unit verdicts are `supported_*` with **opposite signs**; project claim = ≥1 such dataset |
| **Family / Bonferroni** | \(m_{\mathrm{planned}}=6\); CI level \(1-0.05/m\); N/A cells reduce \(m\) |
| **GRU seeds** | `20261101`, `20261102`, `20261103` |
| **Two-method intersection** | Junyi/ASSIST cluster cell = one family member (both methods same `supported_*`); XES cooc-only amendment |
| **Touch-test lock** | Second `--touch-test` requires `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md` |

Pipeline written to replication JSON: eligibility → \(m\) → Bonferroni CIs → unit verdicts → dataset/project claims. Cell floor: ≥500 non-repeat queries (both methods for Junyi/ASSIST cluster). Bootstrap: ≥10,000 sequence-level resamples (`zlib.crc32` seeds). Required models for H-rev tables: popularity, recency, `markov_order1`, `gru_probe`. Attention / RAGR / H-rep / graph / forgetting are **not** in the confirmatory family.

### 3.3 Exploratory and diagnostic analyses

Reported for transparency, **not** as confirmatory claims: Phase 3 memory AUCs, Phase 6 / review_ranking RAGR, full-test ranking and calibration, seed-0 rank-reversal slices, dual-map **validation** GRU−Markov checks (disclosure-only), sensitivity grids. See Data Flow PDF §4 and `paper_assets/`.

## 4. Results

### 4.1 Primary confirmatory H-rev (pending)

Until `touch_test=true` in `outputs_junyi/phases/freeze_partition_replication.json`, every confirmatory cell is `pending_confirmatory`. Machine-readable stub: `paper_assets/tables/tab_hrev_confirmatory.tex` and `paper_assets/paper_main_numbers.json`.

| Dataset | Unit | Verdict | Bonferroni CI | Eligible |
| --- | --- | --- | --- | --- |
| junyi_timed | native | pending_confirmatory | — | — |
| junyi_timed | cluster | pending_confirmatory | — | — |
| assistments | native | pending_confirmatory | — | — |
| assistments | cluster | pending_confirmatory | — | — |
| xes3g5m | native | pending_confirmatory | — | — |
| xes3g5m | cluster | pending_confirmatory | — | — |

Pre-tag dual-map **validation** numbers in `dual_cluster_gru_markov.json` are disclosure-only and must not be read as confirmatory.

**After** `prereg-v1` + OSF + single `--touch-test`: regenerate this table with `PYTHONPATH=. python3 scripts/build_paper_assets.py` from `primary_claims` only. Never edit claim cells by hand.

### 4.2 Exploratory ranking and calibration (diagnostic)

Canonical capped Junyi RAGR R@5 remains `review_ranking.json` (see `paper_assets/tables/tab_ranking_capped_vs_full.tex`). Cross-dataset advance slices, reliability diagrams, and sensitivity tables are exploratory; interpret with the scope table `tab_scope_conf_vs_expl.tex`.

## 5. Discussion

What we registered is whether changing the action unit reverses the GRU−Markov contrast on the freeze holdout—not whether a gated ranker beats Recent-k on Junyi. A null or mixed result still counts. Native XES stays a descriptive curriculum-heavy case even though it sits in the family under an explicit amendment.

## 6. Limitations

- No library-faithful SASRec / RecBole baseline in confirmatory scope yet.
- ASSISTments and XES redistribution licences still unverified.
- Text vs co-occurrence cluster maps disagree (low ARI); intersection is deliberately conservative.
- No confirmatory holdout numbers until registration and one touch-test.

## 7. Data and code availability

Protocol: `docs/PREREGISTRATION.md`, `docs/PREREGISTRATION_REPLICATION.md`, `docs/DEVIATIONS.md`.  
Replication entrypoint: `scripts/run_freeze_partition_replication.py`.  
Paper numbers: `paper_assets/paper_main_numbers.json` (slim) vs `paper_assets/numbers.json` (full audit).  
Technical supplement: `docs/NeuroTrace-DAG_Data_Flow_Report.pdf` (`scripts/build_dataflow_report.py --mode=full`).  
Slim paper-facing PDF: `scripts/build_dataflow_report.py --mode=slim`.

## References

See repository `docs/` data cards and the Data Flow Report bibliography notes. Venue-ready BibTeX is deferred until confirmatory cells are filled.
