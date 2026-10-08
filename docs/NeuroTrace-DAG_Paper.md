# When Next-Item Metrics Mislead: Evaluation Validity Across Educational Logs

> **Status**: OSF submitted ([osf.io/24sxf](https://osf.io/24sxf), pending approval). Holdout cells still blank until one `--touch-test`.  
> Junyi ranking / calibration numbers below are real, but exploratory.

_Journal-facing write-up. Confirmatory claim numbers load only from `freeze_partition_replication.json → primary_claims` via `scripts/build_paper_assets.py`—do not hand-edit those cells. Full doc map: [`DOCUMENTATION.md`](DOCUMENTATION.md). Longer inventory: `NeuroTrace-DAG_Data_Flow_Report.pdf`._

**Code:** https://github.com/Oscar-man-shrestha/eval-validity-edu (tag `prereg-v1`)  
**OSF:** https://osf.io/24sxf

## Abstract

Recall@K is the usual report card for next-item models on student logs. We are less sure it measures what people think it measures—especially once the “item” can mean a question, a skill, or a topic cluster. Across Junyi timed, ASSISTments 2009, and XES3G5M we lock a small comparison: a GRU probe versus first-order Markov at native units and at matched clusters (~120), under a freeze holdout and Bonferroni family \(m\leq 6\). The registered claim is a **native ↔ cluster sign flip** on that contrast. Holdout cells stay blank until OSF registration and one `--touch-test`. Junyi ranking and calibration stay exploratory.

## 1. Introduction

On educational logs, next-item Recall@K often rewards repeats and the curriculum’s natural order. That is not the same as measuring whether a model captured learning structure. Changing the action unit can flip which simple baseline wins.

Our goal is modest: check whether that unit choice systematically reverses a GRU-versus-Markov contrast under a locked protocol—not ship a new recommender champion.

**Registered question.** Does the sign of the paired GRU−Markov non-repeat Recall@5 contrast flip between the native unit and the registered cluster unit on at least one dataset?

## 2. Related work (brief)

KT and educational recommender papers usually score platform-native ids. Duplicates, unit mismatch, and curriculum order show up often as caveats; here they are the object of study. Library-faithful SASRec / RecBole GRU4Rec are outside this confirmatory packet for now; local probes are diagnostic.

## 3. Methods

### 3.1 Datasets and units

| Dataset | Native unit | Cluster unit (registered) |
| --- | --- | --- |
| Junyi timed | topic / exercise id | text MiniLM k-means (freeze primary); co-occurrence for two-method intersection |
| ASSISTments 2009 | composite skill token | text on `skill_name` (primary); co-occurrence for intersection |
| XES3G5M | question id | co-occurrence only (XES amendment; no local stem text) |

Freeze learner partition seed: `20261004`. Maps and freeze winners are locked under `prereg_v1_locked_hashes.json`.

### 3.2 Confirmatory comparison

Registered wording (must match `docs/PREREGISTRATION.md`):

| Term | Registered wording |
| --- | --- |
| **Primary claim** | Native ↔ cluster **GRU−Markov sign flip**: dataset \(D\) satisfies it iff both unit verdicts are `supported_*` with **opposite signs**; project claim = ≥1 such dataset |
| **Family / Bonferroni** | \(m_{\mathrm{planned}}=6\); CI level \(1-0.05/m\); N/A cells reduce \(m\) |
| **GRU seeds** | `20261101`, `20261102`, `20261103` |
| **Two-method intersection** | Junyi/ASSIST cluster cell = one family member (both methods same `supported_*`); XES cooc-only amendment |
| **Touch-test lock** | Second `--touch-test` needs `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md` |

Pipeline: eligibility → \(m\) → Bonferroni CIs → unit verdicts → dataset/project claims. Cell floor: ≥500 non-repeat queries. Bootstrap: ≥10,000 sequence-level resamples. Models in the confirmatory tables: popularity, recency, `markov_order1`, `gru_probe`. Attention / RAGR / graph / forgetting stay out of this family.

### 3.3 Exploratory Junyi work

Reported for context, not as confirmatory claims: Phase 3 memory AUCs, Phase 6 / review_ranking RAGR, full-test ranking and calibration, seed-0 rank-reversal slices, dual-map **validation** GRU−Markov checks (disclosure only), sensitivity grids. See Data Flow PDF §4 and `paper_assets/`.

## 4. Results

### 4.1 Primary confirmatory cells (pending)

Until `touch_test=true` in `outputs_junyi/phases/freeze_partition_replication.json`, every holdout cell stays blank on purpose. Machine stub: `paper_assets/tables/tab_hrev_confirmatory.tex` and `paper_assets/paper_main_numbers.json`.

| Dataset | Unit | Verdict | Bonferroni CI | Eligible |
| --- | --- | --- | --- | --- |
| junyi_timed | native | pending | — | — |
| junyi_timed | cluster | pending | — | — |
| assistments | native | pending | — | — |
| assistments | cluster | pending | — | — |
| xes3g5m | native | pending | — | — |
| xes3g5m | cluster | pending | — | — |

Validation dual-map numbers in `dual_cluster_gru_markov.json` are disclosure only—please do not read them as the confirmatory answer.

**After** OSF + one `--touch-test`: regenerate this table with `PYTHONPATH=. python3 scripts/build_paper_assets.py` from `primary_claims` only. Never edit claim cells by hand.

### 4.2 Exploratory ranking and calibration

Canonical capped Junyi RAGR R@5 remains `review_ranking.json` (see `paper_assets/tables/tab_ranking_capped_vs_full.tex`). Cross-dataset advance slices, reliability diagrams, and sensitivity tables are exploratory; see `tab_scope_conf_vs_expl.tex`.

## 5. Discussion

What we registered is whether changing the action unit reverses the GRU−Markov contrast on the freeze holdout—not whether a gated ranker beats Recent-k on Junyi. A null or mixed result still counts. Native XES stays a descriptive curriculum-heavy case even though it sits in the family under an explicit amendment.

## 6. Limitations

- No library-faithful SASRec / RecBole baseline in confirmatory scope yet.
- ASSISTments and XES redistribution licences still unverified.
- Text vs co-occurrence cluster maps disagree (low ARI); intersection is deliberately conservative.
- No confirmatory holdout numbers until registration and one touch-test.

## 7. Data and code availability

**GitHub:** https://github.com/Oscar-man-shrestha/eval-validity-edu (tag `prereg-v1`)  
**OSF preregistration:** https://osf.io/24sxf  
**Doc map:** [`DOCUMENTATION.md`](DOCUMENTATION.md)  
Protocol: `docs/PREREGISTRATION.md`, `docs/PREREGISTRATION_REPLICATION.md`, `docs/DEVIATIONS.md`.  
Replication entrypoint: `scripts/run_freeze_partition_replication.py`.  
Paper numbers: `paper_assets/paper_main_numbers.json` (slim) vs `paper_assets/numbers.json` (full audit).  
Technical supplement: `docs/NeuroTrace-DAG_Data_Flow_Report.pdf`.

## References

See `docs/` data cards and the Data Flow Report notes. Venue-ready BibTeX waits until confirmatory cells are filled.
