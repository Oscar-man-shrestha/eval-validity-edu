# Preregistration draft (not registered)

Draft updated 2026-10-07 (final pre-tag edits). This file is **not** an OSF or Zenodo registration and it is **not** the git tag `prereg-v1`. Complete the checklist at the end before tagging. The human tags only after reading this file.

Companion: `docs/PREREGISTRATION_REPLICATION.md`. Deviations: `docs/DEVIATIONS.md`.

### Canonical wording (shared with REPLICATION.md / DEVIATIONS.md)

| Term | Registered wording |
| --- | --- |
| **Primary H-rev claim** | Native ↔ cluster **GRU−Markov sign flip**: dataset \(D\) satisfies it iff both unit verdicts are `supported_*` with **opposite signs**; project claim = ≥1 such dataset |
| **Family / Bonferroni** | \(m_{\mathrm{planned}}=6\); CI level \(1-0.05/m\); N/A cells reduce \(m\) |
| **GRU seeds** | `20261101`, `20261102`, `20261103` |
| **Two-method intersection** | Junyi/ASSIST: same `supported_*` on text and cooc → that verdict; else inconclusive (one family cell). XES: cooc-only amendment |
| **Touch-test lock** | Second `--touch-test` requires exact marker `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md` |

Exploratory ranking, seed-0 rank-reversal, dual-map **val** checks, H-rep, graph, and forgetting are **not** part of the primary claim paragraph below.

## Question

Do aggregate next-item Recall@K scores mostly reflect continue / revisit / advance structure, and does a probe's rank against popularity, recency, and a first-order Markov table reverse when the action unit changes from the native catalog to matched concept clusters?

## Phase 3 label (Junyi ktbd only)

The memory task is **correctness on revisit attempts only**. A memory row exists only when a concept is attempted again in the same sequence. The first attempt emits no row.

Headline 0.877 is superseded (first 80k file-order rows). Registered descriptive set: all **449,453** revisit rows; logistic_step_dt AUC **0.864** [0.861, 0.867] (95% sequence bootstrap). **Phase 3 is exploratory** (not in the confirmatory H-rev family). Under the CI verdict rule with \(t=0.005\), paired Δ (logistic − success) CI [0.0046, 0.0059] is **inconclusive**.

## Datasets and units

| Dataset | Native unit | Role of native | Primary cluster map (pinned used count) |
| --- | --- | --- | --- |
| Junyi timed | exercise (835) | secondary | text MiniLM **115** used / target 120; co-occurrence also scored for agreement |
| ASSISTments 2009 | composite skill token (150) | secondary | text MiniLM on `skill_name` **106** used; co-occurrence also scored |
| XES3G5M | question (7,439) | curriculum-order **descriptive** for non-H-rev claims; **allowed in primary H-rev native arm** under the XES amendment below | co-occurrence only **120** used; text N/A |

Pinned seeds: learner split **`20261004`** (70/15/15); clustering **`20261021`** (`dual_clustering.json`). Confirmatory replication scores the **freeze test** once after tagging — see `docs/PREREGISTRATION_REPLICATION.md`. Do **not** use `junyi_pipeline.SEED=0` for confirmatory work.

### Registered GRU training seeds

Confirmatory GRU probes are trained with three fixed seeds: **`20261101`**, **`20261102`**, **`20261103`**.

- Per-sequence R@5 for GRU = **mean over the three seeds**.
- Sequence-level bootstrap (≥ 10,000 resamples) is run on that seed-averaged per-sequence series.
- Report **between-seed spread** (per-seed non-repeat mean R@5, std, range).
- **Markov is deterministic** (single fit; no seed average).
- Training matches freeze structure: `hidden = max(64, emb×2)`, context length 50, batch 64, pair cap **`MAX_WINDOWS = 20_000`**.
- **`GRUProbe` has no dropout module** — the freeze grid’s dropout field is a **no-op for GRU** (recorded for audit only; AttnProbe uses dropout).

### Recency, alignment, and Recall@k ties

- Per-sequence R@5 is stored as **arrays indexed by sequence** (NaN where undefined). Methods are paired by **sequence index**, never by compacted list position.
- **Recency** counts as **0** on advance queries for the `all` and `nonrepeat` aggregates; it is **N/A only for display/ordering on the advance slice**.
- **`recall_at` tie rule:** among items with equal scores, the **lower item index** wins (stable lexsort on `(-score, index)`).

### Seed-0 exploratory vs freeze test overlap

`paper_rank_reversal_full.json` used seed 0. Overlap with freeze test learners (seed 20261004):

| Dataset | \|test₀ ∩ test_freeze\| | \|test_freeze\| | % of freeze test |
| --- | ---: | ---: | ---: |
| Junyi timed | 128 | 900 | **14.2%** |
| ASSISTments | 70 | 439 | **15.9%** |
| XES3G5M | 401 | 2711 | **14.8%** |

Those seed-0 results remain **EXPLORATORY**.

## Verdict rule (CI threshold; no “tie”)

For paired Δ with CI at the registered level and threshold \(t\):

| Verdict | Rule |
| --- | --- |
| **supported_A** | entire CI \(> +t\) (GRU − Markov \(> 0\) beyond \(t\)) |
| **supported_B** | entire CI \(< -t\) |
| **negligible** | entire CI \(\subset (-t,+t)\) |
| **inconclusive** | otherwise |

- R@5: \(t = 0.01\). AUC: \(t = 0.005\) (exploratory only).
- Confirmatory primary H-rev CIs: **sequence-level** bootstrap with **≥ 10,000** resamples; Bonferroni level \(1 - 0.05/m\) (see family size).

## Confirmatory family and Bonferroni (\(m\))

**Planned family size \(m = 6\)**: one GRU−Markov non-repeat R@5 CI per **dataset × unit** cell (3 datasets × {native, cluster}).

**Why \(m = 6\) (not 3):** the primary claim is built from six unit-level confirmatory tests. Each cell produces a separate paired CI/verdict. Bonferroni controls family-wise error across those six CIs at α = 0.05. Dataset-level and project-level claims are **logical combinations** of those unit verdicts; they are not additional independent tests that enlarge \(m\).

**Two-method cluster cells and \(m = 6\):** for Junyi and ASSISTments, the registered cluster-unit verdict requires **both** clustering methods to be `supported_*` with the **same** label (intersection / agreement rule). That intersection is a **single** registered cell verdict in the family of 6 — not two separate family members. Therefore the Bonferroni level \(1 - 0.05/6\) remains valid for those cells; we do **not** inflate \(m\) to count text and co-occurrence as separate confirmatory tests.

| \(m\) | CI level \(1-0.05/m\) | When |
| ---: | ---: | --- |
| 6 | ≈ **0.9917** | all six cells eligible (≥500 non-repeat queries) |
| 5 | **0.9900** | one cell N/A |
| 4 | **0.9875** | two cells N/A |
| 3 | ≈ **0.9833** | three cells N/A |
| 2 | **0.9750** | four cells N/A |
| 1 | **0.9500** | five cells N/A |
| 0 | undefined | all six N/A — no confirmatory H-rev CIs |

**Minimum cell size:** a primary comparison cell is eligible only if it has **≥ 500 non-repeat queries** on the freeze test (Junyi/ASSIST cluster cell: **both** methods ≥ 500). Below that the cell is **N/A** and \(m := m_{\mathrm{planned}} - n_{\mathrm{N/A}}\) under the table above (pre-registered; do not invent another reduction). A dataset with either unit N/A **cannot** satisfy the dataset-level primary claim.

**Implemented pipeline (written to replication JSON):** eligibility → \(m = 6 - n_{\mathrm{NA}}\) → Bonferroni level → CIs at that level → dataset-level claim (opposite `supported_*`) → project-level claim (≥1 dataset).

| # | Dataset | Unit | Notes |
| --- | --- | --- | --- |
| 1 | Junyi timed | native | |
| 2 | Junyi timed | cluster | two-method intersection → one family cell |
| 3 | ASSISTments | native | |
| 4 | ASSISTments | cluster | two-method intersection → one family cell |
| 5 | XES3G5M | native | XES amendment |
| 6 | XES3G5M | cluster | one-method amendment; ARI vs text N/A |

### Primary claim decision rule

- **Dataset \(D\) satisfies the primary claim** iff both of \(D\)'s unit verdicts are `supported_*` **with opposite signs** (one `supported_A` and one `supported_B`).
- **Project-level claim** = **at least one** dataset satisfies the primary claim.
- Datasets that **can** satisfy it (all three, if both units eligible): **Junyi timed, ASSISTments, XES3G5M**.
- Slice-based order reversals outside the primary GRU−Markov unit tests are **not** in \(m\).

### Registered secondary analysis (common query set)

**Secondary (registered, not in \(m\)):** restrict to the **common query set** — queries that are non-repeat (revisit or advance) at **both** the native unit and the freeze primary cluster unit, aligned by (learner sequence, cut index). On that set, report GRU−Markov per unit with the **same** CI verdict rule and Bonferroni level as primary. Report **both** primary and secondary. A sign flip present **only** in the primary unit-specific non-repeat slices (not on the common set) is reported as a **primary-only flip**.

### Prior-formulation and pre-tag inspection disclosure

1. The primary H-rev claim (native vs cluster **sign flip** of GRU−Markov) was **formulated after observing** the XES native vs cluster flip in **validation** (`gru_markov_paired_ci.json`).
2. **Cluster used counts (115 / 106 / 120), ARI values (Junyi 0.038, ASSISTments 0.073), and the Junyi/ASSISTments method-agreement val table** (`dual_cluster_gru_markov.json`) were **inspected on validation before tagging**.
3. Confirmatory replication tests **held-out freeze-test learners** never scored in freeze (`test_touched=false`). Validation inspection does not replace freeze-test confirmatory analysis.

### Why XES native is allowed in the primary claim (amendment)

XES question-level order is curriculum-structured (mode successor ≈ 0.35) and is **descriptive** for recommender-style claims. For **primary H-rev only**, we **amend** the two-method clustering rule: XES has one local clustering method (co-occurrence), so that map is the registered cluster unit and XES native↔cluster stays inside the confirmatory family of 6. We do **not** claim a text/cooc method-agreement contrast for XES (N/A).

## Clustering-method rule (pinned; no post-hoc redefinition)

**Pinned used cluster counts:** Junyi **115**, ASSISTments **106**, XES **120** (freeze primary maps after dense remap). Target \(k=120\). Clustering seed **`20261021`**. Learner-split seed **`20261004`**.

**Junyi / ASSISTments:** “Both methods” means **exactly** text MiniLM k-means **and** train-only co-occurrence as frozen in `dual_clustering.json` / `cluster_maps/*_{text,cooc}_item_to_cluster.json`. Confirmatory scoring runs GRU−Markov under **both** maps. The **co-occurrence map uses the primary (text) cluster-unit frozen GRU config and `best_epoch` with no retuning** (registered). **Forbidden after results:** redefining “both methods,” swapping maps, changing \(k\), or dropping a disagreeing method.

Cluster-unit registered verdict:

| Method A | Method B | Registered cluster-unit |
| --- | --- | --- |
| same `supported_*` | same `supported_*` | that `supported_*` |
| `supported_*` | `negligible` | **inconclusive** (pinned tie rule) |
| `supported_*` | other / disagree | **inconclusive** |
| either inconclusive | — | **inconclusive** |

Partition ARI (descriptive): Junyi **0.038**, ASSISTments **0.073**. Co-occurrence clusters can inflate cluster-unit scores (train-transition leakage into labels).

**XES:** one method → amended as above; method-agreement N/A; ARI N/A.

Pre-tag dual-map val check (`dual_cluster_gru_markov.json`, freeze **val** only; not freeze test):

| Dataset | text | cooc | ARI | Registered cluster-unit (val) |
| --- | --- | --- | ---: | --- |
| Junyi timed | inconclusive | supported_A | 0.038 | **inconclusive** |
| ASSISTments | supported_A | supported_A | 0.073 | **supported_A** |
| XES3G5M | N/A | (one method) | N/A | confirmatory under amendment |

## H-rep

**No confirmatory H-rep test exists in this study.** Within-dataset repetition / share(next=last) contrasts are **exploratory only**. There is no cross-dataset confirmatory H-rep family and no Bonferroni \(m\) for H-rep.

## Applicability table

| Claim / analysis | Junyi timed | ASSISTments | XES3G5M | Scope |
| --- | --- | --- | --- | --- |
| Primary H-rev (native↔cluster opposite `supported_*`) | confirmatory | confirmatory | confirmatory (amended) | freeze test after tag |
| Secondary common-query GRU−Markov | confirmatory secondary | confirmatory secondary | confirmatory secondary | same CI rule/level; not in \(m\) |
| H-rep | exploratory only | exploratory only | exploratory only | **no confirmatory H-rep** |
| RAGR / DAG features | exploratory | N/A (no DAG) | N/A | Junyi |
| Phase 3 memory AUC | exploratory (ktbd) | N/A | N/A | Junyi |
| H-forget (day gap) | exploratory | N/A | N/A | Junyi timed |
| H-graph / readiness | exploratory | N/A | N/A | Junyi |
| paper_rank_reversal_full (seed 0) | EXPLORATORY | EXPLORATORY | EXPLORATORY | do not cite as confirmatory |

## Models

Required for H-rev tables: **popularity, recency, markov_order1, gru_probe**. Recency on advance = **N/A**. Recency on revisit = mechanical sticky (descriptive). Attn / gru4rec_style / RAGR excluded from preregistered reversal.

Replication: retrain on freeze **train+val** with frozen winner **config** and frozen **best_epoch** (fixed epoch count; **no early stopping** when scoring test); GRU over seeds **20261101 / 20261102 / 20261103** as above. No test-derived quantity enters training or selection (`tests/test_freeze_partition_replication.py`). Confirmatory bootstrap: **≥ 10,000** sequence-level resamples.

## Matched cluster counts (feasibility; pinned confirmatory = ~120 row)

| Target k | Junyi (835) | ASSISTments (150) | XES (7,439) |
| --- | --- | --- | --- |
| ~40 | ok | ok | ok |
| ~120 | **115 used** (pinned) | **106 used** (pinned; near-native) | **120 used** (pinned) |
| ~800 | ok near catalog | **N/A** | ok |

## Ranking table (Junyi ktbd; descriptive / exploratory)

Canonical capped RAGR R@5: `review_ranking.json` **0.891** [0.880, 0.902]. Full-test exploratory: `paper_ranking_full.json` **0.889** [0.885, 0.893]. CI verdicts: `paper_paired_verdicts.json` (Markov beats RAGR on advance; DAG negligible; Phase 3 inconclusive at \(t=0.005\)).

## Limitations

Confirmatory scope after `prereg-v1` is the **primary H-rev effect** (family of up to 6 GRU−Markov unit verdicts and the dataset/project decision rules above), plus the registered **secondary** common-query analysis. **Graph / DAG readiness and forgetting (H-forget) are exploratory and Junyi-only.** There is **no confirmatory H-rep**. The sticky / curriculum control is **first-order Markov only**; a stronger sequence baseline (higher-order Markov, neural sequence models beyond the registered GRU probe) could change conclusions about whether unit change reverses structure learning. Seed-0 rank-reversal, RecBole-faithful baselines, and co-occurrence-only claims without the agreement rule are out of confirmatory scope.

## Registration text (OSF / Zenodo)

Attach and register under the `prereg-v1` tag:

- `docs/PREREGISTRATION.md`
- `docs/PREREGISTRATION_REPLICATION.md`
- `docs/DEVIATIONS.md`

Those three documents are **part of the registration packet and covered by the git tag**. Also paste the SHA-256 digests below (canonical machine-readable copy: `outputs_junyi/phases/prereg_v1_locked_hashes.json`). `tests/test_freeze_partition_replication.py::test_prereg_locked_file_hashes` **fails if any locked file changes**.

State in the registration abstract that the **\(m = 6\) Bonferroni level remains valid** for Junyi/ASSISTments cluster cells because those cells use the **two-method intersection rule** (both methods must be the same `supported_*`), yielding one registered verdict per cell rather than two family members.

| File | SHA-256 |
| --- | --- |
| `outputs_junyi/phases/probe_freeze.json` | `28a8248a1664f0e3dcf2a833be3e20f8e2b94bd457d1f4073f3e45095e4ffaff` |
| `scripts/run_freeze_partition_replication.py` | `09b419a5aaaac1326965ef03826e04eaef3c05ecb4cb45070ab50167f6c394fe` |
| `outputs_junyi/phases/dual_clustering.json` | `6c98f9e301ea0566165b492bdc179d9b2444f62504cc2de548778b48a2517923` |
| `cluster_maps/junyi_timed_{text,item}_to_cluster.json` | `be9981bf6d7679bce273a4689937ef20d60d1257fc0897d1d7fd7179daf6c316` |
| `cluster_maps/junyi_timed_cooc_item_to_cluster.json` | `5dfc3e134cd5d64db4c7a5224a026be162a5db78054455c4fe617ccffa44b69e` |
| `cluster_maps/assistments_{text,item}_to_cluster.json` | `7676e8dbe9a90c309c3e2578526c81f85688ef1088dd6a8c7327b74596225327` |
| `cluster_maps/assistments_cooc_item_to_cluster.json` | `e82d2ba2217a792b4fe4ed3ce45b5c2fe5634d258c27934c9f02dccc69e6a159` |
| `cluster_maps/xes3g5m_{cooc,item}_to_cluster.json` | `097696754951b49fceaa4369ee45f4cd7adf6dae75ec6c8cc93ade21fa5f018d` |

## Pre-tag checklist (human)

Before `git tag prereg-v1` and OSF/Zenodo registration:

- [ ] Read this file end-to-end.
- [ ] Working tree clean (`git status` empty of unrelated dirty files you do not intend to tag).
- [ ] Tagged commit contains: `probe_freeze.json`, the three docs above, `DEVIATIONS.md`, `prereg_v1_locked_hashes.json`, replication/verdict scripts, cluster maps.
- [ ] `PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q` passes (includes locked-hash guard).
- [ ] Freeze test still `test_touched=false` in `probe_freeze.json`.
- [ ] OSF/Zenodo registration text includes the SHA-256 table (or full `prereg_v1_locked_hashes.json`) and attaches the three docs.
- [ ] After tag + registration only:  
      `PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test`
- [ ] After touch: regenerate `tab_hrev_confirmatory.tex` / `paper_main_numbers.json` from `freeze_partition_replication.json → primary_claims` **only** (never hand-edit claim cells).
- [ ] A second `--touch-test` is forbidden unless this file’s companion `DEVIATIONS.md` contains `TOUCH_TEST_OVERRIDE:` with justification.

## What this draft does not register as confirmatory

- Library-faithful SASRec / RecBole.
- Seed-0 `paper_rank_reversal_full.json`.
- Any H-rep test.
- H-forget / H-graph / RAGR practical wins.
- Untagged external timestamp (tagging + OSF/Zenodo is a separate human step).
