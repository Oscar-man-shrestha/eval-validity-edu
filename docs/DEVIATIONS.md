# Deviations log

Written **before** any `prereg-v1` tag. Do not treat Follow-up 3 or freeze numbers as preregistered replication.

### Canonical wording (shared with PREREGISTRATION.md / REPLICATION.md)

| Term | Registered wording |
| --- | --- |
| **Primary H-rev claim** | Native ↔ cluster **GRU−Markov sign flip**: dataset \(D\) satisfies it iff both unit verdicts are `supported_*` with **opposite signs**; project claim = ≥1 such dataset |
| **Family / Bonferroni** | \(m_{\mathrm{planned}}=6\); CI level \(1-0.05/m\); N/A cells reduce \(m\) |
| **GRU seeds** | `20261101`, `20261102`, `20261103` |
| **Two-method intersection** | Junyi/ASSIST: same `supported_*` on text and cooc → that verdict; else inconclusive (one family cell). XES: cooc-only amendment |
| **Touch-test lock** | Second `--touch-test` requires exact marker `TOUCH_TEST_OVERRIDE:` in this file |

Exploratory ranking, seed-0 rank-reversal, dual-map **val** checks, H-rep, graph, and forgetting are **not** part of the primary claim.

**XES amendment (primary H-rev only):** XES question-level order is curriculum-structured and remains **descriptive** for recommender-style claims. For confirmatory H-rev, the registered cluster unit is the single local co-occurrence map; native↔cluster stays in the family of 6. No text/cooc method-agreement contrast for XES (N/A).

**Re-touch:** do not insert `TOUCH_TEST_OVERRIDE:` unless a justified emergency re-score is required after the first locked `--touch-test`. The first touch still requires `prereg-v1` + OSF first.

## Probe freeze compute / grid

| Item | Planned / suggested | What we did | Reason |
| --- | --- | --- | --- |
| `max_epochs` | 20 | **10** | CPU wall-clock: multi-dataset × multi-config freeze |
| `early_stopping_patience` | 3 | **2** | Same |
| Mid-training R@5 | Log at epochs {2,5,10,20} | **CE every epoch; R@5 once** at early-stopped checkpoint | Large-catalog eval cost |
| Val R@5 subsample | Full val | Cap **400** sequences when `n_items≥1000` or `n_val>800` | XES ~7k questions |
| Grid (checkpoint 1) | lr×emb×dropout with emb up to 64 | 12 configs, emb∈{32,64,128} | Edge winners at emb=128 |
| Grid (checkpoint 2) | Extend once: **+emb 256**, **+lr 0.002** | 24 configs; winners accepted | Close emb/lr edges once |
| Selection rule (schema 4) | Max val non-repeat R@5 only | Among configs within **0.005** of best val non-repeat R@5, pick **lowest val CE** | Prefer simpler fit among near-ties |
| RAGR in freeze | Full DAG RAGR | **ragr_lite** on Junyi only (not a trained peer) | **Not** part of the equal-budget trainable grid (GRU/Attn only). Exploratory; excluded from H-rev |
| Markov baseline | — | **First-order item→item successor** (parameter-light; not on the 24-config grid) | Curriculum-order / sticky-transition control |

## Method family (pre-prereg decisions)

| Item | Decision | Reason |
| --- | --- | --- |
| Attention probe in **preregistered H-rev** | **Excluded** | Local 1-layer TransformerEncoder probe is **not** validated against SASRec / RecBole / authors’ code |
| H-rev ranking unit | **Primary:** matched concept clusters (~120). **Secondary:** native unit | Cross-dataset native contrasts are **descriptive** if not cluster-matched |
| Clustering | **Both** text MiniLM k-means and train-only co-occurrence on Junyi + ASSISTments; XES text **unavailable locally** → co-occurrence only | Contrast counts only if both methods agree (Junyi ARI=0.038; ASSIST ARI=0.073 — they do not agree) |
| Freeze tuning | Native + cluster both frozen (schema 4); GRU grid on both units | Cluster is primary confirmatory unit |
| Minimum models for a **reversal claim** | popularity, recency, markov_order1, gru_probe on the same split/sample | Attn / RAGR-lite optional exploratory only |
| Cross-dataset native H-rep gaps | Descriptive until cluster-matched H-rep | Granularity-matched test is the informative confirmatory contrast |

## Data / reporting

| Item | Note |
| --- | --- |
| ASSISTments `n_items=150` | Composite skill tokens; **123** atomic `skill_id≥0`; **skill_name present** → text clustering |
| ASSISTments truncation | `truncate_each_learner_to_200` drops **121,660** of **339,305** rows after `len≥12` |
| XES question-level | **Curriculum-order-structured** (mode successor ≈**0.35**); native-unit results **descriptive** |
| Licences | ASSISTments + XES3G5M texts **UNVERIFIED** — release blocker |
| Ranking 0.891 [0.879, 0.905] | **Superseded.** Canonical is `review_ranking.json` RAGR R@5 0.891 [0.880, 0.902] (sequence bootstrap, n_boot=1000) |

## Canonical freeze entrypoint

- Schema 4 freeze (native + cluster + Markov + CE-within-0.005 selection): `PYTHONPATH=. python3 scripts/run_preprereg_freeze_v4.py` (sets `torch.set_num_threads(1)` to avoid OpenMP deadlocks).
- `run_probe_freeze.py` remains the older native-only trainer; do not treat it as schema 4 source of truth.

## Phase 3 test protocol (pre-prereg, 2026-10-06)

| Item | Fact |
| --- | --- |
| Task | **Correctness on revisit attempts only.** A memory row exists only when a concept is attempted again in the same sequence. The first attempt emits no row. |
| Headline 0.877 | **Yes — it used the first 80,000 `test.json` memory rows in file order.** Recomputed on that same slice: logistic_step_dt AUC **0.87687**, correct rate 0.328, CM TN/FP/FN/TP = 49,189 / 4,606 / 10,317 / 15,888. Superseded. |
| Replacement | All **449,453** test memory rows (6,288 sequences with a revisit; correct rate **0.446**). Sequence bootstrap, 400 resamples. logistic_step_dt AUC **0.864** [0.861, 0.867], PR-AUC **0.826** [0.820, 0.831]. CM @0.5: TN 209,734 / FP 39,385 / FN 56,994 / TP 143,340. Success-rate AUC 0.858 [0.855, 0.862]; paired Δ (logistic − success) **0.0053** [0.0046, 0.0059]. HLR 0.789 [0.784, 0.794]. |
| Day vs step | ktbd JSON has no timestamps, so `delta_days` is copied from `delta_steps`. logistic_day_dt matches logistic_step_dt; logistic_both differs only past 1e-9. Do not report them as three results. |
| Figure 7c | Title/caption name those rows (all test.json revisit memory rows), not the 80k file-order slice. |

## Actual cluster counts and matched-k feasibility

Freeze primary maps, unique ids after train+val map and dense remap (`probe_freeze.json` `cluster_unit.n_clusters_used`, confirmed in `dual_clustering.json`):

| Dataset | Native items | Target k | Used clusters |
| --- | --- | --- | --- |
| Junyi timed | 835 | 120 | **115** |
| ASSISTments | 150 composite tokens | 120 | **106** |
| XES3G5M | 7,439 questions | 120 | **120** |

Feasible matched-count table (infeasible cells are **N/A**; this table is the source for the preregistration, which is **not written yet**):

| Target k | Junyi (835) | ASSISTments (150) | XES (7,439) |
| --- | --- | --- | --- |
| ~40 | ok | ok | ok |
| ~120 | ok | **ok, near-native** (106 used) | ok |
| ~800 | **ok, near catalog** (800 of 835) | **N/A** (k ≥ n_items) | ok |

## Clustering methods (both run)

| Dataset | Text MiniLM | Train-only co-occurrence | Partition agreement |
| --- | --- | --- | --- |
| Junyi timed | yes (freeze primary) | yes | ARI **0.038**, NMI 0.598 (n=835) |
| ASSISTments | yes, `skill_name` (freeze primary) | yes | ARI **0.073**, NMI 0.901 (n=150) |
| XES3G5M | **N/A** — local `question_level` CSVs have ids / KC tags only, no stem text (`dual_clustering.json` → `xes_question_text`) | yes (freeze primary) | **N/A** (no second method) |

Co-occurrence used clusters after dense remap: Junyi 116, ASSISTments 103, XES 120. Maps under `outputs_junyi/phases/cluster_maps/`.

**Rule to put in the preregistration (not written in this pass):** a clustering-method contrast counts only if both methods are defined on that dataset **and** they agree on the confirmatory claim (same sign / same ranking conclusion). XES method-agreement contrasts are N/A. Low ARI means the two partitions are not the same groups; agreement has to be checked on the claim, and is not implied by running both clusterings.

## Markov vs GRU (H-rev)

`markov_order1` is in the freeze table at **native and cluster** for all three datasets (parameter-light; not on the 24-config grid). Point estimates, val non-repeat R@5, from `probe_freeze.json`:

| Dataset | Markov native | GRU native | Markov cluster | GRU cluster |
| --- | --- | --- | --- | --- |
| Junyi timed | 0.430 | 0.445 | 0.471 | 0.515 |
| ASSISTments | 0.568 | 0.750 | 0.594 | 0.771 |
| XES3G5M | **0.739** | **0.614** | 0.636 | 0.761 |

XES native freeze check: 7,439-way catalog; winner `lr=0.0005`, `emb=256`, `max_epochs=10`, patience 2; **best epoch 4**; val CE 3.865 → 3.910 at epoch 5 → 3.970 at epoch 6, then stop. `GRUProbe` has **no dropout module** — the grid's dropout value does not change the GRU.

Paired sequence bootstrap (retrain of the frozen config, same seed; GRU and Markov scored on one shared val subsample; n_boot=400). Absolute R@5 here is **not** a replacement of the freeze point estimates: freeze R@5 uses a different eval subsample seed. The paired file is `outputs_junyi/phases/gru_markov_paired_ci.json`.

| Dataset | Unit | GRU R@5 | Markov R@5 | Δ (GRU−Markov) | 95% CI | Call |
| --- | --- | --- | --- | --- | --- | --- |
| Junyi timed | native | 0.415 | 0.408 | +0.007 | [−0.023, +0.039] | tie (CI includes 0) |
| Junyi timed | cluster (115) | 0.514 | 0.480 | +0.034 | [+0.002, +0.066] | GRU |
| ASSISTments | native | 0.725 | 0.568 | +0.157 | [+0.117, +0.196] | GRU |
| ASSISTments | cluster (106) | 0.764 | 0.594 | +0.170 | [+0.127, +0.217] | GRU |
| XES3G5M | native | 0.570 | 0.718 | **−0.148** | **[−0.170, −0.128]** | **Markov** |
| XES3G5M | cluster (120) | 0.765 | 0.618 | +0.147 | [+0.114, +0.181] | GRU |

Longer XES-native diagnostic (`max_epochs=30`, patience 5): still **best epoch 4**, val CE 3.884, stopped at epoch 9 as CE rose (3.929, 3.975, 4.034, 4.118, 4.169). Paired non-repeat R@5 GRU 0.575 vs Markov 0.718, Δ **−0.143** [−0.163, −0.123]. More epochs do not close the gap. Undertraining at `max_epochs=10` is not a sufficient explanation.

**What a probe losing to Markov means for H-rev.** H-rev asks whether a probe's rank versus the required baselines (popularity, recency, Markov, GRU) reverses when the unit changes. Markov is a 1-step successor table. If GRU loses to Markov, or the paired CI includes zero, a win against popularity or recency at that unit is not evidence of structure beyond sticky or curriculum order. A confirmatory H-rev claim there cannot treat GRU as beating the required set. Native XES (mode successor ≈ 0.35; freeze GRU 0.614 < Markov 0.739; paired CI entirely below zero, including after the longer run) stays **descriptive**. Junyi native is a tie under the paired CI. The cluster unit is where GRU's paired CI is above zero on all three datasets.

## PDF reporting (pre-prereg)

- §4.4 CIs are the sequence bootstrap already stored in `review_ranking.json` (n_boot=1000). RAGR R@5 = 0.891 [0.880, 0.902].
- The older Phase 6 print 0.891 [0.879, 0.905] (`phase6_eval.json`) is the same point under a different bootstrap draw and is labeled **superseded**.
- Freeze tables include validation non-repeat query counts.
- §4.9 points at confirmatory numbers in §4.11.

## Not done yet (explicit)

- External registration and the git tag `prereg-v1`. The draft text is `docs/PREREGISTRATION.md` (N/A matched-k cells and the both-methods-must-agree rule are in that file). It is not timestamped.
- Library-faithful SASRec / RecBole baseline. The attention probe stays out of H-rev until that exists.
- Ranking the probes on the second clustering (co-occurrence for Junyi and ASSISTments). Partitions disagree (ARI 0.038 and 0.073); that contrast is not claimed.

## Paper eval-validity pass (journal prep)

Generated by `scripts/run_paper_eval.py` + `scripts/build_paper_assets.py`.
No number was changed to look better; capped vs full are both reported.

| Item | Old | New | Note |
| --- | --- | --- | --- |
| Phase 6 RAGR CI | [0.879, 0.905] (`phase6_eval.json`) | [0.880, 0.902] (`review_ranking.json`) | Canonical capped remains review_ranking |
| RAGR R@5 full test | (not previously reported) | 0.889 [0.885, 0.893] (n=6290) | `paper_ranking_full.json` |
| phase1 `interactions` field | 2,842,883 (train+test, unlabeled) | train-only 2,342,784; `interactions_train_plus_test`=2,842,883; topic n=2,809,159 | `paper_junyi_counts.json` |
| XES method B | was identical to A (first-KC) | expand-all share=0.2620 vs 0.2162 | `paper_xes_kc_methods.json`; old B deprecated |
| RecBole GRU4Rec/SASRec | not installed | local `gru4rec_style` / `attn_probe` robustness | See `paper_rank_reversal_full.json` note |
| Fig embedding pair-AUC title | recompute could print 0.905 | loads `phase2_embeddings.pair_auc_vs_random`=0.904 | `generate_report_assets.py` |
| Ranking figure source | `phase6_eval.json` | `review_ranking.json` CI [0.880, 0.902] | `generate_report_assets.fig_ranking` |
<!-- end paper eval-validity -->
