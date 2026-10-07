# Data card — XES3G5M

| Field | Value |
| --- | --- |
| Role | Pre-registered replication corpus |
| Source | XES3G5M (NeurIPS 2023) |
| Local paths | `data/xes3g5m/XES3G5M.zip`, `data/xes3g5m/XES3G5M/`, `data/xes3g5m/hf/train.parquet` |
| Licence | **UNVERIFIED** — see `docs/data_cards/licences/xes3g5m_UNVERIFIED.md` (**release blocker**) |

## SHA-256 (local files)

| File | SHA-256 |
| --- | --- |
| XES3G5M.zip | `62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73` |
| hf/train.parquet | `0be9cb7fed2951ac28ecc0410df3fd075489cdddcafeb3f809fcd4cba817485d` |

## Ranking unit per hypothesis

| Hypothesis | Unit | Notes |
| --- | --- | --- |
| H-rep (informative) | co-occurrence clusters; **120 used** (target 120). **No question text locally** — text clustering is N/A | Native question/KC = **descriptive** |
| H-rev primary | matched clusters (120 used) | Secondary native = **question** (**curriculum-order-structured**). Method-agreement contrasts are **N/A** (only one clustering method). |
| H-rev freeze | both native + cluster in `probe_freeze.json` schema 4 | Native marked descriptive |
| H-graph | **N/A** | No DAG |
| H-forget | N/A until timestamp→day validation in JSON | `timestamps` column exists |

Minimum slice size: **500 queries**. Below that → N/A.

Matched-k feasibility: ~40 ok; ~120 ok (120 used); ~800 ok. Text method is N/A so clustering-method contrasts are N/A.

## Why question-level results are descriptive (curriculum order)

From freeze schema 4 (`probe_freeze.json`):

- Native GRU val non-repeat R@5 ≈ **0.614** (CE ≈ 3.865; best_epoch=4 / max_epochs=10 / patience=2; 7,439-way softmax; `GRUProbe` ignores dropout)
- Native **Markov order-1** val non-repeat R@5 ≈ **0.739** (beats GRU — quiz-order control). A GRU loss to Markov means this unit cannot claim a learned H-rev win
- Paired retrain CI (same config, shared subsample): native Δ GRU−Markov **−0.148** [−0.170, −0.128]
- Longer run (max 30, patience 5) still **best epoch 4**, CE 3.884, and still loses: Δ **−0.143** [−0.163, −0.123]. Not an undertraining artifact
- Cluster GRU (120 used) val non-repeat R@5 ≈ **0.761** vs cluster Markov **0.636**. Paired retrain Δ **+0.147** [+0.114, +0.181]

Question-level `next==last` is ~**0.008**. From `question_order_structure`:

- Share next is **empirical mode successor** of previous ≈ **0.35**
- Share next is **global question-order successor** ≈ **0.196**

Mark question-level H-rev as **curriculum-order-structured** and **descriptive**. Prefer cluster-matched primary unit for confirmatory claims.

## KC method difference (0.215 vs ~0.262)

| Method | Rule |
| --- | --- |
| F2 expand-all | expand every KC tag → ~0.215 |
| F3 Method A/B | one primary KC per question → ~0.262 |

Parent KC routes: **N/A** in this dump.

## Counts (local)

| Asset | Count |
| --- | --- |
| Question-level CSV rows (train_valid + test) | 34,578 |
| Unique uids | 18,066 |

## Applicability

- H-rep / H-rev on cluster-matched primary unit: confirmatory candidate
- Native question H-rev: **descriptive** (curriculum-order-structured)
- RAGR / H-graph: **N/A**
- Attention probe: **excluded** from preregistered H-rev
