# Data card — Junyi raw timed extract

| Field | Value |
| --- | --- |
| Role | Learner-disjoint day-gap / timed ranking (exploratory) |
| Source | DataShop Junyi dump → local `junyi.rar` → `timed_interactions.npz` |
| Licence | Non-commercial (Chang et al., EDM 2015 / DataShop 1198); excerpt in `licences/junyi_noncommercial_excerpt.md` |
| Local path | `data/junyi_raw/timed_interactions.npz` |
| Companion metadata | `data/junyi_raw/junyi/junyi_Exercise_table.csv` (topic/area) |

## SHA-256 (local files)

| File | SHA-256 |
| --- | --- |
| timed_interactions.npz | `8bf6db6636b18631b5cea439c5b44a608431cf0a1b4876d05b4fcef71e006164` |
| junyi_Exercise_table.csv | `13292cc2c809985abbacedaba434e39aeb02a7fcb2a32f17b60c4d3f4d769546` |

## Schema (`timed_interactions.npz`)

| Array | Meaning |
| --- | --- |
| user_id | learner id |
| concept | exercise / concept id aligned to ktbd space |
| t_us | timestamp (microseconds) |
| correct | 0/1 |

## Counts

| Asset | Count |
| --- | --- |
| Rows (capped extract) | 8,000,000 |
| Unique users in extract | 204,077 |

## How the 6,000 freeze learners are selected

From `probe_freeze.json` → `datasets.junyi_timed.learner_selection` (must match code in `build_learner_sequences_junyi_timed`):

- Stream users in **file order** through the capped 8M-row extract.
- Keep a user when their contiguous block length ≥ 12.
- Take the **first 6,000** such users (cap), truncate each sequence to length ≤ 200.
- **Not** a uniform random sample of the 204k users.

## Timestamps

**Yes — real `t_us`.** Used for exploratory calendar-day gap features.

## Ranking unit (H-rev)

| Role | Unit |
| --- | --- |
| Primary (freeze schema 4) | text MiniLM clusters; **115 used** after dense remap (target 120) |
| Second method | train-only co-occurrence; **116 used**; ARI vs text **0.038** |
| Freeze secondary / native | exercise |
| Cross-dataset native contrasts | **descriptive** until cluster-matched |
| Matched k | ~40 ok; ~120 ok; ~800 feasible but near the 835-item catalog |

Attention probe: documented in `probe_freeze.json` but **excluded** from preregistered H-rev. Markov order-1 successor = **parameter-light** baseline (not on the 24-config grid). Selection: among configs within 0.005 of best val non-repeat R@5, lowest val CE.

Paired retrain (`gru_markov_paired_ci.json`): native GRU − Markov Δ **+0.007** [−0.023, +0.039] (tie). Cluster Δ **+0.034** [+0.002, +0.066] (GRU). Freeze point estimates stay in `probe_freeze.json`; this CI is a separate subsample.

## Applicability

- H-forget / learner-disjoint splits: applicable (exploratory)
- Ranking unit for freeze H-rev primary: **~120 clusters**; native exercise = secondary
