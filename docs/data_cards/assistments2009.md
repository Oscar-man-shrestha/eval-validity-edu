# Data card — ASSISTments 2009–2010 skill-builder (corrected)

| Field | Value |
| --- | --- |
| Role | Pre-registered replication corpus (skill-level) |
| Source | EduData USTC mirror of ASSISTments 2009–2010 skill-builder **corrected** |
| Local path | `data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv` |
| Licence | **UNVERIFIED** — see `docs/data_cards/licences/assistments_UNVERIFIED.md` (**release blocker**) |

## SHA-256 (local file)

| File | SHA-256 |
| --- | --- |
| skill_builder_data_corrected.csv | `1aa296e00b6c88c4d6fad4ca2ae4866484d9fe5484f38f5c8c94dfc49f045e08` |

## Skill-id construction (ranking unit)

Rule (also in `probe_freeze.json` → `assistments_filter_funnel.skill_construction`):

1. Drop rows missing `user_id` / `order_id` / `correct`.
2. `groupby order_id`; `skills = sorted(unique skill_id)` with `skill_id >= 0`.
3. **Composite token** = `'_'.join(skills)` (freeze / H-rev ranking unit).
4. **Primary atomic skill** = first element of that sorted tuple (sensitivity).

### Why `n_items=150` vs ~123 atomic skills

There are **123** distinct atomic `skill_id >= 0` values in the CSV. After multi-skill collapse there are **150** distinct **composite** tokens because many `order_id`s carry multiple skills (joined ids). Freeze `n_items` counts composite tokens, not atomic skills.

From `probe_freeze.json` (freeze redo):

| Quantity | Value |
| --- | --- |
| Atomic `skill_id≥0` | 123 |
| Composite tokens (all collapsed rows) | 150 |
| Multi-skill / single / empty skill rows | 47,037 / 236,068 / 63,755 |
| next==last composite | 0.679 (n=214,725) |
| next==last primary atomic | 0.696 (n=214,725) |

## Timestamps

`ms_first_response` / `overlap_time` are durations, not absolute clocks → **H-forget = N/A**.

## Filter funnel

See `probe_freeze.json` → `assistments_filter_funnel.steps`. **4217 → 2920** is `len >= 12` (+ truncate to 200), **not** skill collapse.

| Step | n_rows |
| --- | --- |
| After `len ≥ 12` | 339,305 |
| After truncate to 200 | 217,645 |
| Rows dropped by truncation | **121,660** |

## Ranking unit (H-rev)

| Role | Unit |
| --- | --- |
| Primary | text MiniLM on `skill_name`; **106 used** (target 120) — **near-native** (150 composite tokens) |
| Second method | train-only co-occurrence; **103 used**; ARI vs text **0.073** |
| Freeze secondary / native | composite skill token |
| Cross-dataset native shares | **descriptive** until cluster-matched |
| Matched k | ~40 ok; ~120 ok but near-native; **~800 = N/A** (k ≥ 150 items) |

Attention probe: **excluded** from preregistered H-rev. Markov order-1 = parameter-light. Selection: R@5 then CE within 0.005.

## Applicability

- H-rep / H-rev (pop, recency, Markov, GRU) on cluster primary unit after freeze
- Report repetition: composite **0.679** and primary atomic **0.696**
- RAGR / H-graph / H-forget: **N/A**
