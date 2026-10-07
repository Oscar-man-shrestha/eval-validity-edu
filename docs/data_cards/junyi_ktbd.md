# Data card — Junyi / EduData ktbd-junyi

| Field | Value |
| --- | --- |
| Role | Primary exploratory corpus (sequences + expert DAG) |
| Source | EduData / USTC ktbd-junyi mirror of Junyi Academy Math Practicing Log |
| Upstream | DataShop dataset 1198; Chang, Hsu & Chen, EDM 2015 |
| URL | http://base.ustc.edu.cn/data/ktbd/junyi/ |
| Local path | `data/junyi_ktbd/` |
| Licence | Non-commercial use only — excerpt in `licences/junyi_noncommercial_excerpt.md` |
| Retrieval note | Files present in workspace; re-download via `scripts/fetch_datasets.py` |

## SHA-256 (local files)

| File | SHA-256 |
| --- | --- |
| train.json | `9a50d9bdbfbc19cd4682cc0303fb0ff5bdb017a955c10d6f1922a60636d494c1` |
| test.json | `c55f0714d9479d35afb25564ef2faf57c17ca46d944a0e509cb3f26780934973` |
| vertex_id2idx | `553271b2804315b879adcd04e7696db61477b0b53fd9033be3e1413136bf4c2c` |
| prerequisite.json | `cd3eaf22c22ebbeb3165068233b7fe16962d1910573f5546628f7612cef435b0` |

## Schema

- `train.json` / `test.json`: one JSON list per line `[[item_id, correct], …]`. **No learner id field.**
- `vertex_id2idx`: English slug → integer id 0…834
- `prerequisite.json`: directed prerequisite pairs
- `similarity.json`: official similarity pairs

## Counts (raw file lines / nodes)

| Asset | Count |
| --- | --- |
| Exercises (vertex_id2idx) | 835 |
| prerequisite.json entries | 985 (unique edges after Phase 1: 978) |
| train.json lines (unfiltered) | 47,833 |
| test.json lines (unfiltered) | 11,959 |
| After length filter 12–200 (Phase 1 JSON) | train 27,434 / test 6,290 |
| Train interactions (attempt rows) | **2,342,784** (`phase1_prepare.json` → `interactions_train`; split=`train.json`) |
| Test interactions | **500,099** (`interactions_test`; split=`test.json`) |
| Train+test interactions | **2,842,883** (`interactions_train`+`interactions_test`) |
| Topic-level n (consec. transitions) | **2,809,159** = train transitions 2,315,350 + test transitions 493,809 (`paper_junyi_counts.json`; split=`train.json`+`test.json`) |
| Identity check | 2,842,883 − 27,434 − 6,290 = 2,809,159 (interactions − sequences = transitions) |

## Timestamps

**No real timestamps** in ktbd JSON. Day-gap analyses use the separate timed extract (see `junyi_timed.md`).

## Known problems

- Platform suggested exercises via the knowledge map (selection bias).
- ktbd has no learner ids → learner-disjointness of the published JSON split is **untestable**.
- Older papers sometimes cite ~722 exercises; we follow EduData 0…834 (835 nodes).
- Published train/test may contain exact-duplicate sequences (chance-matched in Follow-up 3).

## Phase 3 label (ktbd sequences)

The Phase 3 task is **correctness on revisit attempts only**. A memory row is emitted only when a concept is seen again in the same sequence; the first attempt emits no row. Attempt-level accuracy ≈ 0.51 (4k-sequence sample, every attempt) is a different quantity.

The headline AUC **0.877** used the **first 80,000** `test.json` memory rows in file order (recomputed AUC **0.87687**, correct rate 0.328). That figure is superseded. On all **449,453** test memory rows (correct rate 0.446), logistic_step_dt AUC is **0.864** [0.861, 0.867] (sequence bootstrap, n_boot=400), PR-AUC **0.826**. Confusion @0.5: TN 209,734, FP 39,385, FN 56,994, TP 143,340. See `outputs_junyi/phases/phase3_memory_fulltest.json`.

## Ranking unit (H-rev)

Native unit = **exercise**. Primary confirmatory unit = matched concept clusters. Freeze text MiniLM at target 120 leaves **115** used clusters after dense remap. Train-only co-occurrence was also fit (116 used); partition ARI vs text = **0.038**. A method contrast counts only if both methods agree on the claim. Cross-dataset native contrasts are descriptive until then. Minimum reversal-claim models: popularity, recency, Markov order-1, GRU probe.

## Applicability (study)

- H-rep / H-rev: exploratory (Junyi-involving); GRU in prereg H-rev; attention excluded; Markov = parameter-light
- H-graph / H-forget: Junyi exploratory only (forget needs timed extract)
- RAGR with DAG features: Junyi only
