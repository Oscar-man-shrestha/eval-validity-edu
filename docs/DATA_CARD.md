# Junyi / ktbd-junyi data card (NeuroTrace-DAG)

## Identity

| Field | Value |
| --- | --- |
| Corpus | Junyi Academy Math Practicing Log |
| Citation | Chang, Lin, Chen, EDM 2015 |
| Split we use | EduData `ktbd-junyi` (`train.json` / `test.json`) |
| Mirror | http://base.ustc.edu.cn/data/ktbd/junyi/ |
| DataShop | dataset 1198 |

## Counts (after our length filter 12–200)

| Asset | Count |
| --- | --- |
| Exercises (`vertex_id2idx`) | **835** |
| Prerequisite edges (`prerequisite.json`, unique) | **978** |
| Similarity pairs | 1,954 |
| Train sequences | 27,434 (`train.json`, len∈[12,200]) |
| Test sequences | 6,290 (`test.json`, len∈[12,200]) |
| Train interactions | **2,342,784** (`phase1_prepare.json` → `interactions_train`) |
| Test interactions | **500,099** (`interactions_test`) |
| Train+test interactions | **2,842,883** |
| Topic-level n (next==last denom.) | **2,809,159** transitions = train 2,315,350 + test 493,809 (`paper_junyi_counts.json`) |
| Attempt accuracy (4k-seq sample) | ~0.51 |

**Topic n vs 2,842,883:** 2,842,883 is train+test attempt rows after the length filter. Topic-level share uses consecutive transitions on **train+test**, so \(n = 2{,}842{,}883 - 27{,}434 - 6{,}290 = 2{,}809{,}159\). The 6,290 gap vs \(2{,}842{,}883 - 27{,}434\) is exactly the test sequence count. See `outputs_junyi/phases/paper_junyi_counts.json`.

**835 vs ~722:** older papers sometimes cite ~722 Junyi exercises from a different extract. We follow EduData ktbd’s 0…834 id space (same tooling line as RCD / KnowLP).

**Subsample:** ktbd sequences are a processed extract of the full DataShop dump, not the entire raw clickstream. Timed analysis uses a capped stream from `junyi.rar` → `timed_interactions.npz` (8M rows).

## Schema

- `train.json` / `test.json`: one JSON list per line `[[item_id, correct], …]`. **No learner id field.**
- Learner-disjoint experiments use `timed_interactions.npz` (`user_id`, exercise name, correct, `time_done`).
- Published ktbd sequence split cannot be proven learner-disjoint from the JSON alone.

## Action mix (train prefixes; see `phase6_review_fixes.json`)

Three-way labels:

| Action | Meaning |
| --- | --- |
| **continue** | next item == last item |
| **revisit** | seen earlier, but not the last item |
| **advance** | never seen in the prefix |

Most “review” mass is **continue**, not spaced restudy. Forgetting tests belong on **revisit** (+ timed gap bins).

## What not to claim from this dump

- That overall Recall@K measures curriculum progress (it mostly measures continue).
- That text embeddings recover prerequisite direction (same-topic control fails).
- Learner-disjointness of the published JSON split without the raw user map.
