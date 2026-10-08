# NeuroTrace-DAG — Dataset, Input Flow, and Actual Results

**Purpose of this document:** a single, readable account of what data we use, how it moves phase-by-phase through the real implementation, and what outputs we actually produced. Numbers are taken from `outputs_junyi/phases/*.json` (including `probe_freeze.json` schema v3/v4 and paper_* eval JSONs).

**Companion PDF (figures embedded):** [`NeuroTrace-DAG_Data_Flow_Report.pdf`](NeuroTrace-DAG_Data_Flow_Report.pdf) — rebuild with `python3 scripts/build_dataflow_report.py --mode=full`.

**Word copy for advisors:** [`NeuroTrace-DAG_Dataset_Input_Flow_Results.docx`](NeuroTrace-DAG_Dataset_Input_Flow_Results.docx) (generated from this markdown).

**Study framing:** we care whether next-item scores track continue / revisit / advance structure (and action units)—not whether we shipped a ranking champion.

**Code:** https://github.com/Oscar-man-shrestha/eval-validity-edu  
**OSF:** https://osf.io/24sxf (submitted; pending approval)  
**Full doc map:** [`DOCUMENTATION.md`](DOCUMENTATION.md)

### Checklist map (required sections → this document)

| Required topic | Where answered |
| --- | --- |
| 1. Dataset / Input (what, where from, contents, purpose, where used, multi-dataset roles) | **§1** |
| 2. Input at each phase (input → processing → output → next input) | **§2** |
| 3. Implementation plan (dataset travel end-to-end) | **§3** |
| 4. Output / results (product output, samples, tables, files — not accuracy alone) | **§4** |
| 5. Visualizations (distributions, mix, ranking, DAG, CM, …) | **§5** |
| 6. ROC / AUC (what classified, inputs to the curve, AUC values) | **§6** |
| 7. Complete Dataset → … → Final output flow | **§7** |

**Confirmatory note:** OSF is submitted ([osf.io/24sxf](https://osf.io/24sxf)); freeze-holdout cells stay blank until one `--touch-test` after approval. Numbers below are real measured outputs, but exploratory / diagnostic unless labeled otherwise.

---

## 1. Dataset / Input

### 1.1 What datasets are we using?

| Dataset | Role |
| --- | --- |
| **Junyi / EduData ktbd-junyi** | Primary corpus for Phases 1–6, demo, DAG, embeddings |
| **Junyi timed extract** (`timed_interactions.npz`) | Learner-disjoint day-gap forgetting + timed ranking / freeze |
| **ASSISTments 2009 skill-builder corrected** | Cross-dataset skill-level repetition + ranking probes |
| **XES3G5M** | Cross-dataset question/KC repetition + ranking probes |

OULAD × MOOCCubeX was **rejected** (no shared concept ids; ~96.7% pass rate unusable for memory ROC).

### 1.2 Where did each dataset come from?

| Dataset | Source | Local path |
| --- | --- | --- |
| Junyi ktbd | EduData USTC mirror of DataShop #1198 (Chang et al., EDM 2015) — http://base.ustc.edu.cn/data/ktbd/junyi/ | `data/junyi_ktbd/` |
| Junyi timed | DataShop dump → `junyi.rar` → capped stream | `data/junyi_raw/timed_interactions.npz` |
| ASSISTments 2009 corrected | EduData USTC skill-builder corrected | `data/assistments2009/.../skill_builder_data_corrected.csv` |
| XES3G5M | NeurIPS 2023 release | `data/xes3g5m/` |

Per-file SHA-256 and licence notes: `docs/data_cards/`. ASSISTments/XES licence text is **UNVERIFIED** (release blocker).

### 1.3 What does the primary dataset contain?

| File | Features / schema | Counts we use |
| --- | --- | --- |
| `vertex_id2idx` | English exercise slug → integer id 0…834 | **835** concepts |
| `prerequisite.json` | Directed prerequisite edges | **978** unique edges |
| `similarity.json` | Official similarity pairs | 1,954 pairs |
| `train.json` | One sequence per line: `[[item_id, correct], …]` | **27,434** seq (len 12–200) |
| `test.json` | Same schema (published hold-out) | **6,290** sequences |
| Interactions (train) | Sum of filtered train.json lengths | **2,342,784** attempts |
| Interactions (train+test) | train.json + test.json after len filter | **2,842,883** attempts |
| Topic-level n | train+test consecutive transitions | **2,809,159** (= 2,842,883 − 27,434 − 6,290) |

**Targets / labels used in the system**

- **Memory (classification):** correctness on **revisit attempts only** — will the next attempt on a concept already seen in the sequence be correct? (0/1). The first attempt on a concept emits no memory row.
- **Gate (classification):** will the true next item be a *review* (seen) vs *advance* (unseen)?
- **Ranking:** which exercise id is next? (Recall@K / MRR)

Join rule: `native: every log item id is a DAG node id` — every log item id is a DAG node id.

### 1.4 Purpose — why these datasets?

We study whether aggregate next-item metrics mostly reward **continue / revisit / advance** structure (and how that changes with concept granularity). Junyi is the only public dump with learner traces **and** an expert prerequisite DAG in one id space. ASSISTments and XES3G5M check whether sticky `next==last` is platform- or granularity-specific. Timed Junyi supports exploratory day-gap (H-forget) checks.

### 1.5 Where each dataset is used in the implementation

| Dataset | Fit / train | Validate | Test / report |
| --- | --- | --- | --- |
| Junyi `train.json` | Memory, gate, review/advance heads, popularity | Within-train sequence holdout | — |
| Junyi `test.json` | Never fitted | — | Phase 6 R@K; gate AUC; memory headline AUC; exact-dup check |
| Embeddings + DAG + similarity | Features for Phase 5 | — | Demo subgraph |
| Timed Junyi | Learner-split forgetting / freeze train | Freeze val | Follow-up rank-reversal |
| ASSISTments | Freeze train (composite skill) | Freeze val | Follow-up repetition + rank-reversal |
| XES3G5M | Freeze train (question) | Freeze val | KC repetition methods + rank-reversal |

ASSISTments (follow-up): 401,756 raw rows → 346,860 attempts; 2,920 learners (train 2,336 / test 584).

XES3G5M: 18,066 uids (train 14,453 / test 3,613). Question-level next==last ≈ 0.008.

### 1.6 Why multiple datasets?

- **Junyi ktbd** — build and evaluate the full Review–Advance system + DAG.
- **Timed Junyi** — real calendar gaps; learner-disjoint splits the published JSON cannot prove.
- **ASSISTments / XES** — external checks that repetition and ranking *order* are not Junyi-only artifacts.
- They do **not** retrain the shipped Junyi `serve_models.joblib`.

---

## 2. Input at Each Phase

Exact phases from `run_all_phases.py` / `run_valid_eval.py` / follow-ups / freeze — not a generic ML template.

### Phase 1 — Prepare

| | |
| --- | --- |
| **Input (where from)** | Raw files in `data/junyi_ktbd/` |
| **Processing** | Length filter 12–200; write English concept table + edge list; confirm native id join; close OULAD faults |
| **Output** | `concepts_junyi.csv` (835), `prerequisite_edges_junyi.csv` (978), `phase1_prepare.json` — 27,434/6,290 train/test sequences; train interactions 2,342,784; train+test 2,842,883; topic n 2,809,159 |
| **Becomes next input** | Feeds Phase 2 names and Phase 3–6 sequences |

### Phase 2 — Embeddings

| | |
| --- | --- |
| **Input (where from)** | 835 English names from Phase 1 |
| **Processing** | Encode with `all-MiniLM-L6-v2`, L2-normalise; controls = random / same-topic / lexical Jaccard |
| **Output** | `concept_embeddings.npy` (835×384). Linked cosine 0.546 vs random 0.167; pair-AUC vs random 0.904; vs same-topic 0.464 |
| **Becomes next input** | Query = mean of last-5 embeddings for Phase 5 heads |

### Phase 3 — Memory (classification)

| | |
| --- | --- |
| **Input (where from)** | `train.json` (sequence-id split, leakage=0); eval on official `test.json` **revisit** rows only; timed extract is not in ktbd JSON |
| **Processing** | Fit mean / success_rate / HLR / logistic(step). Sequence-level bootstrap on all test memory rows |
| **Output** | All 449,453 test memory rows: logistic_step_dt AUC **0.864** [0.861, 0.867]; success 0.858; HLR 0.789. The old **0.877** [0.874, 0.879] was the first 80k file-order rows and is superseded. Follow-up 3 forgetting status=**unresolved** |
| **Becomes next input** | p̂_recall features for Phase 5 / demo |

### Phase 4 — Graph / next-item mix

| | |
| --- | --- |
| **Input (where from)** | `prerequisite.json` + train prefixes |
| **Processing** | Build DiGraph; soft unlock = mean parent mastery; label true next as review / unlocked advance / hard-gate violation |
| **Output** | DAG acyclic=True, longest_path=65. Mix (n=5000): review 0.898, unlocked advance 0.020, violation 0.082 |
| **Becomes next input** | Unlock / DAG features for Phase 5 advance head |

### Phase 5 — Review–Advance gated ranker (RAGR)

| | |
| --- | --- |
| **Input (where from)** | Embeddings + Phase 3 memory + DAG + similarity + train prefixes |
| **Processing** | Train P(review|history). Review head scores seen items; advance head scores unseen; mix scores |
| **Output** | `serve_models.joblib`; train mode-review rate 0.922; review_head_n=15563, advance_head_n=1880 |
| **Becomes next input** | Frozen models for Phase 6 + live demo |

### Phase 6 — Evaluation

| | |
| --- | --- |
| **Input (where from)** | Held-out test sequences (800 seq × 5 cuts); bootstrap over sequences |
| **Processing** | Recall@K / MRR; force_review / force_advance; last_item; gate AUC |
| **Output** | Primary table `review_ranking.json`: RAGR R@5=0.891 [0.880, 0.902] (sequence bootstrap). The older Phase 6 print 0.891 [0.879, 0.905] is superseded. share(next==last)=0.803; gate AUC=0.749, PR-AUC=0.955 |
| **Becomes next input** | Published metrics + trust limits |

### Follow-up 3 — Cross-dataset / forgetting / rank-reversal (pre-prereg exploratory)

| | |
| --- | --- |
| **Input (where from)** | Timed Junyi, ASSISTments, XES3G5M; gap≥1 day rows; exact-dup chance draw |
| **Processing** | Concept-level next==last; two XES KC methods; forgetting redo; popularity/recency/GRU probe/attn probe/RAGR |
| **Output** | `followup3_*.json` under `outputs_junyi/phases/` |
| **Becomes next input** | Evidence for H-rep / H-forget / H-rev before freeze |

### Probe freeze — equal-budget validation tuning (pre-prereg)

| | |
| --- | --- |
| **Input (where from)** | Learner-disjoint sequences; native + cluster units; data cards first |
| **Processing** | 24-config grid for **trainable** probes (GRU + Attn only; RAGR-lite not in that claim; GRUProbe ignores dropout); early-stop on val CE; **select by val non-repeat R@5**; hash test uids without scoring (`test_touched=false`) |
| **Output** | `probe_freeze.json` winners + CE + R@5 + ASSISTments funnel + XES question-order structure |
| **Becomes next input** | Confirmatory freeze-partition replication after `prereg-v1` (§4.13 in the PDF) |

### Confirmatory H-rev (prereg draft → after tag)

| | |
| --- | --- |
| **Input** | Freeze train+val; frozen configs + `best_epoch`; cluster maps; GRU seeds 20261101/02/03 |
| **Processing** | Fit train+val; score freeze test once (`--touch-test` + lock); two-method cluster intersection; Bonferroni m≤6; secondary common-query analysis |
| **Output** | `freeze_partition_replication.json` (post-tag only) |
| **Docs** | `docs/PREREGISTRATION.md`, `PREREGISTRATION_REPLICATION.md`, `DEVIATIONS.md` |

---

## 3. Implementation Plan (dataset travel)

```
Junyi ktbd files
  → Phase 1 length filter + concept/edge tables
  → Phase 2 MiniLM embeddings
  → Phase 3 memory classifiers (train fit / test.json AUC)
  → Phase 4 DAG + continue/revisit/advance mix
  → Phase 5 gate × (review head, advance head) → serve_models.joblib
  → Phase 6 ranking metrics on test prefixes
  → Demo: history → P(review) + ranked next exercises + subgraph
Parallel: timed Junyi / ASSISTments / XES → Follow-up 3 → probe_freeze (val only)
```

| Stage | Input | Processing | Output | Data used |
| --- | --- | --- | --- | --- |
| Prepare | Raw ktbd | Filter 12–200 | Concept/edge CSVs | Junyi train/test |
| Embed | Names | MiniLM | `.npy` vectors | Junyi names |
| Memory | Attempt rows | Logistic / HLR | p̂_correct + AUC | Junyi train→test.json |
| Graph | Edges + prefixes | Unlock labels | Mix rates | Junyi + DAG |
| Ranker | Features | Gate + 2 heads | joblib | Junyi train |
| Eval | Frozen models | R@K bootstrap | JSON metrics | Junyi test |
| Freeze | Native sequences | 24-config grid | Winner configs | Timed / ASSIST / XES |

---

## 4. Output / Results (what we actually got)

### 4.1 Files written

| Path | What it is |
| --- | --- |
| `outputs_junyi/concepts_junyi.csv` | 835 English concepts + degrees |
| `outputs_junyi/prerequisite_edges_junyi.csv` | 978 prerequisite rows |
| `outputs_junyi/concept_embeddings.npy` | (835, 384) float32 |
| `outputs_junyi/phases/phase1–6_*.json` | Phase contracts |
| `outputs_junyi/phases/followup3_*.json` | Cross-dataset / forgetting / rank-reversal |
| `outputs_junyi/phases/probe_freeze.json` | Schema v3 freeze winners |
| `outputs_junyi/serve_models.joblib` | Demo models |
| `outputs_junyi/figures/*.png` | Charts used below / in the PDF |

### 4.2 Sample live prediction (demo product output)

Scenario **Stuck student** (history: one step equations ✓ → one step inequalities ✓ → solving quadratics by factoring ✗).

System output: **P(review)=0.858**, **mode=review**, then ranked next exercises:

| Rank | Exercise | Kind | Score | p̂_recall |
| --- | --- | --- | --- | --- |
| 1 | solving quadratics by factoring | review | 2.808 | 0.084 |
| 2 | one step inequalities | review | 0.217 | 0.918 |
| 3 | one step equations | review | −0.521 | 0.933 |
| 4 | addition 1 | advance | −7.250 | 0.55 (unseen prior) |
| 5 | number line | advance | −7.345 | 0.55 (unseen prior) |

Categories: **review** = already seen in history; **advance** = never seen. Failed last item is ranked first with low recall probability.

### 4.3 Phase 4 next-item mix (intermediate)

From `phase4_graph.json` (n=5000): review **0.898**, unlocked advance **0.020**, hard-gate violation **0.082**.

### 4.4 Primary ranking table (`review_ranking.json`)

Eval: 800 seq × 5 cuts. Action mix continue/revisit_bounce/revisit_far/advance = 0.803/0.044/0.043/0.110.

| Method | R@1 (seq. boot. CI) | R@5 (seq. boot. CI) | MRR (seq. boot. CI) |
| --- | --- | --- | --- |
| RAGR | 0.802 [0.786, 0.819] | 0.891 [0.880, 0.902] | 0.845 [0.831, 0.858] |
| RAGR (full test, `paper_ranking_full`) | — | 0.889 [0.885, 0.893] (n=6290) | — |
| RAGR-no-DAG (full) | — | 0.889 [0.884, 0.893] | advance 0.080 [0.069, 0.089] |
| Markov order-1 (full) | — | 0.856 [0.850, 0.861] | advance 0.242 [0.226, 0.257] |
| Force review | 0.803 [0.787, 0.819] | 0.886 [0.875, 0.897] | 0.841 [0.828, 0.854] |
| Recent-5 | 0.803 [0.786, 0.818] | 0.885 [0.872, 0.897] | 0.840 [0.826, 0.853] |
| last_item (all_queries) | 0.803 [0.786, 0.819] | 0.803 [0.787, 0.819] | 0.804 [0.787, 0.819] |
| Frequency | 0.468 [0.446, 0.488] | 0.827 [0.812, 0.840] | 0.621 [0.603, 0.637] |

CIs are sequence-level bootstrap (mean of per-sequence means, n_boot=1000) from `review_ranking.json`. The older Phase 6 print **0.891 [0.879, 0.905]** (`phase6_eval.json` ragr_all CI [0.8794875, 0.90475]) is the same point estimate under a different bootstrap draw and is **superseded**.

### 4.5 Concept-level repetition (Follow-up 3)

| Dataset / unit | next==last | n | continue | revisit | advance |
| --- | --- | --- | --- | --- | --- |
| Junyi topic | 0.930 | 2,809,159 | 0.930 | 0.040 | 0.030 |
| ASSISTments composite skill token | 0.679 | 214,725 | 0.679 | 0.184 | 0.136 |
| XES3G5M KC (method A: question→primary KC) | 0.262 | 3,975,344 | 0.262 | 0.313 | 0.425 |
| XES3G5M KC (method B: drop multi-KC extras) | 0.262 | 3,975,344 | 0.262 | 0.313 | 0.425 |
| XES3G5M KC (method B parquet is_repeat=0) | 0.263 | 4,410,668 | 0.263 | 0.302 | 0.435 |
| ASSISTments primary atomic skill | 0.696 | 214,725 | — | — | — |

ASSISTments truncation to 200 dropped **121,660** of **339,305** rows after len≥12.

### 4.6 XES KC methods + quiz-order structure

- Method A primary KC next==last = **0.262** (n=3,975,344)
- Method B drop multi-KC = **0.262** (n=3,975,344)
- Share next is global question-order successor ≈ **0.196**
- Share next is empirical mode successor ≈ **0.350**

This is why question-level non-repeat R@5 can sit near **0.61** without modeling spaced revisit.

### 4.7 Forgetting redo (gap ≥ 1 day)

Status=**unresolved**. success AUC=0.661; +log1p gap=0.661; paired Δ=0.0003 CI [-0.0010604327518647706, 0.0015754188667389726].

### 4.8 Exact-duplicate chance

Real train∩test exact overlaps = 23; chance baseline = 23 (real−chance=0).

### 4.9 Probe freeze winners (schema v3 display; confirmatory freeze is §4.11)

Grid: 24 configs — lr=[0.0005, 0.001, 0.002], emb=[32, 64, 128, 256], dropout=[0.1, 0.3]; max_epochs=10, patience=2 (see `docs/DEVIATIONS.md`).

Attention probe is **documented but excluded** from the preregistered H-rev family (unvalidated local architecture).

| Dataset | Unit | Model | val non-rep R@5 | val CE | Config | In prereg H-rev? |
| --- | --- | --- | --- | --- | --- | --- |
| junyi_timed | exercise | gru_probe | 0.445 | 2.2640 | lr=0.001, emb=256, do=0.1 | True |
| junyi_timed | exercise | attn_probe | 0.370 | 2.2633 | lr=0.0005, emb=256, do=0.1 | False |
| assistments | composite_skill_token | gru_probe | 0.750 | 1.1291 | lr=0.0005, emb=256, do=0.3 | True |
| assistments | composite_skill_token | attn_probe | 0.660 | 1.2444 | lr=0.001, emb=256, do=0.3 | False |
| xes3g5m | question | gru_probe | 0.614 | 3.8651 | lr=0.0005, emb=256, do=0.1 | True |
| xes3g5m | question | attn_probe | 0.515 | 4.7530 | lr=0.0005, emb=256, do=0.3 | False |

Slices with <500 queries are marked N/A (e.g. XES revisit on val subsample).

See **§4.11** for schema-4 freeze tables (cluster-unit Markov, used cluster counts 115/106/120, val query counts, dual clustering). The PDF §4.9 rank-reversal study also points to PDF §4.11.

### 4.11 Schema-4 freeze (native + cluster Markov; val query counts)

Used cluster counts after dense remap: **Junyi 115 / ASSISTments 106 / XES 120** (target 120). ASSISTments k≈120 is near-native (150 composite tokens).

Markov order-1 is frozen at **both** units. A probe that loses to Markov cannot claim a learned H-rev win at that unit: the 1-step table already captures the regularity (sticky or curriculum-order). Native XES: Markov 0.739 vs GRU 0.614 (7,439 classes; freeze max_epochs=10, patience=2, best_epoch=4; `GRUProbe` ignores the dropout hyperparameter). Cluster XES: GRU 0.761 vs Markov 0.636.

| Dataset | n_cl | Pop R@5 (nq) | Markov nat. (nq) | GRU nat. (nq) | Markov cl. (nq) | GRU cl. (nq) |
| --- | --- | --- | --- | --- | --- | --- |
| junyi_timed | 115 | 0.162 (762) | 0.430 (777) | 0.445 (784) | 0.471 (636) | 0.515 (630) |
| assistments | 106 | 0.271 (628) | 0.568 (628) | 0.750 (628) | 0.594 (599) | 0.771 (599) |
| xes3g5m | 120 | 0.011 (1989) | 0.739 (1985) | 0.614 (1987) | 0.636 (651) | 0.761 (679) |

Feasible matched-count cells for a later preregistration: **k≈40** all three; **k≈120** all three but ASSISTments near-native; **k≈800** Junyi + XES only, ASSISTments **N/A**.

Dual clustering: text + train-only co-occurrence on Junyi and ASSISTments; XES question text **unavailable locally** (ids/KC tags only) so co-occurrence only. Junyi text↔cooc ARI=0.038; ASSIST ARI=0.073. A clustering-method contrast counts only if both methods agree.

### 4.13 Calibration and reliability (exploratory product heads)

| Predictor | n | base rate | Brier | ECE | ΔBrier vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate \(p_{\mathrm{rev}}\) | 31,450 | 0.886 | 0.0928 | 0.0335 | −0.0085 |
| \(p_{\mathrm{recall}}\) (continue) | 24,809 | 0.474 | 0.1480 | 0.0410 | — |

Source: `paper_calibration.json`. Negative ΔBrier means better than predicting the constant base rate. Reliability diagrams: `paper_assets/figures/fig_gate_reliability.png`, `fig_memory_reliability.png`.

### 4.12 Intermediate schemas (before → after examples)

| Stage | Before | After |
| --- | --- | --- |
| Phase 1 | Raw `train.json` lines of varying length | Only sequences with length ∈ [12, 200]; CSVs of 835 concepts / 978 edges |
| Phase 2 | English slug strings | L2-normalised 384-d vectors in `concept_embeddings.npy` |
| Phase 3 | Attempt stream `(item, correct)` | Memory rows only on **revisits**; model outputs P(correct) |
| Phase 4 | Prefix + DAG | Labels continue / revisit / advance + soft unlock features |
| Phase 5 | Features + labels | `serve_models.joblib` (gate + review head + advance head) |
| Phase 6 | Test prefixes | Ranked id lists + Recall@K / MRR tables in JSON |
| Demo | Student history | P(review), mode, top-k exercises with kind ∈ {review, advance} |

---

## 5. Visualizations

All under `outputs_junyi/figures/` (also embedded in the PDF):

| Figure file | What it shows |
| --- | --- |
| `data_distributions.png` | Train sequence lengths; correct/incorrect balance |
| `system_flow.png` | End-to-end system flow |
| `embedding_cosine.png` | Linked vs random embedding cosine |
| `next_item_mix.png` | Review / unlock / violation mix |
| `ranking_recall.png` | Ranking method comparison |
| `memory_auc_bars.png` | Memory AUC bake-off |
| `memory_roc.png` | **ROC curves** (logistic vs success on test.json) |
| `memory_confusion.png` | Confusion @0.5 for logistic on **all 449,453** test.json revisit memory rows |
| `dag_quadratic.png` / `dag_addition.png` / `dag_overview.png` | Expert DAG neighbourhoods |
| `gate_regret.png` | Gate threshold vs Recall@5 trade-off |
| `paper_assets/figures/fig_gate_reliability.png` | Gate reliability diagram (full-test calibration) |
| `paper_assets/figures/fig_memory_reliability.png` | p̂_recall reliability on continue queries |
| `system_flow.png` | End-to-end system diagram |

### How to open figures

```bash
open outputs_junyi/figures/memory_roc.png
open outputs_junyi/figures/memory_confusion.png
open outputs_junyi/figures/ranking_recall.png
open paper_assets/figures/fig_gate_reliability.png
open docs/NeuroTrace-DAG_Data_Flow_Report.pdf   # all figures embedded
```

---

## 6. ROC / AUC Curve

### What is being classified?

Phase 3 memory model: binary label = **correctness on revisit attempts only** (0/1). A row exists only when a concept is attempted again in the same sequence; the first attempt on a concept is not a row.

### Inputs / predictions used for the ROC

| Piece | Value |
| --- | --- |
| Fit data | Memory rows from `train.json`, split by **sequence id** (leakage_check=pass) |
| Score data | **All** memory rows from official **`test.json`** (449,453 revisit rows). Task = correctness on revisit attempts only |
| Features (logistic_step_dt) | `log1p(attempts)`, past success rate, `log1p(Δt in steps)` |
| Prediction values | `predict_proba` = P(correct) |
| Ground truth | Observed correctness of that next attempt |

### What the ROC represents

Trade-off between true-positive rate and false-positive rate as the decision threshold on P(correct) moves. AUC ≈ probability that a random correct attempt scores higher than a random incorrect one.

### Measured AUCs (`phase3_memory.json` → `evaluate_on_test_json`)

| Model | AUC | 95% CI | PR-AUC |
| --- | --- | --- | --- |
| logistic_step_dt | **0.864** | [0.861, 0.867] | 0.826 |
| success_rate | 0.858 | [0.855, 0.862] | 0.815 |
| hlr_step | 0.789 | [0.784, 0.794] | 0.762 |

CIs are a sequence-level bootstrap (n_boot=400) on all 449,453 revisit rows (`phase3_memory_fulltest.json`). ktbd has no timestamps, so the day-gap logistic matches the step-gap logistic.

Superseded: first 80,000 file-order rows, logistic AUC **0.877** (row-level CI [0.874, 0.879]), CM TN/FP/FN/TP = 49,189 / 4,606 / 10,317 / 15,888.

Confusion @0.5 on all 449,453 rows (logistic_step_dt): TN=209,734, FP=39,385, FN=56,994, TP=143,340.

**Separate binary task — gate:** P(review|history) on Phase 6 queries: AUC **0.749**, PR-AUC **0.955**, base rate **0.890**.

**Full-test calibration (not ROC, but related):** gate Brier **0.0928**, ECE **0.0335** (n=31,450); p̂_recall on continue queries Brier **0.1480**, ECE **0.0410** (`paper_calibration.json`). Reliability diagrams: `fig_gate_reliability.png`, `fig_memory_reliability.png`.

### Confusion matrix (logistic @ 0.5, all 449,453 revisit rows)

|  | Pred incorrect | Pred correct |
| --- | ---: | ---: |
| **Actual incorrect** | TN 209,734 | FP 39,385 |
| **Actual correct** | FN 56,994 | TP 143,340 |

Accuracy @0.5 ≈ **0.786**. This is a *memory* classifier, not the next-item ranker.

---

## 7. Complete Dataset → Input → Processing → Output Flow

```
Dataset taken: Junyi ktbd-junyi (835 concepts, 978 edges,
               27,434/6,290 train/test sequences)
   (+ timed Junyi / ASSISTments / XES3G5M for cross-checks)
        ↓
Why: traces + expert DAG in one id space; study continue/revisit/advance validity
        ↓
Phase 1 input: raw ktbd files (data/junyi_ktbd/)
Phase 1 processing: length filter 12–200; concept + edge tables
Phase 1 output: concepts/edges CSVs; train interactions 2,342,784
                 (train+test 2,842,883; topic n 2,809,159)
        ↓
Phase 2 input: 835 English names from Phase 1
Phase 2 processing: all-MiniLM-L6-v2 embeddings + random/same-topic controls
Phase 2 output: concept_embeddings.npy (pair-AUC 0.904)
        ↓
Phase 3 input: train sequences (fit) + official test.json revisits (score)
Phase 3 processing: memory classifiers; sequence-id split (leakage=0)
Phase 3 output: 449,453 revisit rows; logistic AUC 0.864 [0.861, 0.867];
                 CM @0.5 TN/FP/FN/TP = 209734/39385/56994/143340
                 (0.877 on first 80k file-order rows is SUPERSEDED)
        ↓
Phase 4 input: DAG + train prefixes
Phase 4 processing: unlock labels; continue/revisit/advance mix
Phase 4 output: review mix 0.898; unlock / violation features
        ↓
Phase 5 input: embeddings + memory + DAG + similarity + train prefixes
Phase 5 processing: train gate × review head × advance head
Phase 5 output: serve_models.joblib (product models)
        ↓
Phase 6 input: frozen models + test prefixes
Phase 6 processing: R@K + gate AUC; sequence bootstrap
Phase 6 output: review_ranking.json — RAGR R@5 0.891 [0.880, 0.902]
                 vs Recent-5 0.885; gate AUC 0.749
                 (older 0.891 [0.879, 0.905] print SUPERSEDED)
        ↓
Paper eval (exploratory): full-test ranking + calibration + sensitivity
        ↓
Follow-up 3: ASSISTments / XES repetition + diagnostic rank grids
        ↓
Probe freeze (val only): equal-budget GRU/Attn + Markov; test hashed not scored
        ↓
Confirmatory H-rev (AFTER prereg-v1 + OSF only):
        one --touch-test → freeze_partition_replication.json primary_claims
        (currently touch_test=false → pending_confirmatory)
        ↓
Final product output: ranked next exercises + P(review) + p̂_recall + DAG subgraph
        ↓
How we show it: this document (§4–6), PDF figures, data cards, phases/*.json,
                IEEE conference draft (paper/ieee/), live demo
```

### Reader map

| Question | Section |
| --- | --- |
| What data / why / where used? | §1 |
| Phase I/O? | §2 |
| End-to-end travel plan? | §3 + §7 |
| Actual outputs / samples / tables? | §4 (+ §4.10–4.12) |
| Charts? | §5 |
| ROC / AUC / CM? | §6 |
| Trace original dataset → final output? | **§7** |

### Reproduce

```bash
python3 scripts/fetch_datasets.py
python3 run_all_phases.py          # or run_valid_eval.py for validated Phase 3/6
python3 run_followup3.py
python3 run_probe_freeze.py
python3 scripts/run_paper_eval.py  # full-test ranking / calibration (slow)
python3 scripts/generate_report_assets.py
python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=full   # this PDF
python3 scripts/build_dataflow_report.py --mode=slim   # paper-facing slim PDF
```

Git tag `prereg-v1` is on GitHub. OSF registration submitted at https://osf.io/24sxf (await approval / DOI). Deviations: `docs/DEVIATIONS.md`. After OSF is public, once:

```bash
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test
PYTHONPATH=. python3 scripts/build_paper_assets.py   # JSON-only fill of H-rev table
```

