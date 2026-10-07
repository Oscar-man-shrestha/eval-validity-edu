# NeuroTrace-DAG — Research Integrity Report

_Generated from `outputs_junyi/phases/research_integrity_audit.json`. Do not hand-edit metrics._

## What this project is

**multi-dataset evaluation-validity and protocol study**.

The project studies whether a standard next-item benchmark actually measures useful educational progression. It does not claim to have created a superior learning recommender.

## Findings that survive the audit

### 1. Action granularity changes what ‘repetition’ means

| Dataset / concept unit | next == last | transitions | continue | revisit | advance |
| --- | --- | --- | --- | --- | --- |
| Junyi topic | 0.930 | 2809159 | 0.930 | 0.040 | 0.030 |
| ASSISTments skill | 0.679 | 214725 | 0.679 | 0.184 | 0.136 |
| XES3G5M KC (method A: question→primary KC) | 0.262 | 3975344 | 0.262 | 0.313 | 0.425 |
| XES3G5M KC (method B: drop multi-KC extras) | 0.262 | 3975344 | 0.262 | 0.313 | 0.425 |
| XES3G5M KC (method B parquet is_repeat=0) | 0.263 | 4410668 | 0.263 | 0.302 | 0.435 |

This is the main contribution: conclusions based on item-level repeat rates do not automatically transfer across topic, skill, and knowledge-component units.

### 2. The prototype is not practically superior to recency

RAGR Recall@5 = **0.891**; Recent-5 = **0.885**. The paired difference is **0.0067** [0.0045, 0.0095], below the predeclared practical bar of **0.01**. Status: **not_practically_superior**.

### 3. Parent readiness is detectable but negligible

Primary readiness ΔAUC = **0.0005** [0.0002, 0.0008]. The practical threshold was **0.005**. Status: **detectable_but_negligible**.

### 4. The timed forgetting question remains unresolved

On 125637 held-out gap≥1-day return rows from 8999 learners, the success-only AUC is **0.661** and the success+log-gap AUC is **0.661**. The paired ΔAUC is **0.0003** [-0.0011, 0.0016]. Status: **unresolved**.

## Method labels that are accurate

The rank-reversal file is **not_comparable_to_published_GRU4Rec_or_SASRec**. The local neural probes trained for 3 epochs on at most 80000 windows using cross_entropy; they are not faithful published GRU4Rec or SASRec implementations. Use only the label **GRU-sequence probe / attention-sequence probe** in exploratory notes.

## What to say to a teacher or interviewer

“We discovered that next-item performance can mostly reflect a dataset’s repetition structure. We introduced a reproducible protocol that separates continue, revisit, and advance actions; reports simple recency baselines; and checks the result at topic, skill, and KC levels across three educational logs. We deliberately report where the prototype does not show a practically meaningful gain.”

## What we do not claim

- state-of-the-art recommender
- RAGR practically outperforms simple recency baselines
- the current probe is a faithful GRU4Rec or SASRec reproduction
- DAG readiness provides a practically meaningful correctness gain
- forgetting features improve prediction on the tested data

## Research references

- [Ren et al. (2019), RepeatNet: A Repeat Aware Neural Recommendation Machine.](https://ojs.aaai.org/index.php/AAAI/article/view/4408) — Repeat/explore gating predates RAGR; cite it instead of claiming the gate as novel.
- [Hidasi et al. (2016), Session-based Recommendations with Recurrent Neural Networks.](https://hidasi.eu/assets/pdf/gru4rec_iclr16.pdf) — Original GRU4Rec uses a session-recommendation ranking protocol; the local two-epoch probe is not a reproduction.
- [Kang and McAuley (2018), Self-Attentive Sequential Recommendation.](https://arxiv.org/abs/1808.09781) — Original SASRec is a separate validated method; the local attention probe is not a published SASRec baseline.

## Reproduce

```bash
python3 scripts/build_research_integrity_audit.py
python3 scripts/render_research_integrity_docs.py
python3 -m pytest tests/test_leakage_guards.py -q
```
