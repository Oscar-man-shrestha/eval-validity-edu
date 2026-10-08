# OSF / Zenodo deposit pack for `prereg-v1`

**Registration:** https://osf.io/24sxf (submitted 2026-10-07 UTC; pending approval — DOI after public).  
Local git tag exists. This file is a **deposit helper** (not part of the locked hash set). After approval, paste DOI below, then run `--touch-test` **once**.

Full documentation map: [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Git lock (already done locally)

| Item | Value |
| --- | --- |
| Tag | `prereg-v1` |
| Commit (this repo) | `ce4a75bd3f8442670f8e09594d29ef0c455337f6` |
| Branch | `main` |
| Remote | `https://github.com/Oscar-man-shrestha/eval-validity-edu.git` |
| Release | https://github.com/Oscar-man-shrestha/eval-validity-edu/releases/tag/prereg-v1 |

File digests in `prereg_v1_locked_hashes.json` are the protocol lock (content SHA-256). Public code home: this repo (`eval-validity-edu`).

```bash
git push origin main
git push origin prereg-v1
```

## Attach these files to the registration

From the tagged commit (also zipped at `docs/osf_prereg_v1_packet.zip`):

1. `docs/PREREGISTRATION.md`
2. `docs/PREREGISTRATION_REPLICATION.md`
3. `docs/DEVIATIONS.md`
4. `outputs_junyi/phases/prereg_v1_locked_hashes.json` (optional but recommended)

## Paste-ready registration abstract (human voice)

Use this (or close) on OSF — keep the numbers/seeds exact; wording can stay plain.

```
We are preregistering one confirmatory test for NeuroTrace-DAG (tag prereg-v1).

Question: when we score next-item non-repeat Recall@5 with a small GRU probe
versus a first-order Markov baseline, does the sign of that paired contrast
flip if we switch from the platform’s native action unit to a coarser concept
cluster (~120 groups)? A dataset counts as supporting the claim only if both
units are supported_* and the signs disagree. The project claim is that this
happens on at least one of Junyi timed, ASSISTments 2009, or XES3G5M.

We planned six family members (3 datasets × native/cluster) and use Bonferroni
at 1 − 0.05/m. Ineligible cells drop out of m. For Junyi and ASSISTments cluster
cells, text and co-occurrence maps must agree on the supported_* label
(intersection), so that cell still counts as one member, not two.

Confirmatory GRU seeds: 20261101, 20261102, 20261103.
Learner freeze split seed: 20261004.
We will score the held-out test learners only once after this registration,
via scripts/run_freeze_partition_replication.py --touch-test. A second run
needs an explicit TOUCH_TEST_OVERRIDE note in docs/DEVIATIONS.md.

Junyi ranking / memory / graph work stays exploratory and is not part of this
claim. GitHub: https://github.com/Oscar-man-shrestha/eval-validity-edu (tag prereg-v1 @ ce4a75b).
Protocol file digests: outputs_junyi/phases/prereg_v1_locked_hashes.json
```

### Optional short field blurbs (paste into remaining OSF pages)

**Title:** NeuroTrace-DAG prereg-v1: does action-unit granularity flip GRU vs Markov on educational logs?

**Subject search:** Educational Assessment → select
`Education / Educational Assessment, Evaluation, and Research`

**Research questions:**  
Primary question (prereg-v1): when we score next-item non-repeat Recall@5 with a small GRU probe versus a first-order Markov baseline, does the sign of that paired contrast flip if we switch from the platform’s native action unit to a coarser concept cluster (~120 groups)?

A dataset supports the claim only if both native and cluster unit verdicts are supported_* with opposite signs. The project claim is that this happens on at least one of Junyi timed, ASSISTments 2009, or XES3G5M.

Code: https://github.com/Oscar-man-shrestha/eval-validity-edu (tag prereg-v1).

**Foreknowledge radio:** Authors have observed the data, but have not performed the proposed analyses.

**Foreknowledge explanation:**  
Public educational logs exist. We used validation learners while choosing the primary claim (including an XES native-vs-cluster pattern on validation, disclosed in docs/PREREGISTRATION.md). We have not scored the confirmatory freeze holdout test yet. Code: https://github.com/Oscar-man-shrestha/eval-validity-edu (tag prereg-v1).

**Study type / causal / blinding:** Non-randomized study; No causal relationship inferred; No blinding is involved.

**Study design:**  
Secondary analysis of three public student logs (Junyi timed, ASSISTments 2009, XES3G5M) under a learner-disjoint freeze split (seed 20261004). We compare a small GRU probe against a first-order Markov baseline at the platform’s native action unit and at matched concept clusters (~120). Confirmatory outcome is paired GRU−Markov non-repeat Recall@5 on held-out test learners, scored only once after this registration. Junyi ranking / memory / graph work stays exploratory.

**Randomization / code home:**  
No new recruitment; public datasets only. Code and locked digests: https://github.com/Oscar-man-shrestha/eval-validity-edu (tag prereg-v1; outputs_junyi/phases/prereg_v1_locked_hashes.json). Freeze learner split seed 20261004; confirmatory GRU seeds 20261101, 20261102, 20261103.

**Data collection:**  
Existing public logs: Junyi timed (capped learners), ASSISTments 2009 skill-builder corrected, XES3G5M question-level. Freeze learner split seed 20261004; train+val for fit; test held out until one --touch-test after registration.

**Sample size:**  
Inclusion: sequences length 12–200 (Junyi timed max_users=6000). ASSISTments truncate_each_learner_to_200. Cell eligibility: ≥500 non-repeat queries (both clustering methods for Junyi/ASSISTments cluster cells).

**Sample size rationale / stopping:**  
Sample sizes follow freeze splits in probe_freeze.json (locked). Confirmatory bootstrap ≥10000 sequence resamples. Confirmatory scoring runs once after this registration; a second run needs TOUCH_TEST_OVERRIDE in docs/DEVIATIONS.md.

**Variables / Analysis Plan (all short answers):**  
Use: “See docs/PREREGISTRATION.md on https://github.com/Oscar-man-shrestha/eval-validity-edu (tag prereg-v1). Primary claim = native↔cluster GRU−Markov sign flip; m_planned=6 Bonferroni; GRU seeds 20261101/02/03; two-method intersection on Junyi/ASSISTments; single --touch-test after registration.”

**Do not Register until Oscar says proceed.**

## SHA-256 table (paste into registration)

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

## After OSF/Zenodo succeeds — record here

| Field | Value |
| --- | --- |
| Draft URL | https://osf.io/registries/drafts/6ac5e15b290d421e40e88a55/metadata (locked — already registered) |
| Registration URL | https://osf.io/24sxf |
| Status | Pending registration approval (submitted; not public yet) |
| DOI | _appears after approval — check https://osf.io/24sxf_ |
| Timestamp (UTC) | 2026-10-07T07:03:34Z |

Then **once**:

```bash
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=slim
python3 scripts/build_dataflow_report.py --mode=full
```

Do **not** hand-edit H-rev claim cells. Do **not** re-touch without `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md`.
