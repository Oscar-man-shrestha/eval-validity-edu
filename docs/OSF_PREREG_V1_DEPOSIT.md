# OSF / Zenodo deposit pack for `prereg-v1`

Local git tag is already created. This file is a **deposit helper** (not part of the locked hash set). Fill the DOI/URL after registration, then run `--touch-test` once.

## Git lock (already done locally)

| Item | Value |
| --- | --- |
| Tag | `prereg-v1` |
| Tag object | `cd7ebbc97a36bc0bb49261a5dbf3f67f2da97300` |
| Commit | `29e98ef791139e4c095c33c931fe219f027345ab` |
| Branch | `audit/verify-all-phases` |
| Remote | `https://github.com/Oscar-man-shrestha/eval-validity-edu.git` |

Push (if not already):

```bash
git push origin audit/verify-all-phases
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
claim. Git commit locked by tag prereg-v1: 29e98ef791139e4c095c33c931fe219f027345ab
```

### Optional short field blurbs (if OSF asks again)

**Study design (plain):** Secondary analysis of three public student logs under a
learner-disjoint freeze split. We compare GRU vs Markov at native vs cluster
units; test is untouched until after this registration.

**Foreknowledge (plain):** We have seen the public logs and used validation
learners while choosing the primary claim (including an XES native-vs-cluster
pattern on validation). We have not scored the confirmatory freeze test yet.

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
| Draft URL | https://osf.io/registries/drafts/6ac5e15b290d421e40e88a55/metadata |
| Registration URL | _paste after submit_ |
| DOI | _paste after submit_ |
| Timestamp (UTC) | _paste after submit_ |

Then **once**:

```bash
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test
PYTHONPATH=. python3 scripts/build_paper_assets.py
python3 scripts/build_dataflow_report.py --mode=slim
python3 scripts/build_dataflow_report.py --mode=full
```

Do **not** hand-edit H-rev claim cells. Do **not** re-touch without `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md`.
