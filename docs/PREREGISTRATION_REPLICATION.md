# Preregistration replication protocol

Run **after** the human tags `prereg-v1` and registers on OSF/Zenodo. Companion: `docs/PREREGISTRATION.md`. This file is attached to the registration and covered by the tag (with `docs/DEVIATIONS.md`).

## Canonical wording (must match PREREGISTRATION.md / DEVIATIONS.md)

| Term | Registered wording |
| --- | --- |
| **Primary H-rev claim** | Native ↔ cluster **GRU−Markov sign flip**: dataset \(D\) satisfies it iff both unit verdicts are `supported_*` with **opposite signs**; project claim = ≥1 such dataset |
| **Family / Bonferroni** | \(m_{\mathrm{planned}}=6\) (3 datasets × {native, cluster}); CI level \(1-0.05/m\); N/A cells reduce \(m\) |
| **GRU seeds** | `20261101`, `20261102`, `20261103` (per-seq R@5 = mean; bootstrap on that mean) |
| **Two-method intersection** | Junyi/ASSIST cluster cell = one family member: both text and cooc must share the same `supported_*`; else inconclusive. Cooc uses primary cluster GRU config (no retune). XES: cooc only (amendment) |
| **Touch-test lock** | First `--touch-test` writes `touch_test.lock.json`. A second run is refused unless `docs/DEVIATIONS.md` contains the exact marker `TOUCH_TEST_OVERRIDE:` plus justification |

Do **not** treat exploratory ranking, seed-0 rank-reversal, dual-map **val** checks, H-rep, graph, or forgetting as part of the primary claim.

## Human steps

1. Read `docs/PREREGISTRATION.md` end-to-end.
2. Confirm `PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q` passes (locked hashes + seeds + leakage).
3. `git tag prereg-v1` and push.
4. Register on OSF/Zenodo: attach the three docs; paste SHA-256 table / `prereg_v1_locked_hashes.json`; state that the \(m=6\) Bonferroni level remains valid under the two-method intersection rule.
5. Only then: `PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test`
6. After touch: regenerate paper H-rev tables from `freeze_partition_replication.json → primary_claims` **only** (no hand-edited claim cells).

## Protocol summary

| Item | Registered value |
| --- | --- |
| Holdout | freeze seed `20261004` (not a new seed; not `junyi_pipeline.SEED=0`) |
| Fit | freeze train+val; frozen config + fixed `best_epoch`; no early stop on test |
| GRU seeds | **20261101, 20261102, 20261103**; per-seq R@5 = mean; bootstrap on that mean; report between-seed spread |
| Markov | deterministic single fit |
| Bootstrap | ≥ **10,000** **sequence-level** resamples; `zlib.crc32` seeds |
| Family | \(m_{\mathrm{planned}}=6\); Bonferroni \(1-0.05/m\); two-method cluster = **one** cell (intersection) |
| Cell floor | ≥ **500** non-repeat queries (both methods for Junyi/ASSIST cluster); else N/A and \(m\) drops |
| Primary claim | native↔cluster GRU−Markov sign flip (opposite `supported_*`); project ≥1 dataset |
| Secondary | common non-repeat query set; same CI rule/level; primary-only flips labeled |
| H-rep | **no confirmatory test** |
| Re-touch | requires `TOUCH_TEST_OVERRIDE:` in `docs/DEVIATIONS.md` |

## Commands

```bash
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py
PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q
# after tag + OSF only:
PYTHONPATH=. python3 scripts/run_freeze_partition_replication.py --touch-test
```
