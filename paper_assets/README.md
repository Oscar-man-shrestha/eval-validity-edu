# Paper assets (LaTeX-ready)

All scalars load from `outputs_junyi/phases/*.json` via `scripts/build_paper_assets.py`.

## Main paper vs supplement

| Main paper | Supplement / Data Flow PDF |
| --- | --- |
| `tab_hrev_confirmatory.tex` (+ dataset claims) | Full freeze grids, SHA locks, phase I/O |
| `tab_ranking_capped_vs_full.tex` | Sensitivity grids (`tab_sensitivity_*`) |
| `tab_calibration.tex` | Seed-0 full rank-reversal slice tables |
| `tab_advance_r5_cross_dataset.tex` | Dual-clustering deep-dive |
| `tab_scope_conf_vs_expl.tex` | Demo / product walkthrough |
| `paper_main_numbers.json` | `numbers.json` (full audit dump) |

Figures for main paper: `embedding_cosine`, `next_item_mix`, `ranking_recall`, reliability diagrams, one DAG neighbourhood.

## Confirmatory H-rev fill rule

Until `freeze_partition_replication.json` has `touch_test=true`, H-rev cells are `pending_confirmatory`.

After the human tags `prereg-v1`, registers on OSF, and runs `--touch-test` **once**:

```bash
PYTHONPATH=. python3 scripts/build_paper_assets.py
```

This regenerates `tab_hrev_*.tex` and `paper_main_numbers.json` from
`primary_claims` **only**. Never edit claim cells by hand.

Regenerate (exploratory assets):

```bash
PYTHONPATH=. python3 scripts/run_paper_eval.py   # slow: full test + rank-reversal
PYTHONPATH=. python3 scripts/build_paper_assets.py
```

Canonical capped ranking CI remains `review_ranking.json` [0.880, 0.902].
