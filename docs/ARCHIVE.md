# Archive layout

Legacy OULAD + MOOCCubeX work is **not** the student model. It lives under `archive/` so the repo root stays Junyi-only.

| Path | What |
| --- | --- |
| `archive/legacy_oulad_mooccubex/` | Old CSVs, embeddings, DAG drawings, translation caches, Phase 1 Word report |
| `archive/legacy_scripts/` | Old phase1–4 OULAD/MOOCCubeX scripts and PDF builders |
| `docs/` | Paper, team guide, data-flow PDF, validated results, this note |
| `data/junyi_ktbd/` | Official EduData sequences + DAG (primary) |
| `data/junyi_raw/` | `junyi.rar` + `timed_interactions.npz` (gitignored) |
| `outputs_junyi/` | Current phase JSON, figures, serve bundle |
| `serve/` | FastAPI demo |
| `run_valid_eval.py` | Validated re-eval |
| `run_nontrivial_eval.py` | Non-last / mask-last slices |
| `run_inertia_diagnostics.py` | Inertia Illusion extras |
| `run_all_phases.py` | Original six-phase runner |
| `scripts/draw_junyi_dags.py` | Graphviz DAG figures |

## Do not cite as current results

- `docs/NeuroTrace-DAG_Pipeline.pdf` — pre-validity pipeline sketch  
- `outputs_junyi/results.txt` / old row-split memory AUCs in that file  

Use `outputs_junyi/all_phases_results.txt` and `docs/VALIDATED_RESULTS.md`.

Do not delete `archive/` unless you are sure you will never need the English MOOCCubeX resource graph again.
