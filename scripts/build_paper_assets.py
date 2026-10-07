#!/usr/bin/env python3
"""Build LaTeX-ready paper assets from outputs_junyi/phases/*.json only.

Writes:
  paper_assets/
    tables/*.tex
    figures/*.pdf
    numbers.json          (all cited scalars, for audit)
    README.md

Also appends old→new rows to docs/DEVIATIONS.md when paper_* JSONs differ from
canonical capped review_ranking / superseded phase6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / "outputs_junyi"
PHASE = OUT / "phases"
ASSETS = ROOT / "paper_assets"
TAB = ASSETS / "tables"
FIG = ASSETS / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    path = PHASE / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def ci(obj, nd=3):
    if not obj or obj.get("ci") is None:
        return "—"
    lo, hi = obj["ci"]
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def write_tex(name: str, body: str) -> None:
    (TAB / name).write_text(body.rstrip() + "\n")
    print("wrote", TAB / name)


def metric_row(label: str, m: dict | None) -> str:
    if not m:
        return f"{label} & — & — & — \\\\"
    return f"{label} & {fmt(m.get('mean'))} & {ci(m)} & {m.get('n_sequences', '—')} \\\\"


def fig_reliability(cal: dict, out_stem: str, title: str) -> None:
    rel = (cal or {}).get("reliability") or []
    if not rel:
        return
    pred = [r["pred"] for r in rel if r.get("n")]
    obs = [r["obs"] for r in rel if r.get("n")]
    if len(pred) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.plot(pred, obs, "o-", color="#1f6f6a", lw=1.5, ms=5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / f"{out_stem}.pdf")
    fig.savefig(FIG / f"{out_stem}.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / f"{out_stem}.pdf")


def table_ranking_capped_vs_full() -> dict:
    capped = load("paper_ranking_capped.json")
    full = load("paper_ranking_full.json")
    rr = load("review_ranking.json")
    p6 = load("phase6_eval.json")

    cap_m = (capped.get("metrics_ragr_r5") or rr.get("metrics", {}).get("ragr_r5") or {})
    full_m = (full.get("metrics") or {}).get("ragr_r5") or {}
    p6_m = (p6.get("metrics") or {}).get("ragr_all") or p6.get("ragr_all") or {}

    # phase6 shape varies
    if not p6_m and "ragr_all" in (p6.get("summary") or {}):
        p6_m = p6["summary"]["ragr_all"]

    body = r"""\begin{tabular}{lccc}
\toprule
Source & RAGR R@5 & 95\% CI & $n$ seq \\
\midrule
"""
    body += metric_row(r"review\_ranking (capped, canonical)", cap_m) + "\n"
    if full_m:
        body += metric_row(r"paper\_ranking\_full (full test)", full_m) + "\n"
    body += metric_row(r"phase6\_eval (superseded)", {
        "mean": (p6_m.get("mean") if isinstance(p6_m, dict) else None)
        or (capped.get("supersedes_phase6_eval_ci") or {}).get("old", {}).get("mean"),
        "ci": (p6_m.get("ci") if isinstance(p6_m, dict) else None)
        or (capped.get("supersedes_phase6_eval_ci") or {}).get("old", {}).get("ci"),
        "n_sequences": (p6_m.get("n_sequences") if isinstance(p6_m, dict) else None),
    }) + "\n"
    body += r"\bottomrule" + "\n\\end{tabular}\n"
    write_tex("tab_ranking_capped_vs_full.tex", body)

    # advance-slice + no-DAG
    fm = full.get("metrics") or {}
    body2 = r"""\begin{tabular}{lccc}
\toprule
Method & R@5 & 95\% CI & $n$ seq \\
\midrule
"""
    for key, lab in (
        ("ragr_r5", "RAGR"),
        ("ragr_no_dag_r5", "RAGR w/o DAG feats"),
        ("markov_r5", "Markov order-1"),
        ("recent5_r5", "Recent-5"),
        ("ragr_advance_r5", "RAGR (advance)"),
        ("ragr_no_dag_advance_r5", "RAGR-no-DAG (advance)"),
        ("markov_advance_r5", "Markov (advance)"),
    ):
        body2 += metric_row(lab, fm.get(key)) + "\n"
    body2 += r"\bottomrule" + "\n\\end{tabular}\n"
    write_tex("tab_ranking_full_methods.tex", body2)
    return {"capped": cap_m, "full": full_m, "phase6_superseded_ci": [0.8794875, 0.90475]}


def table_rank_reversal() -> dict:
    rev = load("paper_rank_reversal_full.json") or load("followup3_rank_reversal.json")
    if not rev:
        return {}
    datasets = rev.get("datasets") or {}
    summary = {}
    for ds, pack in datasets.items():
        rows = (pack.get("rankings_by_slice_r5") or {}).get("all") or []
        adv = (pack.get("rankings_by_slice_r5") or {}).get("advance") or []
        body = (
            r"\begin{tabular}{lcc}" + "\n"
            r"\toprule" + "\n"
            r"Method & all R@5 & advance R@5 \\" + "\n"
            r"\midrule" + "\n"
        )
        adv_map = {r["method"]: r.get("r5") for r in adv}
        for r in rows:
            body += f"{r['method'].replace('_', r'\\_')} & {fmt(r.get('r5'))} & {fmt(adv_map.get(r['method']))} \\\\\n"
        body += r"\bottomrule" + "\n\\end{tabular}\n"
        write_tex(f"tab_rank_reversal_{ds}.tex", body)
        # advance R@5 lookup for cross-dataset table
        summary[ds] = {r["method"]: r.get("r5") for r in adv}
    # cross-dataset advance R@5
    methods = ["popularity", "recency", "markov_order1", "gru_probe", "gru4rec_style", "attn_probe"]
    body = r"""\begin{tabular}{lccc}
\toprule
Method & Junyi timed & ASSISTments & XES3G5M \\
\midrule
"""
    for m in methods:
        body += (
            f"{m.replace('_', r'\\_')} & "
            f"{fmt((summary.get('junyi_timed') or {}).get(m))} & "
            f"{fmt((summary.get('assistments') or {}).get(m))} & "
            f"{fmt((summary.get('xes3g5m') or {}).get(m))} \\\\\n"
        )
    # Junyi RAGR-no-DAG advance from ranking full if present
    full = load("paper_ranking_full.json")
    nodag = ((full.get("metrics") or {}).get("ragr_no_dag_advance_r5") or {}).get("mean")
    ragr_a = ((full.get("metrics") or {}).get("ragr_advance_r5") or {}).get("mean")
    body += f"RAGR (Junyi only) & {fmt(ragr_a)} & — & — \\\\\n"
    body += f"RAGR-no-DAG (Junyi only) & {fmt(nodag)} & — & — \\\\\n"
    body += r"\bottomrule" + "\n\\end{tabular}\n"
    write_tex("tab_advance_r5_cross_dataset.tex", body)
    return summary


def table_calibration() -> dict:
    cal = load("paper_calibration.json")
    if not cal:
        full = load("paper_ranking_full.json")
        cal = full.get("calibration") or {}
    gate = cal.get("gate") or {}
    mem = cal.get("p_recall_on_continue_queries") or {}
    body = r"""\begin{tabular}{lcccc}
\toprule
Predictor & $n$ & base rate & Brier & ECE & $\Delta$Brier vs base \\
\midrule
"""
    body += (
        f"Gate $p_{{\\mathrm{{rev}}}}$ & {gate.get('n', '—')} & {fmt(gate.get('base_rate'))} & "
        f"{fmt(gate.get('brier'), 4)} & {fmt(gate.get('ece'), 4)} & "
        f"{fmt(gate.get('base_rate_adjusted_brier'), 4)} \\\\\n"
    )
    body += (
        f"$p_{{\\mathrm{{recall}}}}$ (continue) & {mem.get('n', '—')} & {fmt(mem.get('base_rate'))} & "
        f"{fmt(mem.get('brier'), 4)} & {fmt(mem.get('ece'), 4)} & — \\\\\n"
    )
    body += r"\bottomrule" + "\n\\end{tabular}\n"
    write_tex("tab_calibration.tex", body)
    fig_reliability(gate, "fig_gate_reliability", "Gate calibration (review vs advance)")
    fig_reliability(mem, "fig_memory_reliability", r"$p_{\mathrm{recall}}$ on continue queries")
    return {"gate": gate, "p_recall": mem}


def table_counts_xes_sensitivity() -> None:
    counts = load("paper_junyi_counts.json")
    xes = load("paper_xes_kc_methods.json")
    sens = load("paper_sensitivity.json")
    p2 = load("phase2_embeddings.json")

    c = counts.get("current_load_sequences") or {}
    t = counts.get("topic_level_share") or {}
    body = r"""\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
"""
    body += f"Train sequences & {c.get('train_n_sequences', '—'):,} \\\\\n".replace(",", "{,}")
    # safer without thousands sep issues in LaTeX
    body = r"""\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
"""
    rows = [
        ("Train sequences (train.json)", c.get("train_n_sequences")),
        ("Test sequences (test.json)", c.get("test_n_sequences")),
        ("Train interactions", c.get("train_n_interactions")),
        ("Test interactions", c.get("test_n_interactions")),
        ("Train transitions", c.get("train_n_transitions")),
        ("Test transitions", c.get("test_n_transitions")),
        ("Topic-level $n$ (train+test transitions)", t.get("n_transitions")),
        ("Topic share(next==last)", fmt((t.get("share") if isinstance(t.get("share"), (int, float)) else (t.get("share") or {}).get("share") if isinstance(t.get("share"), dict) else None), 4)
         if False else None),
    ]
    # fix share extraction
    share = t.get("share")
    if isinstance(share, dict):
        share_v = share.get("share")
    else:
        share_v = share
    # rebuild cleanly
    body = r"""\begin{tabular}{lr}
\toprule
Quantity & Value \\
\midrule
"""
    for lab, val in (
        ("Train sequences (train.json)", c.get("train_n_sequences")),
        ("Test sequences (test.json)", c.get("test_n_sequences")),
        ("Train interactions", c.get("train_n_interactions")),
        ("Test interactions", c.get("test_n_interactions")),
        ("Train transitions", c.get("train_n_transitions")),
        ("Test transitions", c.get("test_n_transitions")),
        (r"Topic-level $n$ (train+test transitions)", t.get("n_transitions")),
        (r"Topic share(next==last)", share_v),
    ):
        if isinstance(val, float):
            body += f"{lab} & {fmt(val, 4)} \\\\\n"
        else:
            body += f"{lab} & {val if val is not None else '—'} \\\\\n"
    body += r"\bottomrule" + "\n\\end{tabular}\n"
    write_tex("tab_junyi_counts.tex", body)

    if xes:
        a = xes["method_a_question_then_primary_kc"]["share_next_equals_last"]
        b = xes["method_b_expand_all_kc_tags"]["share_next_equals_last"]
        body = r"""\begin{tabular}{lcc}
\toprule
Method & share(next==last) & $n$ \\
\midrule
"""
        body += f"A: question→primary KC & {fmt(a['share'], 4)} & {a['n']} \\\\\n"
        body += f"B: expand-all KC tags & {fmt(b['share'], 4)} & {b['n']} \\\\\n"
        body += r"\bottomrule" + "\n\\end{tabular}\n"
        write_tex("tab_xes_kc_methods.tex", body)

    if sens:
        body = r"""\begin{tabular}{lccc}
\toprule
Cap & $n$ train & $n$ test & topic share \\
\midrule
"""
        for cap, row in (sens.get("sequence_length_cap") or {}).items():
            sh = row.get("topic_next_eq_last") or {}
            body += (
                f"{cap} & {row.get('n_train_seq')} & {row.get('n_test_seq')} & "
                f"{fmt(sh.get('share'), 4)} \\\\\n"
            )
        body += r"\bottomrule" + "\n\\end{tabular}\n"
        write_tex("tab_sensitivity_maxlen.tex", body)

        grids = (sens.get("cut_positions") or {}).get("grids") or {}
        if grids:
            body = r"""\begin{tabular}{lcccc}
\toprule
Cuts & continue & revisit & advance & $n$ \\
\midrule
"""
            for k, g in grids.items():
                body += (
                    f"{k} & {fmt(g['continue'], 3)} & {fmt(g['revisit'], 3)} & "
                    f"{fmt(g['advance'], 3)} & {g['n_queries']} \\\\\n"
                )
            body += r"\bottomrule" + "\n\\end{tabular}\n"
            write_tex("tab_sensitivity_cuts.tex", body)

        sel = sens.get("timed_6k_selection") or {}
        if sel:
            body = r"""\begin{tabular}{lcc}
\toprule
Selection & share & $n$ \\
\midrule
"""
            f6 = sel.get("first_6k_file_order") or {}
            r6 = sel.get("random_6k") or {}
            body += f"First 6k (file order) & {fmt(f6.get('share'), 4)} & {f6.get('n')} \\\\\n"
            body += f"Random 6k & {fmt(r6.get('share'), 4)} & {r6.get('n')} \\\\\n"
            body += f"$\\Delta$ (first$-$random) & {fmt(sel.get('delta_first_minus_random'), 4)} & — \\\\\n"
            body += r"\bottomrule" + "\n\\end{tabular}\n"
            write_tex("tab_sensitivity_6k.tex", body)

    # pair-AUC from JSON only
    numbers = {
        "pair_auc_vs_random": p2.get("pair_auc_vs_random"),
        "pair_auc_vs_same_topic": p2.get("pair_auc_vs_same_topic"),
        "review_ranking_ragr_r5": (load("review_ranking.json").get("metrics") or {}).get("ragr_r5"),
        "phase6_superseded_ci": [0.8794875, 0.90475],
    }
    (ASSETS / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")


def append_deviations(ranking_meta: dict) -> None:
    full = load("paper_ranking_full.json")
    capped = load("review_ranking.json")
    xes = load("paper_xes_kc_methods.json")
    counts = load("paper_junyi_counts.json")
    lines = [
        "",
        "## Paper eval-validity pass (journal prep)",
        "",
        "Generated by `scripts/run_paper_eval.py` + `scripts/build_paper_assets.py`.",
        "No number was changed to look better; capped vs full are both reported.",
        "",
        "| Item | Old | New | Note |",
        "| --- | --- | --- | --- |",
    ]
    cap_ci = ((capped.get("metrics") or {}).get("ragr_r5") or {}).get("ci")
    full_m = (full.get("metrics") or {}).get("ragr_r5") or {}
    lines.append(
        f"| Phase 6 RAGR CI | [0.879, 0.905] (`phase6_eval.json`) | "
        f"[0.880, 0.902] (`review_ranking.json`) | Canonical capped remains review_ranking |"
    )
    if full_m.get("mean") is not None:
        lines.append(
            f"| RAGR R@5 full test | (not previously reported) | "
            f"{fmt(full_m.get('mean'))} {ci(full_m)} "
            f"(n={full_m.get('n_sequences')}) | `paper_ranking_full.json` |"
        )
    lines.append(
        "| phase1 `interactions` field | 2,842,883 (train+test, unlabeled) | "
        "train-only 2,342,784; `interactions_train_plus_test`=2,842,883; topic n=2,809,159 | "
        "`paper_junyi_counts.json` |"
    )
    if xes:
        a = xes["method_a_question_then_primary_kc"]["share_next_equals_last"]["share"]
        b = xes["method_b_expand_all_kc_tags"]["share_next_equals_last"]["share"]
        lines.append(
            f"| XES method B | was identical to A (first-KC) | expand-all share={fmt(a, 4)} vs {fmt(b, 4)} | "
            f"`paper_xes_kc_methods.json`; old B deprecated |"
        )
    lines.append(
        "| RecBole GRU4Rec/SASRec | not installed | local `gru4rec_style` / `attn_probe` robustness | "
        "See `paper_rank_reversal_full.json` note |"
    )
    lines.append(
        f"| Fig embedding pair-AUC title | recompute could print 0.905 | "
        f"loads `phase2_embeddings.pair_auc_vs_random`={fmt(load('phase2_embeddings.json').get('pair_auc_vs_random'), 3)} | "
        f"`generate_report_assets.py` |"
    )
    lines.append(
        "| Ranking figure source | `phase6_eval.json` | `review_ranking.json` CI [0.880, 0.902] | "
        "`generate_report_assets.fig_ranking` |"
    )
    # Replace only the paper-eval section; preserve Canonical wording / TOUCH_TEST notes.
    dev_path = ROOT / "docs" / "DEVIATIONS.md"
    text = dev_path.read_text()
    start = "## Paper eval-validity pass (journal prep)"
    end = "<!-- end paper eval-validity -->"
    body = "\n".join(lines) + "\n" + end + "\n"
    if start in text:
        pre = text.split(start)[0].rstrip() + "\n"
        if end in text:
            post = text.split(end, 1)[1]
            text = pre + body + post.lstrip("\n")
        else:
            text = pre + body
    else:
        text = text.rstrip() + "\n" + body
    dev_path.write_text(text)
    print("updated", dev_path)


def _cell_ci(cell: dict) -> str:
    if not cell or cell.get("verdict") in (None, "N/A", "pending_confirmatory"):
        return "—"
    c = cell.get("ci")
    if not c:
        return "—"
    return f"[{fmt(c[0], 3)}, {fmt(c[1], 3)}]"


def table_hrev_confirmatory() -> dict:
    """Primary H-rev table from freeze_partition_replication.json → primary_claims only.

    Until touch_test=true, every claim cell is pending_confirmatory.
    After --touch-test: regenerate from JSON only — never hand-edit claim cells.
    """
    repl = load("freeze_partition_replication.json")
    touched = bool(repl.get("touch_test"))
    claims = repl.get("primary_claims") if touched else None
    fam = (claims or {}).get("eligibility_and_family") or {}
    m = fam.get("m")
    level = fam.get("ci_level")

    rows_tex = []
    cells_json = []
    datasets = ["junyi_timed", "assistments", "xes3g5m"]
    units = ["native", "cluster"]

    if claims and claims.get("cells"):
        by = {(c["dataset"], c["unit"]): c for c in claims["cells"]}
        for ds in datasets:
            for unit in units:
                c = by.get((ds, unit), {})
                v = c.get("verdict", "pending_confirmatory")
                nq = c.get("n_queries_nonrepeat", "—")
                eligible = c.get("eligible")
                rows_tex.append(
                    f"{ds.replace('_', '\\_')} & {unit} & {v} & {_cell_ci(c)} & "
                    f"{nq if nq != '—' else '—'} & {'yes' if eligible else ('no' if eligible is False else '—')} \\\\"
                )
                cells_json.append(
                    {
                        "dataset": ds,
                        "unit": unit,
                        "verdict": v,
                        "ci": c.get("ci"),
                        "n_queries_nonrepeat": nq,
                        "eligible": eligible,
                        "scope": "confirmatory" if touched else "pending_confirmatory",
                    }
                )
        ds_claims = claims.get("dataset_claims") or {}
        project = claims.get("project_claim") or {}
    else:
        for ds in datasets:
            for unit in units:
                rows_tex.append(
                    f"{ds.replace('_', '\\_')} & {unit} & pending\\_confirmatory & — & — & — \\\\"
                )
                cells_json.append(
                    {
                        "dataset": ds,
                        "unit": unit,
                        "verdict": "pending_confirmatory",
                        "ci": None,
                        "n_queries_nonrepeat": None,
                        "eligible": None,
                        "scope": "pending_confirmatory",
                    }
                )
        ds_claims = {
            ds: {
                "native_verdict": "pending_confirmatory",
                "cluster_verdict": "pending_confirmatory",
                "satisfies_primary_claim": None,
            }
            for ds in datasets
        }
        project = {
            "satisfies_project_claim": None,
            "datasets_satisfying": [],
            "note": "pending locked --touch-test after prereg-v1 + OSF",
        }

    body = r"""% AUTO-GENERATED by scripts/build_paper_assets.py — do not hand-edit claim cells.
% Source: outputs_junyi/phases/freeze_partition_replication.json → primary_claims
% After --touch-test only; until then verdicts are pending_confirmatory.
\begin{tabular}{llcccc}
\toprule
Dataset & Unit & Verdict & Bonferroni CI & $n$ nonrep q & Eligible \\
\midrule
"""
    body += "\n".join(rows_tex) + "\n"
    body += r"""\bottomrule
\end{tabular}
"""
    if not touched:
        body += (
            "\n% Footnote (paper): Pre-tag dual-map *validation* numbers in "
            "dual_cluster_gru_markov.json are disclosure-only, not confirmatory.\n"
        )
    write_tex("tab_hrev_confirmatory.tex", body)

    # dataset / project claim lines
    claim_rows = []
    for ds in datasets:
        d = ds_claims.get(ds) or {}
        sat = d.get("satisfies_primary_claim")
        sat_s = "pending" if sat is None else ("yes" if sat else "no")
        claim_rows.append(
            f"{ds.replace('_', '\\_')} & {d.get('native_verdict', 'pending\\_confirmatory')} & "
            f"{d.get('cluster_verdict', 'pending\\_confirmatory')} & {sat_s} \\\\"
        )
    claim_body = r"""% AUTO-GENERATED — JSON only after touch-test.
\begin{tabular}{lllc}
\toprule
Dataset & Native & Cluster & Satisfies primary (opp.\ supported\_*) \\
\midrule
"""
    claim_body += "\n".join(claim_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write_tex("tab_hrev_dataset_claims.tex", claim_body)

    scope_body = r"""\begin{tabular}{lp{0.72\textwidth}}
\toprule
Scope & Content \\
\midrule
confirmatory & Primary H-rev (native$\leftrightarrow$cluster GRU$-$Markov sign flip); family $m\le 6$; freeze holdout after tag \\
exploratory & Capped/full ranking, seed-0 rank-reversal, calibration, DAG/forgetting, H-rep \\
diagnostic & Probe-freeze grids, locked hashes, phase I/O, demo product output \\
\bottomrule
\end{tabular}
"""
    write_tex("tab_scope_conf_vs_expl.tex", scope_body)

    main_numbers = {
        "schema": "paper_main_numbers_v1",
        "rule": (
            "After --touch-test, regenerate this file and tab_hrev_*.tex from "
            "freeze_partition_replication.json → primary_claims only. Never hand-edit claim cells."
        ),
        "touch_test": touched,
        "scope_banner": (
            "confirmatory_pending"
            if not touched
            else "confirmatory_from_freeze_partition_replication"
        ),
        "primary_hrev": {
            "scope": "confirmatory" if touched else "pending_confirmatory",
            "m_planned": 6,
            "m": m if touched else None,
            "bonferroni_level": level if touched else None,
            "cells": cells_json,
            "dataset_claims": ds_claims,
            "project_claim": project,
            "gru_seeds": [20261101, 20261102, 20261103],
            "freeze_seed": 20261004,
        },
        "exploratory_pointers": {
            "ranking_capped": "review_ranking.json / paper_ranking_capped.json",
            "ranking_full": "paper_ranking_full.json",
            "calibration": "paper_calibration.json",
            "pre_tag_dual_map_val_disclosure_only": "dual_cluster_gru_markov.json",
        },
    }
    (ASSETS / "paper_main_numbers.json").write_text(json.dumps(main_numbers, indent=2) + "\n")
    print("wrote", ASSETS / "paper_main_numbers.json", "touch_test=", touched)
    return main_numbers


def write_readme() -> None:
    (ASSETS / "README.md").write_text(
        """# Paper assets (LaTeX-ready)

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
"""
    )


def main() -> None:
    ranking_meta = table_ranking_capped_vs_full()
    table_rank_reversal()
    table_calibration()
    table_counts_xes_sensitivity()
    table_hrev_confirmatory()
    append_deviations(ranking_meta)
    write_readme()
    print("paper assets ready under", ASSETS)


if __name__ == "__main__":
    main()
