"""Build the figures used by NeuroTrace-DAG_Data_Flow_Report.pdf from real data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve

from junyi_pipeline import (
    DATA,
    OUT,
    SEED,
    fit_hlr,
    hlr_predict,
    iter_memory_rows,
    load_graph,
    load_names,
    load_sequences,
    pair_auc,
)

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PHASES = OUT / "phases"


def load_phase(n: int) -> dict:
    names = {
        1: "phase1_prepare.json",
        2: "phase2_embeddings.json",
        3: "phase3_memory.json",
        4: "phase4_graph.json",
        5: "phase5_ranker.json",
        6: "phase6_eval.json",
    }
    return json.loads((PHASES / names[n]).read_text())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#243447",
    "text.color": "#243447",
    "axes.labelcolor": "#243447",
    "xtick.color": "#243447",
    "ytick.color": "#243447",
})


def save(fig, name: str) -> Path:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    paper_fig = ROOT / "paper_assets" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_fig / Path(name).with_suffix(".pdf").name
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    print("wrote", pdf_path)
    return path


def fig_next_item_mix():
    mix = load_phase(4)["next_item_mix_train_sample"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = ["Review\n(already seen)", "Advance,\nunlocked", "Advance,\nwould fail hard gate"]
    vals = [mix["review_frac"], mix["advance_unlocked_frac"], mix["advance_violation_frac"]]
    colors = ["#1f6f61", "#c4a35a", "#b3261e"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of true next items")
    ax.set_title(f"Junyi next click is mostly restudy (train sample n={mix['n']})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1%}", ha="center", fontsize=10)
    save(fig, "next_item_mix.png")


def fig_ranking():
    """Canonical capped ranking from review_ranking.json (not superseded phase6)."""
    rr = json.loads((PHASES / "review_ranking.json").read_text())
    m = rr["metrics"]
    slices = rr.get("slices") or {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    methods = ["RAGR", "Force\nreview", "Last-item"]
    vals = [
        m["ragr_r5"]["mean"],
        m["force_review_r5"]["mean"],
        m["last_item_r5"]["mean"],
    ]
    colors = ["#0d9488", "#3d8b7a", "#0a1628"]
    bars = axes[0].bar(methods, vals, color=colors, width=0.62)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Recall@5")
    axes[0].set_title("Overall (sticky restudy dominates)")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    hard = slices.get("far") or slices.get("advance")
    if hard and "ragr_r5" in hard:
        m2 = ["RAGR", "Recent-5", "Force\nreview"]
        v2 = [
            hard["ragr_r5"]["mean"],
            hard["recent5_r5"]["mean"],
            hard["force_review_r5"]["mean"],
        ]
        bars2 = axes[1].bar(m2, v2, color=["#0d9488", "#94a3b8", "#1e4d7b"], width=0.62)
        axes[1].set_ylim(0, max(0.7, max(v2) + 0.1))
        axes[1].set_ylabel("Recall@5")
        axes[1].set_title("Hard slice: revisit-far")
        for b, v in zip(bars2, v2):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "See review_ranking.json slices", ha="center")
        axes[1].axis("off")

    nq = m["ragr_r5"].get("n_queries", rr.get("eval", {}).get("n_queries", 4000))
    ci = m["ragr_r5"]["ci"]
    fig.suptitle(
        f"Canonical ranking (review_ranking) · RAGR R@5={m['ragr_r5']['mean']:.3f} "
        f"[{ci[0]:.3f}, {ci[1]:.3f}] · {nq} queries",
        fontsize=10,
        y=1.02,
    )
    save(fig, "ranking_recall.png")


def fig_embeddings():
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, _ = load_graph(len(names))
    emb = np.load(OUT / "concept_embeddings.npy")
    linked = [float(emb[s] @ emb[t]) for s, t in graph.edges() if s != t]
    random = []
    n = emb.shape[0]
    while len(random) < len(linked):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            random.append(float(emb[i] @ emb[j]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(random, bins=28, density=True, alpha=0.55, color="#9a7b3c", label=f"Random pairs  mean={np.mean(random):.3f}")
    ax.hist(linked, bins=28, density=True, alpha=0.55, color="#1f6f61", label=f"Prerequisite pairs  mean={np.mean(linked):.3f}")
    # Title MUST use frozen phase2 JSON (recompute can round 0.904↔0.905)
    p2 = load_phase(2)
    pair = float(p2["pair_auc_vs_random"])
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Embeddings recover relatedness, not chance  pair-AUC={pair:.3f}")
    ax.legend(frameon=False)
    save(fig, "embedding_cosine.png")


def fig_memory_roc():
    """Real ROC + confusion on official test.json (same protocol as run_valid_eval)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

    from junyi_pipeline import DATA, load_sequences
    from run_valid_eval import assert_no_leak, memory_rows_by_sequence, split_by_id

    p3 = load_phase(3)
    te = p3["evaluate_on_test_json"]
    labels = [
        ("Logistic (step Δt)", te["logistic_step_dt"]["auc"]),
        ("Past success rate", te["success_rate"]["auc"]),
        ("HLR (step Δt)", te["hlr_step"]["auc"]),
        ("Mean baseline", te["mean"]["auc"]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    names = [x[0] for x in labels]
    aucs = [x[1] for x in labels]
    colors = ["#0d9488", "#3d8b7a", "#c4a35a", "#9aa0a6"]
    bars = ax.barh(names[::-1], aucs[::-1], color=colors[::-1])
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("AUC on all test.json revisit memory rows")
    ax.set_title("Phase 3 memory bake-off (revisit attempts only)")
    for b, v in zip(bars, aucs[::-1]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    save(fig, "memory_auc_bars.png")

    train_kt = load_sequences(DATA / "train.json")
    test_kt = load_sequences(DATA / "test.json")
    feats, ys, sids = [], [], []
    for feat, y, sid in memory_rows_by_sequence(train_kt):
        feats.append(feat)
        ys.append(y)
        sids.append(sid)
        if len(feats) >= 200_000:
            break
    x = np.vstack(feats)
    y = np.array(ys)
    sids = np.array(sids)
    tr_ids, te_ids = split_by_id(sids)
    assert_no_leak(tr_ids, te_ids, "ktbd-sequence")
    tr_m = np.array([s in tr_ids for s in sids])
    feats_te, y_pub = [], []
    for feat, yy, _ in memory_rows_by_sequence(test_kt):
        feats_te.append(feat)
        y_pub.append(yy)
    x_pub = np.vstack(feats_te)
    y_pub = np.array(y_pub)
    clf = LogisticRegression(max_iter=400).fit(x[tr_m][:, 1:4], y[tr_m])
    p_log = clf.predict_proba(x_pub[:, 1:4])[:, 1]
    p_succ = x_pub[:, 2]
    fpr_l, tpr_l, _ = roc_curve(y_pub, p_log)
    fpr_s, tpr_s, _ = roc_curve(y_pub, p_succ)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot(fpr_l, tpr_l, color="#0d9488", lw=2.2, label=f"logistic_step_dt  AUC={auc(fpr_l, tpr_l):.3f}")
    ax.plot(fpr_s, tpr_s, color="#c4a35a", lw=1.8, label=f"success_rate      AUC={auc(fpr_s, tpr_s):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Phase 3 ROC — all {len(y_pub):,} test.json revisit memory rows")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "memory_roc.png")

    cm = confusion_matrix(y_pub, (p_log >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay(cm, display_labels=["wrong(0)", "correct(1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"Phase 3 logistic @0.5 — all {len(y_pub):,} test.json revisit memory rows")
    for t in ax.texts:
        try:
            t.set_text(f"{int(float(t.get_text())):,}")
        except ValueError:
            pass
    save(fig, "memory_confusion.png")


def fig_sequence_and_labels():
    train = load_sequences(DATA / "train.json")
    lens = [len(s) for s in train]
    acc = np.mean([c for s in train[:4000] for _, c in s])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].hist(lens, bins=40, color="#1f6f61")
    axes[0].set_title(f"Train sequence length (n={len(train)})")
    axes[0].set_xlabel("Attempts in sequence")
    axes[0].set_ylabel("Sequences")
    axes[1].bar(["Wrong (0)", "Correct (1)"], [1 - acc, acc], color=["#b3261e", "#1f6f61"], width=0.55)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Attempt labels (sample acc={acc:.2f})")
    axes[1].set_ylabel("Share")
    save(fig, "data_distributions.png")


def fig_flow():
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, "Junyi\nraw / KTBD"),
        (2.0, "Phase 1\nclean tables"),
        (3.8, "Phase 2\nembeddings"),
        (5.6, "Phase 3+4\nmemory + DAG"),
        (7.4, "Phase 5–6\ngate + rank"),
    ]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 0.55), 1.55, 1.05, fc="#d7efe8", ec="#1f6f61", lw=1.2))
        ax.text(x + 0.78, 1.07, text, ha="center", va="center", fontsize=8)
    for x in (1.75, 3.55, 5.35, 7.15):
        ax.annotate("", xy=(x + 0.22, 1.07), xytext=(x, 1.07),
                    arrowprops=dict(arrowstyle="->", color="#243447"))
    ax.set_title("Actual implementation flow (not a generic ML cartoon)")
    save(fig, "system_flow.png")


def fig_gate_regret():
    path = PHASES / "phase6_inertia_diagnostics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())["gate_threshold_regret_curve"]
    taus = [r["tau_review_if_p_ge"] for r in data]
    all_r = [r["r5_all"] for r in data]
    adv_r = [r["r5_advance"] for r in data]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(taus, all_r, "o-", color="#0a1628", label="All queries R@5")
    ax.plot(taus, adv_r, "s-", color="#0d9488", label="Advance slice R@5")
    ax.set_xlabel("τ  (use review head if P(review) ≥ τ)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Gate threshold regret: unlocking costs overall hit-rate")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    save(fig, "gate_regret.png")


def main() -> None:
    fig_flow()
    fig_next_item_mix()
    fig_ranking()
    fig_embeddings()
    fig_sequence_and_labels()
    fig_memory_roc()
    fig_gate_regret()
    print("figures ready in", FIG)


if __name__ == "__main__":
    main()
