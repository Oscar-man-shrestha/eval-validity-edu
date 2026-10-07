"""Re-score Phase 6 with non-last / mask-last slices using the serve bundle (fast)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from junyi_pipeline import DATA, OUT, SEED, load_graph, load_names, load_sequences
from run_all_phases import build_heads, write_json
from run_valid_eval import EVAL_POS, EVAL_SEQS, evaluate_ranking

BUNDLE = OUT / "serve_models.joblib"


def main() -> None:
    rng = np.random.default_rng(SEED)
    names = load_names()
    graph, similar = load_graph(len(names))
    embeddings = np.load(OUT / "concept_embeddings.npy")
    test_kt = load_sequences(DATA / "test.json")
    parent_lists = [list(graph.predecessors(c)) for c in range(len(names))]
    bundle = joblib.load(BUNDLE)
    popularity = np.asarray(bundle["popularity"])
    feat_fn = build_heads(
        embeddings, graph, similar, popularity, parent_lists, bundle["logreg"]
    )
    models = (bundle["mode_clf"], bundle["review_clf"], bundle["advance_clf"], feat_fn)
    print(f"ranking eval nontrivial slices… EVAL_SEQS={EVAL_SEQS} cuts={EVAL_POS}", flush=True)
    ranking = evaluate_ranking(
        test_kt, embeddings, graph, similar, popularity, models, parent_lists, rng
    )
    phase6_path = OUT / "phases" / "phase6_eval.json"
    phase6 = json.loads(phase6_path.read_text()) if phase6_path.exists() else {"phase": 6}
    phase6.update(
        {
            "bootstrap": "over_sequences",
            "eval_sequences": EVAL_SEQS,
            "cuts_per_sequence": EVAL_POS,
            "ranking": ranking,
            "diagnostic_note": (
                "ragr_nonlast / ragr_mask_last_nonlast isolate the harder ~20% of clicks "
                "where next != last item (sticky restudy)."
            ),
        }
    )
    write_json("phase6_eval.json", phase6)

    # append diagnostic lines to headline
    summary = OUT / "all_phases_results.txt"
    lines = summary.read_text().splitlines() if summary.exists() else []
    # drop old diagnostic lines if re-run
    lines = [ln for ln in lines if not ln.startswith("         nonlast") and not ln.startswith("         mask_last")]
    if "ragr_nonlast" in ranking:
        lines.append(
            f"         nonlast RAGR R@5={ranking['ragr_nonlast']['mean']:.3f} "
            f"CI={ranking['ragr_nonlast']['ci']}  "
            f"force_review={ranking['force_review_nonlast']['mean']:.3f}  "
            f"share_nontrivial_review={ranking['share_nontrivial_review']:.3f}"
        )
    if "ragr_mask_last_nonlast" in ranking:
        lines.append(
            f"         mask_last on nonlast R@5={ranking['ragr_mask_last_nonlast']['mean']:.3f} "
            f"CI={ranking['ragr_mask_last_nonlast']['ci']}"
        )
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))
    print("wrote", phase6_path)


if __name__ == "__main__":
    main()
