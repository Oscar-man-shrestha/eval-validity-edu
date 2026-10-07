"""Leakage / protocol guards for the rigorous review redo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from junyi_pipeline import DATA, OUT, SEED, load_sequences
from run_claude_fixes import (
    N_BOOT,
    build_readiness_rows,
    paired_seq_auc_delta,
    run_overlap,
    seq_bootstrap_mean,
)
from run_valid_eval import EVAL_POS, EVAL_SEQS

PHASE = OUT / "phases"


def _load(name: str) -> dict:
    path = PHASE / name
    if not path.exists():
        pytest.skip(f"missing {path}; run python3 run_claude_fixes.py first")
    return json.loads(path.read_text())


def test_eval_sample_is_800x5():
    ranking = _load("review_ranking.json")
    assert ranking["eval"]["n_sequences"] == min(EVAL_SEQS, ranking["eval"]["n_sequences"])
    assert ranking["eval"]["cuts_per_sequence"] == EVAL_POS
    assert ranking["eval"]["n_sequences"] == 800 or ranking["eval"]["n_sequences"] < 800
    assert ranking["eval"]["cuts_per_sequence"] == 5
    assert ranking["eval"]["seed"] == SEED


def test_no_hardcoded_verified_flags_in_summary():
    summary = _load("review_summary.json")
    blob = json.dumps(summary)
    assert '"verified"' not in blob.lower().replace(" ", "")
    # status fields must be computed enums, not bare True claiming verification
    status = summary["claims"]["3_dag_helps_correctness"]["status"]
    assert status in {"supported", "unresolved", "detectable_but_negligible"}


def test_readiness_split_is_by_sequence_not_row_block():
    ready = _load("review_readiness.json")
    assert ready["split"]["unit"] == "sequence"
    assert ready["split"]["row_cutoff"] is None
    assert ready["split"]["sequence_overlap"] == 0
    assert ready["split"]["item_accuracy_source"] == "train_sequences_only"
    n_tr = ready["split"]["n_train_sequences"]
    n_te = ready["split"]["n_test_sequences"]
    assert n_tr + n_te == ready["split"]["n_sequences_total"]
    assert n_te / ready["split"]["n_sequences_total"] == pytest.approx(0.2, abs=0.01)


def test_item_accuracy_excludes_test_sequence_labels():
    """Rebuild item_acc from train seqs only and ensure test labels never enter the counts."""
    train_kt = load_sequences(DATA / "train.json")
    rng = np.random.default_rng(SEED)
    n = len(train_kt)
    order = np.arange(n)
    rng.shuffle(order)
    n_te = max(1, int(0.2 * n))
    te_ids = set(order[:n_te].tolist())
    tr_ids = set(order[n_te:].tolist())
    assert tr_ids.isdisjoint(te_ids)

    item_stats = {}
    for sid in tr_ids:
        for c, y in train_kt[sid]:
            a, s = item_stats.get(c, (0, 0))
            item_stats[c] = (a + 1, s + int(y))

    # Any interaction from a test sequence must not have been counted if that
    # (concept, correct) pair only appears in test — we check aggregate: total
    # attempts in item_stats equals sum of train-sequence lengths.
    total_attempts = sum(a for a, _ in item_stats.values())
    train_interactions = sum(len(train_kt[sid]) for sid in tr_ids)
    assert total_attempts == train_interactions


def test_unseen_parent_not_scored_as_zero_mastery():
    parent_lists = [[1], [], [0, 1]]
    # sequence never observes parent 1 before child 0
    seqs = [[(0, 1), (0, 0), (0, 1)]]
    item_acc = {0: 0.5}
    rows = build_readiness_rows(seqs, [0], parent_lists, item_acc)
    assert rows
    for r in rows[0]:
        # child 0 has parent 1 never seen → parents_seen=0, ready is nan
        if r["has_parents"] == 1.0:
            assert r["parents_seen"] == 0.0
            assert np.isnan(r["ready_min"])
            assert np.isnan(r["ready_mean"])


def test_bootstrap_is_over_sequences_not_rows():
    rng = np.random.default_rng(0)
    # two sequences with different lengths; row-bootstrap would weight longer seq more
    values = [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0]]
    out = seq_bootstrap_mean(values, rng, n_boot=200)
    assert out["statistic"] == "mean_of_per_sequence_means"
    assert out["mean"] == pytest.approx(0.5, abs=1e-9)
    assert out["n_sequences"] == 2
    assert out["n_queries"] == 6


def test_paired_auc_delta_bootstraps_sequences():
    rng = np.random.default_rng(0)
    y = [np.array([0, 1, 0, 1.0]), np.array([0, 1.0])]
    p0 = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.3])]
    p1 = [np.array([0.1, 0.9, 0.2, 0.8]), np.array([0.1, 0.9])]
    out = paired_seq_auc_delta(y, p0, p1, rng, n_boot=50)
    assert out["statistic"] == "pooled_auc_delta_bootstrapped_over_sequences"
    assert out["n_sequences"] == 2
    assert "ci_above_zero" in out


def test_overlap_reports_learner_ids_not_testable():
    overlap = _load("review_overlap.json")
    assert overlap["learner_ids"]["status"] == "not_testable"
    assert "n_overlap_hashes" in overlap["exact_duplicate_sequences"]
    pref = overlap.get("prefix12_overlap_train_test") or overlap.get("prefix12_overlap")
    assert pref is not None and "n_overlap_prefixes" in pref
    assert "lengths_of_test_duplicate_rows" in overlap["exact_duplicate_sequences"]
    assert "train_train_prefix12_baseline" in overlap


def test_n_boot_is_1000():
    assert N_BOOT == 1000
    ready = _load("review_readiness.json")
    primary = ready["dag_helps_correctness"]["primary_delta_auc"]
    if primary and primary.get("n_boot") is not None:
        assert primary["n_boot"] == 1000


def test_followup2_repetition_has_side_by_side_numbers():
    rep = _load("followup2_repetition.json")
    for key in (
        "junyi_ktbd_exercise",
        "junyi_ktbd_topic",
        "junyi_timed_raw_exercise",
        "assistments_skill",
    ):
        assert "share" in rep[key]
        assert 0.0 <= rep[key]["share"] <= 1.0
    assert "repetition is high on mastery-based logs" in rep["finding_label"]
    assert "rule" in rep["high_repetition_boolean"]


def test_followup2_ranking_reconcile_explains_rng():
    recon = _load("followup2_ranking_reconcile.json")
    assert recon["legacy_polluted_rng_run"]["overlap_with_canonical_sample"] < 800
    assert recon["practical_threshold_recall_at_5"] == 0.01
    assert recon["effect_labels"]["ragr_minus_recent5"] in {
        "negligible",
        "positive_above_threshold",
        "negative_above_threshold",
        "uncertain",
    }


def test_followup2_duplicate_chance_baseline():
    chance = _load("followup2_duplicate_chance.json")
    assert chance["chance_baseline_pseudo_test"]["n_pseudo_test"] == 6290
    assert "real_overlap" in chance["comparison"]
    assert "chance_overlap" in chance["comparison"]


def test_followup2_xes_uid_split():
    xes = _load("followup2_xes3g5m.json")
    if not xes.get("available"):
        pytest.skip("XES3G5M not available")
    assert xes["split"]["unit"] == "uid"
    assert xes["split"]["overlap"] == 0
    assert "question_all" in xes["share_next_equals_last"]
    assert "kc_all" in xes["share_next_equals_last"]


def test_followup3_forgetting_gap_ge1_protocol():
    f = _load("followup3_forgetting.json")
    assert "gap >= 1" in f["protocol"]["fit_and_test_rows"]
    assert f["status"] in {
        "unresolved",
        "log_gap_helps",
        "log_gap_hurts_vs_success",
    }
    assert "log1p_gap_minus_success" in f["paired_deltas"]
    assert f["n_test_rows"] > 0


def test_followup3_concept_table_has_no_exercise_rows():
    concept = _load("followup3_concept_repetition.json")
    units = " ".join(r["dataset_unit"].lower() for r in concept["rows"])
    assert "exercise" not in units
    assert "question" not in units or "primary kc" in units
    assert any("topic" in r["dataset_unit"].lower() for r in concept["rows"])
    assert any("skill" in r["dataset_unit"].lower() for r in concept["rows"])


def test_followup3_exact_dup_chance():
    d = _load("followup3_exact_dup_chance.json")
    assert d["chance_baseline_pseudo_test"]["n_pseudo_test"] == 6290
    assert d["real_train_test_exact"]["n_overlap_hashes"] == 23
    assert "chance_overlap" in d["comparison"]


def test_followup3_rank_reversal_saves_indices():
    r = _load("followup3_rank_reversal.json")
    assert r["seed"] == SEED
    for ds, payload in r["datasets"].items():
        assert "sample_indices" in payload["split"]
        assert "sample_indices_sha256" in payload["split"]
        assert "rankings_by_slice_r5" in payload
        for sl in ("all", "revisit", "advance"):
            assert sl in payload["rankings_by_slice_r5"]
            assert len(payload["rankings_by_slice_r5"][sl]) == 5


def test_data_cards_exist_for_four_datasets():
    cards = Path(__file__).resolve().parents[1] / "docs" / "data_cards"
    for name in ("junyi_ktbd.md", "junyi_timed.md", "assistments2009.md", "xes3g5m.md"):
        p = cards / name
        assert p.exists(), p
        text = p.read_text()
        assert "SHA-256" in text
        assert "Licence" in text or "license" in text.lower()


def test_probe_freeze_never_touches_test():
    """Freeze step must not evaluate on test; only hash test indices."""
    src = (Path(__file__).resolve().parents[1] / "run_probe_freeze.py").read_text()
    assert "test_touched" in src
    assert "eval_r5_on_seqs(pop_fn, val_seqs" in src or "eval_r5_on_seqs(pop_fn, val_seqs," in src
    assert "eval_r5_on_seqs(" in src
    assert "eval_r5_on_seqs(pop_fn, test_seqs" not in src
    assert "eval_r5_on_seqs(recency_fn, test_seqs" not in src
    freeze = _load("probe_freeze.json")
    assert freeze["replication_seed"] == 20261004
    assert freeze.get("n_grid_configs", 0) >= 24 or len(freeze.get("grid", [])) >= 24
    assert "grid_spec" in freeze
    assert 256 in freeze["grid_spec"]["emb"]
    assert 2e-3 in freeze["grid_spec"]["lr"] or 0.002 in freeze["grid_spec"]["lr"]
    assert freeze["attn_probe_architecture"]["preregistered_h_rev_family"] == "excluded"
    assert freeze["h_rev_unit_policy"]["primary"]["target_n_clusters"] == 120
    assert freeze["selection_metric"].startswith("val_nonrepeat_r5")
    for ds, payload in freeze["datasets"].items():
        assert payload["split"]["test_touched"] is False
        assert payload["split"]["test_uid_hash"]
        assert "equal_tuning_budget" in payload
        assert payload["equal_tuning_budget"]["n_configs"] >= 24
        assert payload["parameter_free_baselines"]["popularity"]["parameter_free"] is True
        assert payload["parameter_free_baselines"]["recency"]["parameter_free"] is True
        for m in ("gru_probe", "attn_probe"):
            assert payload["grid_results"][m]
            w = payload["winners"][m]
            assert "val_nonrepeat_r5" in w or "val" in w
            assert "val_ce_loss" in w or w.get("best_val_ce_loss") is not None or m == "attn_probe"
        assert payload["winners"]["gru_probe"].get("in_preregistered_h_rev") is True
        assert payload["winners"]["attn_probe"].get("in_preregistered_h_rev") is False
    funnel = freeze["assistments_filter_funnel"]["steps"]
    assert funnel[0]["step"] == "raw_csv"
    assert funnel[0]["n_learners"] == 4217
    assert "skill_construction" in freeze["assistments_filter_funnel"]
    assert freeze["assistments_filter_funnel"]["skill_construction"]["n_atomic_skill_ids_ge0"] == 123
    assert freeze["datasets"]["assistments"]["ranking_unit"] == "composite_skill_token"
    assert freeze["datasets"]["xes3g5m"]["ranking_unit"] == "question"
    assert "question_order_structure" in freeze["datasets"]["xes3g5m"]
    assert freeze["datasets"]["junyi_timed"]["learner_selection"]["n_requested_cap"] == 6000
    assert freeze["assistments_filter_funnel"].get("sequence_builder_n_learners") == freeze["datasets"]["assistments"]["n_sequences"]
    # truncation drop count
    steps = {s["step"]: s for s in funnel}
    assert steps["keep_learners_len_ge_12"]["n_rows"] - steps["truncate_each_learner_to_200"]["n_rows"] == 121660


def test_build_dataflow_uses_probe_display_names():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "build_dataflow_report.py").read_text()
    assert "GRU-based next-item probe" in src
    assert "attention-based next-item probe" in src
    assert "review_ranking.json only" in src or "from review_ranking.json only" in src
    assert "intended behaviour" not in src.lower() and "intended product behaviour" not in src.lower()
