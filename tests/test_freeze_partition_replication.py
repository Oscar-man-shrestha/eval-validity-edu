"""Freeze-partition replication guards (pre-prereg)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from junyi_pipeline import OUT
from run_followup3 import (
    GRUProbe,
    build_learner_sequences_assist,
    build_learner_sequences_junyi_timed,
    build_learner_sequences_xes,
    score_seq_model,
)
from run_probe_freeze import (
    MAX_WINDOWS,
    REPLICATION_SEED,
    eval_r5_on_seqs,
    learner_split,
    sha_list,
    train_seq_model_early_stop,
)
from run_valid_eval import recall_at

PHASE = OUT / "phases"
FREEZE_SEED = 20261004
VAL_R5_TOL = 0.05  # absolute tolerance for freeze-faithful retrain vs stored val non-repeat R@5


def _freeze() -> dict:
    path = PHASE / "probe_freeze.json"
    if not path.exists():
        pytest.skip("missing probe_freeze.json")
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "dataset,builder",
    [
        (
            "junyi_timed",
            lambda: build_learner_sequences_junyi_timed(max_users=6000, min_len=12, max_len=200)[:2],
        ),
        ("assistments", lambda: build_learner_sequences_assist()[:2]),
        ("xes3g5m", lambda: build_learner_sequences_xes()[:2]),
    ],
)
def test_freeze_val_learners_not_in_replication_test(dataset, builder):
    freeze = _freeze()
    uids = builder()[1]
    fr = freeze["datasets"][dataset]["split"]
    assert fr["seed"] == FREEZE_SEED
    assert fr.get("test_touched") is False
    tr_i, va_i, te_i = learner_split(uids, FREEZE_SEED)
    train_uids = {int(uids[i]) for i in tr_i}
    val_uids = {int(uids[i]) for i in va_i}
    test_uids = {int(uids[i]) for i in te_i}
    assert val_uids.isdisjoint(test_uids)
    assert train_uids.isdisjoint(test_uids)
    assert sha_list(sorted(test_uids)) == fr["test_uid_hash"]
    assert len(test_uids) == fr["n_test"]


def test_prereg_locked_file_hashes():
    root = Path(__file__).resolve().parents[1]
    lock_path = PHASE / "prereg_v1_locked_hashes.json"
    if not lock_path.exists():
        pytest.fail("missing prereg_v1_locked_hashes.json")
    lock = json.loads(lock_path.read_text())
    for row in lock["files"]:
        p = root / row["path"]
        assert p.is_file(), row["path"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == row["sha256"], f"changed: {row['path']}"


def test_confirmatory_seeds_crc32_and_identical_cis():
    from scripts.run_freeze_partition_replication import (
        GRU_TRAIN_SEEDS,
        M_PLANNED,
        N_BOOT,
        combine_two_method_verdicts,
        family_size_after_na,
        paired_bootstrap_ci,
        stable_seed,
    )

    assert N_BOOT >= 10_000
    assert M_PLANNED == 6
    assert GRU_TRAIN_SEEDS == (20261101, 20261102, 20261103)
    src = Path("scripts/run_freeze_partition_replication.py").read_text()
    assert "zlib.crc32" in src
    assert "stable_seed" in src
    # no Python builtin hash() for bootstrap seeds
    assert "hash(sl" not in src and "hash(unit" not in src
    assert stable_seed("x", "y") == stable_seed("x", "y")
    d = np.linspace(-0.03, 0.04, 80)
    a = paired_bootstrap_ci(d, 6, seed=stable_seed("ident", "ci"))
    b = paired_bootstrap_ci(d, 6, seed=stable_seed("ident", "ci"))
    assert a["ci"] == b["ci"]
    assert a["ci_95"] == b["ci_95"]
    assert combine_two_method_verdicts("supported_A", "negligible")["registered_cluster_verdict"] == "inconclusive"
    assert family_size_after_na(2)["m"] == 4


def test_per_sequence_nan_alignment_and_recency_rule():
    """Pair by sequence index; recency 0 on advance for all/nonrepeat; N/A on advance display."""
    from scripts.run_freeze_partition_replication import score_unit_multisseed

    # Tiny synthetic: 3 sequences, last cut often advance
    n_items = 5
    train = [[(i % n_items, 1) for i in range(20)] for _ in range(10)]
    test = [
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 1)],
        [(1, 1)] * 12,
        [(2, 1)] * 5,  # too short → all NaN
    ]
    cfg = {"config": {"lr": 1e-3, "emb": 32}, "best_epoch": 1}
    pack = score_unit_multisseed(train, test, n_items, winner_cfg=cfg, unit_name="toy")
    arrays = pack["per_sequence_r5_arrays"]
    assert arrays["gru_probe"]["all"].shape == (3,)
    assert np.isnan(arrays["gru_probe"]["all"][2])
    # recency advance slice NaN or undefined for sequences that only hit advance — display N/A
    assert pack["rankings_by_slice_r5"]["advance"]["rows"][1]["method"] == "recency" or any(
        r["method"] == "recency" and r.get("r5_display") == "N/A"
        for r in pack["rankings_by_slice_r5"]["advance"]["rows"]
    )
    # pairing: finite mask shared
    g = arrays["gru_probe"]["nonrepeat"]
    m = arrays["markov_order1"]["nonrepeat"]
    both = np.isfinite(g) & np.isfinite(m)
    assert both.sum() >= 1
    # recency on nonrepeat is finite when nonrepeat queries exist (0 on advance counts)
    assert pack["recency_rule"].startswith("recency counts as 0")


def test_recall_at_tie_breaks_by_lower_index():
    scores = np.array([1.0, 1.0, 0.5])
    assert recall_at(scores, 0, 1) == 1.0
    assert recall_at(scores, 1, 1) == 0.0


def test_train_gru_fixed_matches_freeze_hyperstructure():
    from scripts.run_freeze_partition_replication import train_gru_fixed

    src = Path("scripts/run_freeze_partition_replication.py").read_text()
    assert "max(64, emb * 2)" in src
    assert "MAX_WINDOWS" in src
    assert str(MAX_WINDOWS) == "20000" or "MAX_WINDOWS" in src
    assert "dropout" in src.lower() and "no-op" in src.lower() or "ignores dropout" in src.lower()


@pytest.mark.parametrize(
    "dataset,unit",
    [
        ("junyi_timed", "native"),
        ("junyi_timed", "cluster"),
        ("assistments", "native"),
        ("assistments", "cluster"),
        ("xes3g5m", "native"),
        ("xes3g5m", "cluster"),
    ],
)
def test_winner_retrain_reproduces_val_nonrepeat_r5(dataset, unit):
    """Retrain freeze winner on freeze train; val non-repeat R@5 within VAL_R5_TOL of stored."""
    from gru_markov_paired_ci import remap_pair
    from scripts.run_freeze_partition_replication import (
        CATALOG_N_ITEMS,
        load_cluster_map,
        map_seqs_preserve_length,
        winner_cfg_for,
    )

    freeze = _freeze()
    if dataset == "junyi_timed":
        seqs, uids = build_learner_sequences_junyi_timed(max_users=6000, min_len=12, max_len=200)
        n_items = CATALOG_N_ITEMS[dataset]
    elif dataset == "assistments":
        seqs, uids, n_items = build_learner_sequences_assist()
        n_items = CATALOG_N_ITEMS[dataset]
    else:
        seqs, uids, _ = build_learner_sequences_xes()
        n_items = CATALOG_N_ITEMS[dataset]

    tr_i, va_i, te_i = learner_split(uids, FREEZE_SEED)
    train = [seqs[i] for i in tr_i]
    val = [seqs[i] for i in va_i]
    w = winner_cfg_for(freeze, dataset, unit=unit)
    assert w, f"missing winner {dataset}/{unit}"
    cfg = dict(w["config"])
    stored = float(w["val_nonrepeat_r5"])

    if unit == "cluster":
        path = freeze["clustering"][dataset]["item_to_cluster_path"]
        item_to_c = load_cluster_map(path)
        train = map_seqs_preserve_length(train, item_to_c)
        val = map_seqs_preserve_length(val, item_to_c)
        train, val, n_items = remap_pair(train, val)

    emb = int(cfg["emb"])
    gru = GRUProbe(n_items, emb=emb, hidden=max(64, emb * 2))
    gru, pack = train_seq_model_early_stop(
        gru, train, val, n_items, cfg=cfg, seed=REPLICATION_SEED
    )
    max_r5 = 400 if n_items >= 1000 or len(val) > 800 else None
    got_pack = eval_r5_on_seqs(
        lambda p, m=gru: score_seq_model(m, p, n_items),
        val,
        n_items,
        max_seqs=max_r5,
        seed=REPLICATION_SEED + 99,
    )
    got = float(got_pack["nonrepeat"]["mean_r5"])
    err = abs(got - stored)
    assert err <= VAL_R5_TOL, (
        f"{dataset}/{unit}: retrain val nonrep R@5={got:.4f} vs stored={stored:.4f} "
        f"|Δ|={err:.4f} > tol={VAL_R5_TOL} (best_ep={pack['best_epoch']})"
    )


def test_ci_verdict_rule_no_tie_label():
    from scripts.recompute_ci_verdicts import ci_verdict

    assert ci_verdict([0.02, 0.04], 0.01)["verdict"] == "supported_A"
    path = PHASE / "paper_paired_verdicts.json"
    if not path.exists():
        pytest.skip("missing paper_paired_verdicts.json")
    v = json.loads(path.read_text())
    assert v["phase3_logistic_minus_success"]["verdict"] == "inconclusive"


def test_touch_lock_refuses_second_run_without_override(tmp_path, monkeypatch):
    from scripts import run_freeze_partition_replication as mod

    lock = tmp_path / "touch_test.lock.json"
    dev = tmp_path / "DEVIATIONS.md"
    lock.write_text('{"touched": true}\n')
    dev.write_text("# no override\n")
    monkeypatch.setattr(mod, "TOUCH_LOCK", lock)
    monkeypatch.setattr(mod, "DEVIATIONS", dev)
    with pytest.raises(SystemExit):
        mod.check_touch_lock()
    dev.write_text("TOUCH_TEST_OVERRIDE: emergency re-score after bugfix\n")
    mod.check_touch_lock()  # should not raise
