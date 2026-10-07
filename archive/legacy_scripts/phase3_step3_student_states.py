"""
Phase 3 - Step 3: Student States + Saved Evaluation Metrics
NeuroTrace-DAG

Two jobs, using the model trained in Step 2:

1. STUDENT STATES: for every student's most RECENT attempt, compute their
   current half-life h = 2^(w.x) and their recall probability at several
   review gaps (1 / 7 / 30 days from now) -> phase3_student_states.csv

2. EVALUATION METRICS: re-run the same train/test split Step 2 used (same
   seed, so it's the exact same held-out students), score the saved model,
   and write AUC / RMSE / MAE to a file instead of just printing them
   -> phase3_evaluation_metrics.txt

Run AFTER phase3_step1_features.py and phase3_step2_train_hlr.py, in the
same folder as their outputs (interactions_features.csv, hlr_model.npz).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "interactions_features.csv"
MODEL_PATH = ROOT / "hlr_model.npz"
STATES_OUT = ROOT / "phase3_student_states.csv"
METRICS_OUT = ROOT / "phase3_evaluation_metrics.txt"

REVIEW_GAPS_DAYS = [1, 7, 30]


# ---------------------------------------------------------------------
# Shared helpers - identical logic to Step 2, so predictions line up
# exactly with how the model was trained.
# ---------------------------------------------------------------------
def load_model():
    d = np.load(MODEL_PATH, allow_pickle=True)
    return d["w"], d["feat_mean"], d["feat_std"], list(d["features"])


def build_design_matrix(df: pd.DataFrame, features, feat_mean, feat_std):
    X = ((df[features] - feat_mean) / feat_std).values
    bias = np.ones((len(df), 1))
    return np.hstack([bias, X])


def predict_p(w, X, delta_t):
    log_h = X @ w
    h = np.clip(2 ** log_h, 1e-3, 1e6)
    p = 2 ** (-delta_t / h)
    return np.clip(p, 1e-6, 1 - 1e-6), h


def train_test_split_by_student(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Must match Step 2 exactly (same seed) so 'test' here means the same
    held-out students the model never trained on."""
    rng = np.random.default_rng(seed)
    student_ids = df["id_student"].unique()
    rng.shuffle(student_ids)
    n_test = int(len(student_ids) * test_frac)
    test_students = set(student_ids[:n_test])
    return df[~df["id_student"].isin(test_students)].reset_index(drop=True), \
           df[df["id_student"].isin(test_students)].reset_index(drop=True)


# ---------------------------------------------------------------------
# Job 1: evaluation metrics, saved to disk
# ---------------------------------------------------------------------
def compute_and_save_metrics(df, w, feat_mean, feat_std, features):
    _, test_df = train_test_split_by_student(df)

    X_test = build_design_matrix(test_df, features, feat_mean, feat_std)
    delta_t_test = test_df["delta_t"].values
    y_test = test_df["target"].values

    p_test, h_test = predict_p(w, X_test, delta_t_test)

    mae = mean_absolute_error(y_test, p_test)
    rmse = np.sqrt(mean_squared_error(y_test, p_test))
    auc = roc_auc_score(test_df["recall_label"].values, p_test)

    lines = [
        "PHASE 3 - HLR EVALUATION METRICS",
        f"Held-out test set: {len(test_df)} rows, {test_df['id_student'].nunique()} students",
        f"(same student-level split as Step 2, seed=0, test_frac=0.2)",
        "",
        f"AUC-ROC : {auc:.4f}   (predicted p(t) ranked against binary recall_label)",
        f"RMSE    : {rmse:.4f}  (predicted p(t) vs continuous target = score/100)",
        f"MAE     : {mae:.4f}",
        "",
        f"Predicted half-life on test set (days): "
        f"median={np.median(h_test):.1f}  p10={np.percentile(h_test,10):.1f}  "
        f"p90={np.percentile(h_test,90):.1f}",
        "",
        "NOTE: target is continuous (score/100), chosen to avoid the 96.7%/3.3%",
        "imbalance in recall_label (see Step 1). AUC still uses recall_label",
        "since it needs a binary reference. concept is a single module-level",
        "placeholder ('BBB') for every row - see Phase 1 known limitation.",
    ]
    METRICS_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved -> {METRICS_OUT.name}")


# ---------------------------------------------------------------------
# Job 2: current per-student state, saved to disk
# ---------------------------------------------------------------------
def compute_and_save_student_states(df, w, feat_mean, feat_std, features):
    df = df.sort_values(["id_student", "assessment_date"]).reset_index(drop=True)

    # Most recent attempt per student.
    last = df.groupby("id_student", as_index=False).tail(1).copy()

    # IMPORTANT: last row's own n_attempts_so_far / past_success_rate /
    # days_since_first_exposure describe the student's state BEFORE that
    # attempt (that's how Step 1 built them, on purpose, to avoid leakage
    # during training). For "current state right now", we roll each of
    # those forward by one step to include the attempt that just happened.
    counts = df.groupby("id_student")["recall_label"].agg(["count", "mean"])
    first_date = df.groupby("id_student")["assessment_date"].min()

    last = last.set_index("id_student")
    last["n_attempts_so_far"] = counts["count"]          # total attempts, now including the last one
    last["past_success_rate"] = counts["mean"]            # success rate including the last one
    last["days_since_first_exposure"] = last["assessment_date"] - first_date
    last = last.reset_index()

    X_now = build_design_matrix(last, features, feat_mean, feat_std)
    log_h = X_now @ w
    h_now = np.clip(2 ** log_h, 1e-3, 1e6)

    out = last[["id_student", "code_presentation", "assessment_date",
                "n_attempts_so_far", "past_success_rate",
                "days_since_first_exposure"]].rename(
        columns={"assessment_date": "last_assessment_date"})
    out["half_life_days"] = h_now
    for g in REVIEW_GAPS_DAYS:
        out[f"p_recall_{g}d"] = np.clip(2.0 ** (-g / h_now), 1e-6, 1 - 1e-6)

    out.to_csv(STATES_OUT, index=False)
    print(f"\nSaved -> {STATES_OUT.name}  ({len(out)} students, "
          f"one row each = their most recent state)")
    print(out.head(5).to_string())


def main():
    w, feat_mean, feat_std, features = load_model()
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded model (features: {features})")
    print(f"Loaded {len(df)} rows, {df['id_student'].nunique()} students\n")

    compute_and_save_metrics(df, w, feat_mean, feat_std, features)
    compute_and_save_student_states(df, w, feat_mean, feat_std, features)


if __name__ == "__main__":
    main()