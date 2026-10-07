"""
Phase 3 - Step 2: Train the Half-Life Regression (HLR) model
NeuroTrace-DAG

The one step in this whole project with real model training. Fits
h = 2^(w . x), p(t) = 2^(-delta_t / h), by gradient descent minimizing
squared error against the continuous target from Step 1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "interactions_features.csv"
MODEL_PATH = ROOT / "hlr_model.npz"

FEATURES = ["n_attempts_so_far", "past_success_rate", "days_since_first_exposure"]


def train_test_split_by_student(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Split by student ID, not by row - so test students are never seen in training."""
    rng = np.random.default_rng(seed)
    student_ids = df["id_student"].unique()
    rng.shuffle(student_ids)
    n_test = int(len(student_ids) * test_frac)
    test_students = set(student_ids[:n_test])

    train_df = df[~df["id_student"].isin(test_students)].reset_index(drop=True)
    test_df = df[df["id_student"].isin(test_students)].reset_index(drop=True)
    return train_df, test_df


def build_design_matrix(df: pd.DataFrame, feat_mean, feat_std):
    """Normalize features and prepend a bias column of 1s."""
    X = ((df[FEATURES] - feat_mean) / feat_std).values
    bias = np.ones((len(df), 1))
    return np.hstack([bias, X])


def predict_p(w, X, delta_t):
    """h = 2^(w.x); p(t) = 2^(-delta_t/h). h is clipped to avoid overflow/underflow."""
    log_h = X @ w
    h = np.clip(2 ** log_h, 1e-3, 1e6)
    p = 2 ** (-delta_t / h)
    return np.clip(p, 1e-6, 1 - 1e-6), h


def train_hlr(X_train, delta_t_train, y_train, lr=0.08, epochs=900, l2=0.001):
    """Gradient descent on squared error between predicted p(t) and the continuous target."""
    w = np.zeros(X_train.shape[1])
    for epoch in range(epochs):
        p_hat, h = predict_p(w, X_train, delta_t_train)
        error = p_hat - y_train

        # Chain rule: d(loss)/dw = d(loss)/dp * dp/d(log_h) * d(log_h)/dw
        dp_dlogh = p_hat * np.log(2) * (delta_t_train / h)
        grad = (2 * error * dp_dlogh)[:, None] * X_train
        grad = grad.mean(axis=0) + l2 * w

        w -= lr * grad

        if epoch % 150 == 0 or epoch == epochs - 1:
            loss = np.mean(error ** 2)
            print(f"  epoch {epoch:4d}  |  mean squared error = {loss:.4f}")
    return w


def main():
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows, {df['id_student'].nunique()} students")

    train_df, test_df = train_test_split_by_student(df)
    print(f"Train: {len(train_df)} rows ({train_df['id_student'].nunique()} students)")
    print(f"Test:  {len(test_df)} rows ({test_df['id_student'].nunique()} students)")

    feat_mean = train_df[FEATURES].mean()
    feat_std = train_df[FEATURES].std() + 1e-8

    X_train = build_design_matrix(train_df, feat_mean, feat_std)
    delta_t_train = train_df["delta_t"].values
    y_train = train_df["target"].values

    X_test = build_design_matrix(test_df, feat_mean, feat_std)
    delta_t_test = test_df["delta_t"].values
    y_test = test_df["target"].values

    print("\nTraining HLR via gradient descent...")
    w = train_hlr(X_train, delta_t_train, y_train)

    p_test, _ = predict_p(w, X_test, delta_t_test)
    mae = mean_absolute_error(y_test, p_test)
    # AUC still needs a binary reference; recall_label works fine for this one metric.
    auc = roc_auc_score(test_df["recall_label"].values, p_test)

    print(f"\nHeld-out test performance:")
    print(f"  MAE (predicted vs actual target): {mae:.4f}")
    print(f"  AUC (ranking recall vs non-recall): {auc:.4f}")

    np.savez(MODEL_PATH, w=w, feat_mean=feat_mean.values, feat_std=feat_std.values,
             features=FEATURES)
    print(f"\nSaved trained model -> {MODEL_PATH.name}")


if __name__ == "__main__":
    main()