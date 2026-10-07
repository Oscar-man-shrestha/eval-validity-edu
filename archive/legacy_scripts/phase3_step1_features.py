"""
Phase 3 - Step 1: Feature Engineering for HLR
NeuroTrace-DAG

Turns interactions_BBB.csv (one row per assessment attempt) into a feature
table HLR can train on. For every row, we add three columns describing
what we knew about that student's history BEFORE that attempt happened -
never information from the attempt itself, or that would leak the answer
into the input.

Target: continuous score (rescaled to 0-1) instead of binary recall_label,
to avoid the 96.7% / 3.3% class imbalance in recall_label.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "interactions_BBB.csv"
OUTPUT_PATH = ROOT / "interactions_features.csv"


def load_interactions(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the CSV, sorted chronologically within each student."""
    df = pd.read_csv(path)
    df = df.sort_values(["id_student", "assessment_date"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_attempts_so_far, past_success_rate, days_since_first_exposure,
    and target (score/100, clipped) - computed per student, using only
    rows strictly BEFORE the current one.
    """
    df = df.copy()
    grouped = df.groupby("id_student")

    # How many prior attempts: cumcount() counts rows seen so far within
    # each student (0-indexed) - exactly "how many rows came before this
    # one", which is 0 on a student's first row.
    df["n_attempts_so_far"] = grouped.cumcount()

    # past_success_rate: running mean of PRIOR recall_label values only.
    # shift(1) inside the transform is the label-leakage guard - it drops
    # the current row out of its own average before expanding().mean()
    # computes the running mean of everything that came before it.
    # transform() keeps one groupby pass and returns a Series already
    # aligned to df's original row order/index.
    df["past_success_rate"] = grouped["recall_label"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["past_success_rate"] = df["past_success_rate"].fillna(0.5)

    # days_since_first_exposure: this row's date minus that student's
    # earliest date, per student.
    first_date = grouped["assessment_date"].transform("min")
    df["days_since_first_exposure"] = df["assessment_date"] - first_date

    # Continuous target instead of binary recall_label.
    df["target"] = (df["score"] / 100.0).clip(0.001, 0.999)

    return df


def main():
    df = load_interactions()
    print(f"Loaded {len(df)} rows, {df['id_student'].nunique()} students")

    df = build_features(df)

    assert df["n_attempts_so_far"].min() == 0, "First attempt should have 0 prior attempts"
    assert df["days_since_first_exposure"].min() == 0, "First attempt should have 0 days elapsed"
    assert df["target"].between(0, 1).all(), "target must be in [0, 1]"
    print("Sanity checks passed.")

    print(df[["id_student", "assessment_date", "delta_t", "score", "recall_label",
               "n_attempts_so_far", "past_success_rate",
               "days_since_first_exposure", "target"]].head(10).to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH.name}: {len(df)} rows")


if __name__ == "__main__":
    main()