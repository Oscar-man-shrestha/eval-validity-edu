"""
Phase 1 - Data Preprocessing (OULAD side)
Member 1 - NeuroTrace-DAG

What this script does:
1. Loads the raw OULAD CSV files.
2. Builds a per-student, per-module timeline of "interaction events" (VLE clicks)
   and "outcome events" (assessment submissions).
3. Computes delta_t (days since the student's last interaction with that module)
   for every assessment attempt.
4. Computes a binary recall label y (did they pass or fail that assessment).
5. Saves a clean interactions.csv - this is the direct input to Phase 3 (HLR training).

Note on granularity: OULAD tracks at the *module* level (e.g. "BBB"), not fine-grained
"concepts" (e.g. "loops", "recursion"). Until this is joined with MOOCCubeX's concept-level
data, we treat each module as a single "concept" as a placeholder. This is a known
limitation to document in your report - flag it explicitly, don't hide it.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: Load raw files
# ---------------------------------------------------------------------------

def load_oulad_tables(data_dir: str) -> dict:
    """Load the raw OULAD CSVs into a dict of DataFrames."""
    data_dir = Path(data_dir)

    tables = {
        "studentInfo": pd.read_csv(data_dir / "studentInfo.csv"),
        "studentAssessment": pd.read_csv(data_dir / "studentAssessment.csv"),
        "studentVle": pd.read_csv(data_dir / "studentVle.csv"),
        "assessments": pd.read_csv(data_dir / "assessments.csv"),
        "courses": pd.read_csv(data_dir / "courses.csv"),
    }

    for name, df in tables.items():
        print(f"{name:<20} shape={df.shape}")

    return tables


# ---------------------------------------------------------------------------
# Step 2: Filter to one module (recommended starting point - keeps things
# fast and debuggable before scaling to all seven modules)
# ---------------------------------------------------------------------------

def filter_to_module(tables: dict, code_module: str = "BBB") -> dict:
    """Keep only rows belonging to one course module, e.g. 'BBB'."""
    filtered = {}
    filtered["studentInfo"] = tables["studentInfo"][
        tables["studentInfo"]["code_module"] == code_module
    ].copy()
    filtered["studentAssessment"] = tables["studentAssessment"].copy()  # filtered after merge
    filtered["studentVle"] = tables["studentVle"][
        tables["studentVle"]["code_module"] == code_module
    ].copy()
    filtered["assessments"] = tables["assessments"][
        tables["assessments"]["code_module"] == code_module
    ].copy()

    print(f"\nFiltered to module '{code_module}':")
    print(f"  studentInfo rows: {len(filtered['studentInfo'])}")
    print(f"  studentVle rows:  {len(filtered['studentVle'])}")
    print(f"  assessments rows: {len(filtered['assessments'])}")

    return filtered


# ---------------------------------------------------------------------------
# Step 3: Build the per-student VLE interaction timeline
# ---------------------------------------------------------------------------

def build_vle_timeline(student_vle: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw click events into one row per (student, day) they interacted
    with the module - we only need "did they engage that day", not click counts,
    for computing delta_t.
    """
    timeline = (
        student_vle.groupby(["id_student", "code_module", "code_presentation", "date"])
        .size()
        .reset_index(name="click_count")
        .sort_values(["id_student", "date"])
    )
    return timeline


# ---------------------------------------------------------------------------
# Step 4: Compute delta_t and recall label per assessment attempt
# ---------------------------------------------------------------------------

def compute_interaction_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    vle_timeline: pd.DataFrame,
    pass_threshold: float = 40.0,
) -> pd.DataFrame:
    """
    For every assessment a student submitted, find:
      - delta_t: days since their last VLE interaction with that module
                 before this assessment date
      - recall_label: 1 if score >= pass_threshold else 0

    This table is the direct training input for Phase 3 (HLR).
    """
    # Attach module/presentation/assessment-date info to each submission
    merged = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
        suffixes=("", "_assessment"),
    )
    merged = merged.rename(columns={"date": "assessment_date"})

    records = []
    # Group so we only loop per student-module (fast enough for one module at a time)
    for (student_id, module, presentation), group in merged.groupby(
        ["id_student", "code_module", "code_presentation"]
    ):
        # This student's VLE interaction days for this module
        student_vle_days = vle_timeline[
            (vle_timeline["id_student"] == student_id)
            & (vle_timeline["code_module"] == module)
            & (vle_timeline["code_presentation"] == presentation)
        ]["date"].sort_values().values

        for _, row in group.sort_values("assessment_date").iterrows():
            assessment_day = row["assessment_date"]

            if pd.isna(assessment_day):
                continue  # skip rows with missing assessment date

            # Find the most recent VLE interaction strictly before this assessment
            prior_days = student_vle_days[student_vle_days < assessment_day]
            if len(prior_days) > 0:
                last_interaction_day = prior_days[-1]
                delta_t = assessment_day - last_interaction_day
            else:
                delta_t = np.nan  # no prior interaction logged - handle in cleaning step

            score = row["score"]
            recall_label = int(score >= pass_threshold) if pd.notna(score) else np.nan

            records.append(
                {
                    "id_student": student_id,
                    "code_module": module,
                    "code_presentation": presentation,
                    "concept": module,  # placeholder until MOOCCubeX mapping is added
                    "assessment_date": assessment_day,
                    "delta_t": delta_t,
                    "score": score,
                    "recall_label": recall_label,
                }
            )

    result = pd.DataFrame(records)
    return result


# ---------------------------------------------------------------------------
# Step 5: Clean and save
# ---------------------------------------------------------------------------

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows we can't use for training and report how much was dropped."""
    before = len(df)
    df = df.dropna(subset=["delta_t", "recall_label"])
    after = len(df)
    print(f"\nDropped {before - after} rows with missing delta_t/recall_label "
          f"({before} -> {after})")
    return df


def main(data_dir: str, code_module: str = "BBB", output_path: str = "interactions.csv"):
    tables = load_oulad_tables(data_dir)
    filtered = filter_to_module(tables, code_module=code_module)

    vle_timeline = build_vle_timeline(filtered["studentVle"])
    interactions = compute_interaction_features(
        filtered["studentAssessment"], filtered["assessments"], vle_timeline
    )
    interactions = clean_interactions(interactions)

    print("\nSample of final interactions table:")
    print(interactions.head(10))

    print("\nRecall label distribution:")
    print(interactions["recall_label"].value_counts(normalize=True))

    print("\nDelta_t summary stats (days since last interaction):")
    print(interactions["delta_t"].describe())

    interactions.to_csv(output_path, index=False)
    print(f"\nSaved {len(interactions)} rows to {output_path}")

    return interactions


if __name__ == "__main__":
    # Adjust this path to wherever you extracted the OULAD CSVs
    DATA_DIR = "./oulad_data"
    main(DATA_DIR, code_module="BBB", output_path="interactions_BBB.csv")
