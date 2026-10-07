"""
Phase 4 - OULAD <-> MOOCCubeX Interface (Option A: decoupled, no ID mapping)
NeuroTrace-DAG

DECISION: OULAD's module codes (e.g. "BBB") are anonymized - there is no
real subject-matter description to verify a concept-level mapping against.
Building a fake one-to-one mapping to MOOCCubeX's 287 concepts would be an
unverifiable claim, so we don't.

INSTEAD: the two systems stay decoupled at the ID level and combine only
through the Phase 5 scoring formula:

    Score(r) = w1 * SemanticRelevance(query, r)
             + w2 * (1 - general_forgetting_signal(student))
             + w3 * CurriculumProgress(r)

The key change from the original plan: the memory term is no longer
per-(student, concept) - Phase 3 has no way to know that, given the
placeholder concept. It is a per-STUDENT general signal (their current
retention level in the module they're enrolled in), applied identically
to every candidate concept r for that student.

This is a real, load-bearing simplification. Document it as such:
"the memory-retention term reflects the student's general forgetting
level in their enrolled OULAD module, not concept-specific forgetting
for the MOOCCubeX topic being recommended."
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATES_PATH = ROOT / "phase3_student_states.csv"

DEFAULT_REVIEW_GAP_DAYS = 7  # which p_recall_Xd column to use as "current" retention


def load_student_states(path: Path = STATES_PATH) -> pd.DataFrame:
    return pd.read_csv(path).set_index("id_student")


def general_forgetting_signal(states: pd.DataFrame, student_id: int,
                               gap_days: int = DEFAULT_REVIEW_GAP_DAYS) -> float:
    """
    Returns a single 0-1 number: this student's estimated recall
    probability in their OULAD module, at a `gap_days`-day review gap.

    This is deliberately NOT concept-specific - see module docstring.
    Falls back to 0.5 (neutral) for a student_id with no Phase 3 state,
    e.g. one who was in the held-out test split and has no logged history.
    """
    col = f"p_recall_{gap_days}d"
    if student_id not in states.index:
        return 0.5
    return float(states.loc[student_id, col])


def main():
    states = load_student_states()
    print(f"Loaded {len(states)} student states")

    sample_ids = states.index[:5].tolist()
    print(f"\nGeneral forgetting signal (p_recall at {DEFAULT_REVIEW_GAP_DAYS}-day gap) "
          f"for 5 sample students:")
    for sid in sample_ids:
        signal = general_forgetting_signal(states, sid)
        print(f"  student {sid}: {signal:.3f}  "
              f"(memory term w2*(1-p) = {1 - signal:.3f})")

    # Fallback check for an unknown student
    fallback = general_forgetting_signal(states, student_id=-1)
    print(f"\nUnknown student fallback: {fallback} (neutral default, as expected)")


if __name__ == "__main__":
    main()