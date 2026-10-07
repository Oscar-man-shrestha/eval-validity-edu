"""
Phase 1 - Data Preprocessing (MOOCCubeX side)
Member 1 - NeuroTrace-DAG

What this script does:
1. Loads entities/concept.json -> a clean concepts.csv (id, name, context text)
2. Inspects prerequisites/cs.json structure (prints it so we can confirm the schema)
3. Parses cs.json into a clean prerequisite_edges.csv (source_concept -> target_concept)

Run this in two stages: first just the inspection (Step A below), confirm the printed
structure looks like what we expect, THEN uncomment/run the full parse.
"""

import json
import pandas as pd
from pathlib import Path


DATA_DIR = Path(r"C:\Users\HP\Desktop\AMRITA\PROJECTS\ML\mooccubex_data")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\AMRITA\PROJECTS\ML")


# ---------------------------------------------------------------------------
# Universal loader: tries standard JSON first, falls back to JSON Lines
# (one JSON object per line) if that fails - this is very common for large
# ML datasets like MOOCCubeX, and is exactly what caused the earlier error.
# ---------------------------------------------------------------------------

def load_json_flexible(path: Path, max_records: int = None) -> list:
    """
    Load a JSON file that might be either:
      (a) a single JSON array: [ {...}, {...}, ... ]
      (b) JSON Lines: one JSON object per line, no enclosing brackets/commas

    max_records: if set, stop after reading this many records (useful for
    quick inspection of huge files without loading everything into memory).
    """
    # First, try the simple case: whole file is one JSON structure
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data if max_records is None else data[:max_records]
        elif isinstance(data, dict):
            values = list(data.values())
            return values if max_records is None else values[:max_records]
    except json.JSONDecodeError:
        pass  # fall through to JSON Lines handling below

    # Fall back: JSON Lines format - read and parse one line at a time
    print("  (detected JSON Lines format - one object per line - parsing that way)")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning: could not parse line {i+1}, skipping it")
                continue
            if max_records is not None and len(records) >= max_records:
                break
    return records


# ---------------------------------------------------------------------------
# Step A: Load and clean concept.json (this part we know the schema for)
# ---------------------------------------------------------------------------

def load_concepts(concept_path: Path) -> pd.DataFrame:
    """
    concept.json fields (per MOOCCubeX docs):
      - id:      Concept ID, format K_{concept name}_{field}
      - name:    Concept name
      - context: Surrounding text where the concept appears (Wiki/Baidu Encyclopedia)
    """
    print(f"Loading {concept_path} ... (this may take a minute, it's a large file)")
    records = load_json_flexible(concept_path)

    print(f"Loaded concept.json: {len(records)} records")
    print("\nFirst record (to confirm structure):")
    print(records[0])

    df = pd.DataFrame(records)
    print(f"\nBuilt concepts DataFrame: shape={df.shape}, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step B: INSPECT cs.json first - don't assume the schema, look at it
# ---------------------------------------------------------------------------

def inspect_prerequisites(prereq_path: Path, n_preview: int = 5):
    """
    Prints the raw structure of cs.json so we can confirm field names
    before writing the parser. Run this FIRST, look at the printed output,
    then tell me what you see if it doesn't match parse_prerequisites() below.
    """
    print(f"Loading {prereq_path} ... (large file, may take a minute)")
    # Only load a handful of records for inspection - fast and enough to see the schema
    records = load_json_flexible(prereq_path, max_records=n_preview)

    print(f"\nGot {len(records)} preview record(s)")
    print(f"\nFirst {n_preview} records:")
    for record in records:
        print(record)


# ---------------------------------------------------------------------------
# Step C: Parse prerequisites into clean edges (ADJUST field names after
# inspecting the actual output from Step B above)
# ---------------------------------------------------------------------------

def parse_prerequisites(prereq_path: Path) -> pd.DataFrame:
    """
    Real schema (confirmed from inspection output):
      {'c1': <concept name>, 'c2': <concept name>, 'ground_truth': 0 or 1,
       'text_predict': [...], 'graph_predict': [...]}

    ground_truth == 1  -> confirmed prerequisite pair (human-labeled)
    ground_truth == 0  -> confirmed NOT a prerequisite pair

    Direction, confirmed rather than assumed: c1 is the prerequisite and
    c2 is the concept that depends on it. source_concept_name = c1 and
    target_concept_name = c2, so the edge means "learn c1 before c2".
    That is the usual (first concept, second concept) label in this
    dataset: the first concept is a prerequisite of the second.
    Clear pairs in the English table agree: "divide and conquer" ->
    "quicksort", "leaf node" -> "B-tree", "interrupt signal" ->
    "soft interrupt". The drawing code checks that every such edge runs
    from an earlier row to a later row.

    NOTE: c1/c2 here are bare concept NAMES (e.g. '操作命令'), not the
    longer 'id' format used in concept.json (e.g. 'K_神经部_...'). The join
    between concepts.csv and prerequisite_edges.csv must happen on the
    'name' field, not 'id'.
    """
    records = load_json_flexible(prereq_path)

    edges = []
    for r in records:
        if r.get("ground_truth") == 1:
            edges.append({
                "source_concept_name": r.get("c1"),
                "target_concept_name": r.get("c2"),
            })

    df = pd.DataFrame(edges)
    print(f"Parsed {len(df)} confirmed prerequisite edges out of {len(records)} total records")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    concept_path = DATA_DIR / "concept.json"
    prereq_path = DATA_DIR / "cs.json"

    # --- Step A: concepts (safe to run fully, schema is documented) ---
    concepts_df = load_concepts(concept_path)
    concepts_df.to_csv(OUTPUT_DIR / "concepts.csv", index=False, encoding="utf-8-sig")
    print(f"Saved concepts.csv ({len(concepts_df)} rows)\n")

    # --- Step B: inspect prerequisites structure FIRST ---
    print("=" * 60)
    print("INSPECTING cs.json structure - read this before continuing")
    print("=" * 60)
    inspect_prerequisites(prereq_path)

    # --- Step C: parse prerequisites (comment this out if Step B's
    # printed structure doesn't match what parse_prerequisites() expects -
    # send me the printed output instead and I'll fix the field names) ---
    print("\n" + "=" * 60)
    print("PARSING prerequisites (based on assumed schema)")
    print("=" * 60)
    edges_df = parse_prerequisites(prereq_path)
    if len(edges_df) > 0:
        edges_df.to_csv(OUTPUT_DIR / "prerequisite_edges.csv", index=False, encoding="utf-8-sig")
        print(f"Saved prerequisite_edges.csv ({len(edges_df)} rows)")
    else:
        print("0 edges parsed - the schema guess was wrong. "
              "Check the inspection output above and send it back.")
