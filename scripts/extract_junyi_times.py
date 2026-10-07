"""Stream junyi.rar ProblemLog → compact timed interactions.

Writes data/junyi_raw/timed_interactions.npz with columns:
  user_id int32, concept int16, t_us int64, correct int8
Only exercises in the ktbd 835-vertex map. Caps at MAX_KEEP rows
so a nearly-full disk still finishes.

Run: python3 scripts/extract_junyi_times.py
"""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAR = ROOT / "data" / "junyi_raw" / "junyi.rar"
VERT = ROOT / "data" / "junyi_ktbd" / "vertex_id2idx"
OUT = ROOT / "data" / "junyi_raw" / "timed_interactions.npz"
UNAR = "/opt/homebrew/bin/unar"
MAX_KEEP = 8_000_000


def load_name_to_idx() -> dict[str, int]:
    mapping = {}
    for line in VERT.read_text().splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit(",", 1)
        mapping[name.strip()] = int(idx)
    return mapping


def main() -> None:
    if OUT.exists():
        z = np.load(OUT)
        print("exists", OUT, {k: z[k].shape for k in z.files})
        return
    if not RAR.exists():
        raise SystemExit(f"missing {RAR}")

    name_to_idx = load_name_to_idx()
    print("vertices", len(name_to_idx), "MAX_KEEP", MAX_KEEP)

    users = np.empty(MAX_KEEP, dtype=np.int32)
    concepts = np.empty(MAX_KEEP, dtype=np.int16)
    times = np.empty(MAX_KEEP, dtype=np.int64)
    corrects = np.empty(MAX_KEEP, dtype=np.int8)

    proc = subprocess.Popen(
        [UNAR, "-o", "-", "-q", str(RAR), "junyi_ProblemLog_original.csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(text)

    kept = scanned = 0
    for row in reader:
        scanned += 1
        idx = name_to_idx.get(row.get("exercise") or "")
        if idx is None:
            continue
        try:
            t = int(float(row["time_done"]))
            c = 1 if str(row["correct"]).lower() in {"true", "1"} else 0
            u = int(row["user_id"])
        except (KeyError, ValueError, TypeError):
            continue
        users[kept] = u
        concepts[kept] = idx
        times[kept] = t
        corrects[kept] = c
        kept += 1
        if kept % 500_000 == 0:
            print(f"kept={kept:,} scanned={scanned:,}", flush=True)
        if kept >= MAX_KEEP:
            print("hit MAX_KEEP, stopping stream", flush=True)
            break

    try:
        proc.kill()
    except Exception:
        pass
    text.close()
    print(f"done kept={kept:,} scanned={scanned:,}")
    if kept == 0:
        raise SystemExit("no rows matched vertex map")

    users = users[:kept]
    concepts = concepts[:kept]
    times = times[:kept]
    corrects = corrects[:kept]
    order = np.lexsort((times, users))
    np.savez_compressed(
        OUT,
        user_id=users[order],
        concept=concepts[order],
        t_us=times[order],
        correct=corrects[order],
    )
    print("wrote", OUT, "MB", round(OUT.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
