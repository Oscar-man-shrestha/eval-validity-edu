#!/usr/bin/env python3
"""Download / verify NeuroTrace study datasets. Does not commit data files."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "data/junyi_ktbd/train.json": "9a50d9bdbfbc19cd4682cc0303fb0ff5bdb017a955c10d6f1922a60636d494c1",
    "data/junyi_ktbd/test.json": "c55f0714d9479d35afb25564ef2faf57c17ca46d944a0e509cb3f26780934973",
    "data/junyi_ktbd/vertex_id2idx": "553271b2804315b879adcd04e7696db61477b0b53fd9033be3e1413136bf4c2c",
    "data/junyi_ktbd/prerequisite.json": "cd3eaf22c22ebbeb3165068233b7fe16962d1910573f5546628f7612cef435b0",
    "data/assistments2009/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv": (
        "1aa296e00b6c88c4d6fad4ca2ae4866484d9fe5484f38f5c8c94dfc49f045e08"
    ),
    "data/xes3g5m/XES3G5M.zip": "62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73",
    "data/junyi_raw/timed_interactions.npz": "8bf6db6636b18631b5cea439c5b44a608431cf0a1b4876d05b4fcef71e006164",
    "data/junyi_raw/junyi/junyi_Exercise_table.csv": "13292cc2c809985abbacedaba434e39aeb02a7fcb2a32f17b60c4d3f4d769546",
}

URLS = {
    "junyi_ktbd": "http://base.ustc.edu.cn/data/ktbd/junyi/",
    "assistments": "EduData USTC ASSISTments2009 skill-builder corrected mirror",
    "xes3g5m": "XES3G5M NeurIPS 2023 release",
    "junyi_timed": "DataShop dataset 1198 → junyi.rar → timed_interactions.npz (local extract)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("Expected sources:")
    for k, u in URLS.items():
        print(f"  - {k}: {u}")
    print("\nVerifying local SHA-256 against data cards…")
    missing = 0
    bad = 0
    for rel, expect in EXPECTED.items():
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING {rel}")
            missing += 1
            continue
        got = sha256(path)
        ok = got == expect
        print(("OK" if ok else "MISMATCH"), rel)
        if not ok:
            print(f"  expected {expect}")
            print(f"  got      {got}")
            bad += 1
    if missing or bad:
        print(f"\nFailed: missing={missing} mismatch={bad}", file=sys.stderr)
        return 1
    print("\nAll expected local files match data-card hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
