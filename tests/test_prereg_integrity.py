"""Dependency-light guard for files frozen by the preregistration."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "outputs_junyi" / "phases" / "prereg_v1_locked_hashes.json"


def test_prereg_locked_file_hashes():
    lock = json.loads(LOCK.read_text())
    for row in lock["files"]:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == row["sha256"], f"changed: {row['path']}"
