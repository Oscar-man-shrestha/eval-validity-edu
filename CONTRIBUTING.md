# Contributing

Issues and pull requests about documentation, reproducibility, and implementation are welcome. Please describe the affected dataset or pipeline stage, the expected behavior, and how you checked the change.

The [`prereg-v1`](https://github.com/Oscar-man-shrestha/eval-validity-edu/releases/tag/prereg-v1) protocol is frozen. Changes to its analysis rules, locked files, or confirmatory reporting need a documented rationale in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md). Do not run the one-time `--touch-test` holdout for routine development or edit generated H-rev claim cells by hand. See [`docs/PREREGISTRATION_REPLICATION.md`](docs/PREREGISTRATION_REPLICATION.md) for the reproduction sequence.

Before submitting a pull request, run the checks relevant to your change. For protocol-related edits, start with:

```bash
PYTHONPATH=. python3 -m pytest tests/test_freeze_partition_replication.py -q -k 'not winner_retrain'
```

Document any missing dataset or dependency that prevented a check from running. Dataset licences and citation requirements are listed in [`docs/data_cards/`](docs/data_cards/); do not commit bulk source data or personal learner information.
