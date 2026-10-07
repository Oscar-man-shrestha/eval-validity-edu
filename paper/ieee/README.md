# IEEE A4 conference draft (advisor / pre-journal)

Built from the IEEE **Conference-template-A4** Word template structure.

| File | Use |
| --- | --- |
| `NeuroTrace-DAG_IEEE_A4_Conference.docx` | **Open this for ma’am** (Word) |
| `conference_paper.tex` | IEEEtran LaTeX (needs TeX Live + `IEEEtran.cls`) |
| `build_conference_docx.py` | Regenerate the `.docx` from JSON numbers |
| `Conference-template-A4.doc` | Original IEEE template copy |

```bash
python3 paper/ieee/build_conference_docx.py
```

**Honesty gate:** Table II confirmatory H-rev cells stay `pending_confirmatory` until OSF/Zenodo + one `--touch-test`. Exploratory Junyi ranking numbers are filled and labeled exploratory.

Fill author affiliation / email placeholders before sharing.
