# AI usage log

Record of every interaction with AI assistants that shaped the code or the reports. It feeds the
"Use of AI assistants" section of both reports and the AI Error Bounty sheets (Appendix A of the
January reports, see `bounty-guideline-example`).

**Assistant:** Claude (Anthropic), Claude desktop app with read/write access to the project folder.
**Workflow:** Claude writes the files; the author runs the notebooks in VS Code; Claude reads the
outputs from `results/`, checks them and explains them; the author verifies and decides.

Entry format:

- **Stage / date**
- **Request** (summary of the prompt)
- **What the AI produced**
- **Verification** (how it was checked)
- **Corrections** (what was wrong and how it was fixed)
- **Bounty candidate** (yes / no, and why)

---

## Stage 0 — Repository and environment (2026-10-23)

- **Request:** create the repository structure, environment files, README, `.gitignore` and
  `src/config.py` for a nested-CV project on the TCGA glioma dataset, following the decisions
  listed in `docs/DECISIONS.md`.
- **What the AI produced:** folder structure, `environment.yml` (conda, Python 3.11),
  `requirements.txt` with pinned versions, `src/config.py`, `src/check_env.py`, README,
  `.gitignore`, `.gitattributes`, `docs/DECISIONS.md`, this log.
- **Verification:**
  - The pinned package set was installed in a clean Python 3.11 environment by the AI and a
    `GridSearchCV` + `XGBClassifier` smoke test was run, to rule out the known incompatibility
    between some XGBoost and scikit-learn versions.
  - The dataset citation (authors, year, DOI 10.24432/C5R62J, CC BY 4.0, 839 instances) was
    checked on the UCI dataset page before entering `references.bib`.
  - The author runs `python -m src.check_env` on her computer; the result is stored in
    `results/logs/environment.json`.
- **Corrections:**
  - The AI first wrote "Code: MIT" in the README license section without being asked; removed,
    because the license of the code is the author's decision.
  - In `docs/DECISIONS.md` the AI first cited a student report (Engvall) as the source for a
    methodological choice and a course chapter it had not checked; both replaced by sources it had
    actually read (rule: student reports are format references only).
- **Bounty candidate:** no (minor, caught by the AI itself before delivery).
