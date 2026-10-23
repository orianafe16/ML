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

## Stage 0 (continued) — GitHub push (2026-10-23)

- **Issue:** `git push` returned 403 because Windows had cached the credentials of another GitHub
  account; after fixing it, the push was rejected (non-fast-forward) because the commit had been
  amended after an earlier successful push.
- **AI advice:** put the account name in the remote URL; check the commit author; before
  overwriting, inspect the remote (`git fetch`, `git log origin/main`) and compare contents
  (`git diff origin/main main --stat` was empty), then use `--force-with-lease` instead of `--force`.
- **Verification:** the author ran the commands; `git log` shows the correct author and the remote
  holds a single commit.
- **Bounty candidate:** no.

## Stage 1 — Data preparation (2026-10-23)

- **Request:** notebook `01_data_preparation` and `src/data.py`: load, check quality, clean and
  encode the data, avoiding leakage.
- **What the AI produced:** `src/data.py`, `src/reporting.py` (LaTeX tables and JSON summaries),
  `notebooks/01_data_preparation.ipynb`.
- **Verification:**
  - The AI reverse-engineered the UCI cleaning (which rows were removed and how age was converted)
    and the notebook asserts that the rebuilt dataset equals the UCI file in every value.
  - The AI executed the notebook in a clean copy of the repository before delivering it.
  - The author runs it on her computer; the AI reads `results/logs/01_data_preparation.json`.
- **Corrections:** none so far.
- **Bounty candidate:** no.
