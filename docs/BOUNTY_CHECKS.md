# AI Error Bounty — reproduction of each check

Each sheet in Appendix A of the PRDL (P1–P4) and MLLB (C1–C3) reports ends with a **Reproduction**
field. Run the step below, write what you observed in that field (replacing the red `[verify]` tag),
and add any new error you find while reviewing the reports as a new sheet in the same section.

Python snippets: run them from the repository root in the `glioma-ml` environment
(`python` then paste, or a scratch notebook in `notebooks/`).

## PRDL (AI scientific assistants)

**P1 — Wrong DOI.** Open https://doi.org/10.1093/neuonc/noad108 and https://doi.org/10.1093/neuonc/noad100
in a browser and note the title of each. Expected: the first is an article on CAR-T cells in
glioblastoma; the second is the EANO guideline on molecular diagnostic tools (Sahm et al., 2023).
`python report/check_references.py` also prints `OK sahm2023` with the corrected DOI.

**P2 — Same wrong authors from two tools.** Open `report/literature/21.pdf` and read the title and author
list on the first page. Expected: Sánchez-Marqués et al., 2024 (Scientific Reports), not Garbulowski and
Komorowski.

**P3 — Statements broader than their quotes.** Read `19.pdf`, page 5, Section 2.1 ("The default values
were assigned as the corresponding parameter values...") and `42.pdf`, page 3, Results and Table 1
("811 samples ... consistent classification across both methods, while 134 cases ... changed"). Check
that the first is about hyperparameters (not nested CV) and that the two methods of the second are the
authors' 2016- and 2021-based pipelines, not the original TCGA labels.

**P4 — Wrong interpretation of the regression predictions.** Run `notebooks/04_regression.ipynb` and look
at Section 6: the printed correlation of the predicted age with the grade and with the IDH status, and
the table of mean predicted age by grade × IDH. Expected: the correlation with IDH is stronger, and
IDH-mutant GBM patients are predicted about 20 years younger than IDH-wildtype GBM patients, so the
predictions follow IDH, not the grade. Compare with the proportion mediated in
`notebooks/08_causal.ipynb`, Section 5.

## MLLB (AI code assistant)

**C1 — Unseeded solver.** Removes the seed of logistic regression and runs its nested CV twice:

```python
from src import data, models, evaluation as ev
df = data.load_processed()
X, y = df[data.feature_columns("classification")], df["Grade"].values
pipe, grid = models.classification_models(list(X.columns))["Logistic regression"]
pipe.set_params(model__random_state=None)          # the original, faulty version
runs = [ev.nested_cv(pipe, grid, X, y, "classification", ev.CLASSIFICATION_SCORING,
                     ev.classification_metrics, importance=False) for _ in range(2)]
print("max difference in out-of-fold probabilities:", abs(runs[0]["oof"] - runs[1]["oof"]).max())
print(runs[0]["folds"][["penalty", "C"]].compare(runs[1]["folds"][["penalty", "C"]]))
```

Expected: a non-zero difference in the out-of-fold probabilities (in a test run, up to about 0.002);
the chosen `penalty` and `C` may also differ in some folds (the second print is empty when they do
not). With `random_state=42` (the current code) the difference is exactly 0.

**C2 — Silent NaN (leverage 1).**

```python
import pandas as pd, statsmodels.api as sm
from src import config, data
df = data.load_processed()
race = pd.get_dummies(df["Race"].map({0: "White", 1: "Black", 2: "Asian", 3: "Native"})).drop(columns="White").astype(float)
X = sm.add_constant(pd.concat([df[["Gender"]], race, df[config.GENE_COLS]], axis=1).astype(float))
infl = sm.OLS(df["Age_at_diagnosis"], X).fit().get_influence()
print("max leverage:", infl.hat_matrix_diag.max())
print("Cook's distance of that patient:", infl.cooks_distance[0][infl.hat_matrix_diag.argmax()])
```

Expected: maximum leverage 1.0 (the only American Indian or Alaska Native patient) and a NaN Cook's
distance for that patient.

**C3 — Tree figure.**

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier
from src import data
df = data.load_processed()
X, y = df[data.feature_columns("classification")], df["Grade"].values
tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
plot_tree(tree, feature_names=list(X.columns), class_names=["LGG", "GBM"], filled=True, precision=0)
plt.show()
print(export_text(tree, feature_names=list(X.columns), decimals=1))
```

Expected: the figure reads "IDH1 <= 0.0" while `export_text` gives "IDH1 <= 0.5", and the LGG leaves
are painted orange (the project uses blue for LGG). Compare with `results/figures/clf_decision_tree.png`.
