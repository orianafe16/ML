"""Generate report/common/numbers.tex: one LaTeX macro per number quoted in the report text.

Every number written in the PRDL and MLLB reports is a macro defined here, read from the JSON
summaries in results/logs/. If a notebook is re-run and a result changes, re-running this script
updates every sentence of both reports. No number is typed by hand.

Run from the repository root:  python report/make_numbers.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "results" / "logs"
OUT = ROOT / "report" / "common" / "numbers.tex"


def load(name):
    return json.loads((LOGS / f"{name}.json").read_text(encoding="utf-8"))


macros = {}


def num(name, value, fmt="{:.3f}"):
    """Register \\name -> formatted value. Names must contain letters only (LaTeX rule)."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"Invalid macro name: {name}")
    if name in macros:
        raise ValueError(f"Duplicated macro: {name}")
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def row(table, key, value, col="Model"):
    return next(r for r in table if r[col] == value)[key] if isinstance(table, list) else table[value][key]


def ci(r, metric, d=3):
    return f"{r[metric]:.{d}f} [{r[metric + ' CI low']:.{d}f}, {r[metric + ' CI high']:.{d}f}]"


# ---------------------------------------------------------------------------
# 01 Data preparation
# ---------------------------------------------------------------------------
d = load("01_data_preparation")
num("NRaw", d["n_raw"], "{}")
num("NClean", d["n_clean"], "{}")
num("NRemoved", d["n_removed"], "{}")
steps = {s["Step"]: s["Removed"] for s in d["removal_steps"]}
num("NRemovedAllMissing", steps["All clinical fields missing"], "{}")
num("NRemovedRace", steps["Race not reported"], "{}")
num("NRemovedAge", steps["Age at diagnosis missing"], "{}")
num("NRemovedLGG", d["removed_by_grade"]["LGG"], "{}")
num("NRemovedGBM", d["removed_by_grade"]["GBM"], "{}")
num("NLGG", d["grade_counts"]["LGG"], "{}")
num("NGBM", d["grade_counts"]["GBM"], "{}")
num("PctGBM", pct(d["grade_gbm_proportion"]))
num("PctLGG", pct(1 - d["grade_gbm_proportion"]))
num("NWhite", d["race_counts"]["0"], "{}")
num("NBlack", d["race_counts"]["1"], "{}")
num("NAsian", d["race_counts"]["2"], "{}")
num("NNative", d["race_counts"]["3"], "{}")
num("PctWhite", pct(d["race_counts"]["0"] / d["n_clean"]))
num("NMale", d["gender_counts"]["0"], "{}")
num("NFemale", d["gender_counts"]["1"], "{}")
num("AgeMean", d["age_summary"]["mean"], "{:.1f}")
num("AgeSD", d["age_summary"]["std"], "{:.1f}")
num("AgeMin", d["age_summary"]["min"], "{:.1f}")
num("AgeMax", d["age_summary"]["max"], "{:.1f}")
prev = d["gene_prevalence"]
num("NRareGenes", sum(v < 0.05 for v in prev.values()), "{}")
num("PrevIDHOne", pct(prev["IDH1"]))
num("PrevRarestGene", pct(min(prev.values())))
num("NProjectGrade", d["n_raw"] if d["project_identical_to_grade"] else -1, "{}")

# ---------------------------------------------------------------------------
# 02 EDA
# ---------------------------------------------------------------------------
e = load("02_eda")
q = e["quality"]
num("NProfiles", q["distinct_profiles_without_age"], "{}")
num("NNoMutation", q["patients_without_mutations"], "{}")
num("NNoMutationGBM", q["patients_without_mutations_by_class"].get("GBM", 0), "{}")
num("NSixMutations", q["patients_with_6_or_more_mutations"], "{}")
num("MajorityAcc", e["majority_baseline_accuracy"])
num("AgeMedianLGG", e["age_by_class"]["LGG"]["50%"], "{:.1f}")
num("AgeMedianGBM", e["age_by_class"]["GBM"]["50%"], "{:.1f}")
num("AgeKS", e["age_ks"]["statistic"], "{:.2f}")
tests = {t["Feature"]: t for t in e["classification_tests"]}
num("AgeCohenD", tests["Age_at_diagnosis"]["Cohen d"], "{:.2f}")
num("NTests", len(tests), "{}")
num("NSigBonf", e["n_significant_bonferroni"], "{}")
num("NSigBH", e["n_significant_bh"], "{}")
num("PrevIDHOneLGG", pct(tests["IDH1"]["Prevalence LGG"]))
num("PrevIDHOneGBM", pct(tests["IDH1"]["Prevalence GBM"]))
num("PrevPTENLGG", pct(tests["PTEN"]["Prevalence LGG"]))
num("PrevPTENGBM", pct(tests["PTEN"]["Prevalence GBM"]))
num("PrevEGFRLGG", pct(tests["EGFR"]["Prevalence LGG"]))
num("PrevEGFRGBM", pct(tests["EGFR"]["Prevalence GBM"]))
num("PrevCICLGG", pct(tests["CIC"]["Prevalence LGG"]))
num("PrevCICGBM", pct(tests["CIC"]["Prevalence GBM"]))
num("PrevNOTCHLGG", pct(tests["NOTCH1"]["Prevalence LGG"]))
num("PrevNOTCHGBM", pct(tests["NOTCH1"]["Prevalence GBM"]))
rel = {r["Feature"]: r for r in e["relevance"]}
num("AUCIDHOne", rel["IDH1"]["AUC"])
num("AUCIDHOneCI", f"[{rel['IDH1']['AUC CI low']:.3f}, {rel['IDH1']['AUC CI high']:.3f}]")
num("AUCAge", rel["Age_at_diagnosis"]["AUC"])
num("AUCAgeCI", f"[{rel['Age_at_diagnosis']['AUC CI low']:.3f}, {rel['Age_at_diagnosis']['AUC CI high']:.3f}]")
num("AUCATRX", rel["ATRX"]["AUC"])
num("AUCPTEN", rel["PTEN"]["AUC"])
tc = e["top_correlations"]
pair = lambda a, b: next(r["r"] for r in tc if {r["Feature 1"], r["Feature 2"]} == {a, b})
num("CorrAgeIDH", pair("Age_at_diagnosis", "IDH1"), "{:.2f}")
num("CorrTPATRX", pair("TP53", "ATRX"), "{:.2f}")
num("CorrCICFUBP", pair("CIC", "FUBP1"), "{:.2f}")
num("MaxVIFEDA", max(v["VIF"] for v in e["vif"]), "{:.2f}")
num("YoudenAge", e["age_youden"]["threshold"], "{:.1f}")
num("YoudenSens", e["age_youden"]["sensitivity"])
num("YoudenSpec", e["age_youden"]["specificity"])
cpa = e["conditional_probability_age"]
num("PGBMYoungest", pct(cpa[0]["P(GBM)"], 0))
num("PGBMOldest", pct(cpa[-1]["P(GBM)"], 0))
bl = {b["Feature set"]: b for b in e["baseline"]}
num("BaseAUCAll", bl["All 23 predictors"]["ROC-AUC mean"])
num("BaseAUCAgeIDH", bl["Age + IDH1"]["ROC-AUC mean"])
num("BaseAUCGenes", bl["20 genes only"]["ROC-AUC mean"])
num("BaseAUCLeak", bl["All 23 + diagnosis (LEAKAGE)"]["ROC-AUC mean"])
num("NAgeSigBH", e["n_age_significant_bh"], "{}")
num("NAgeTests", len(e["age_tests"]), "{}")
at = {t["Feature"]: t for t in e["age_tests"]}
num("AgeDiffIDH", at["IDH1"]["Median difference"], "{:+.1f}")
num("AgeRaceKWp", e["age_by_race_kruskal"]["p_value"], "{:.2f}")

# ---------------------------------------------------------------------------
# 03 Unsupervised
# ---------------------------------------------------------------------------
u = load("03_unsupervised")
num("PCOneCentred", pct(u["pca"]["explained_centred"][0]))
num("PCOneScaled", pct(u["pca"]["explained_scaled"][0]))
num("NCompCentred", u["pca"]["n_components_80pct_centred"], "{}")
num("NCompScaled", u["pca"]["n_components_80pct_scaled"], "{}")
num("PctJaccardOne", pct(u["jaccard"]["share_pairs_distance_1"]))
num("MDSStress", u["jaccard"]["mds_stress1"], "{:.2f}")
num("KMain", u["k_chosen"], "{}")
num("KSecond", u["k_secondary"], "{}")
num("ARIKMWardMain", u[f"k{u['k_chosen']}"]["kmeans_vs_ward_ari"])
num("ARIKMWardSecond", u[f"k{u['k_secondary']}"]["kmeans_vs_ward_ari"])
num("NStableMain", sum(a > 0.999 for a in u[f"k{u['k_chosen']}"]["kmeans_stability_ari"]), "{}")
num("NStableSecond", sum(a > 0.999 for a in u[f"k{u['k_secondary']}"]["kmeans_stability_ari"]), "{}")
agr = {a["Partition"]: a for a in u["agreement_with_grade"]}
num("ARIGradeMain", agr[f"k-means (k = {u['k_chosen']})"]["ARI"])
num("NMIGradeMain", agr[f"k-means (k = {u['k_chosen']})"]["NMI"])
num("ARIGradeSecond", agr[f"k-means (k = {u['k_secondary']})"]["ARI"])
lk = {(r["Distance"], r["Linkage"]): r for r in u["linkage_comparison"]}
num("JaccardAverageLargest", lk[("Jaccard", "average")]["Largest cluster (%)"], "{:.0f}")
num("JaccardCompleteLargest", lk[("Jaccard", "complete")]["Largest cluster (%)"], "{:.0f}")
for k, tag in ((u["k_chosen"], "Main"), (u["k_secondary"], "Second")):
    prof = u[f"k{k}"]["profiles"]
    for c, letter in zip(sorted(prof), "ABCD"):
        num(f"Clu{tag}{letter}N", prof[c]["n"], "{}")
        num(f"Clu{tag}{letter}GBM", prof[c]["GBM (%)"], "{:.0f}")
        num(f"Clu{tag}{letter}Age", prof[c]["Median age"], "{:.0f}")

# ---------------------------------------------------------------------------
# 04 Regression
# ---------------------------------------------------------------------------
r = load("04_regression")
rs = {m["Model"]: m for m in r["summary"]}
num("RegBest", r["best_mae"])
num("RegBestMAE", rs[r["best_mae"]]["MAE"], "{:.2f}")
num("RegBestMAECI", f"[{rs[r['best_mae']]['MAE CI low']:.2f}, {rs[r['best_mae']]['MAE CI high']:.2f}]")
num("RegBestRTwo", rs[r["best_mae"]]["R2"])
num("RegBestRTwoCI", f"[{rs[r['best_mae']]['R2 CI low']:.3f}, {rs[r['best_mae']]['R2 CI high']:.3f}]")
num("RegBaseMAE", rs["Mean baseline"]["MAE"], "{:.2f}")
num("RegBaseRMSE", rs["Mean baseline"]["RMSE"], "{:.2f}")
models_ = [m for m in rs if m != "Mean baseline"]
num("RegMinMAE", min(rs[m]["MAE"] for m in models_), "{:.2f}")
num("RegMaxMAE", max(rs[m]["MAE"] for m in models_), "{:.2f}")
num("RegMinRTwo", min(rs[m]["R2"] for m in models_))
num("RegMaxRTwo", max(rs[m]["R2"] for m in models_))
num("RegOLSMAE", rs["Linear regression"]["MAE"], "{:.2f}")
num("RegOLSRTwo", rs["Linear regression"]["R2"])
num("RegNTies", len(r["ties_mae"]), "{}")
pd_ = {p["Model"]: p for p in r["paired_mae_differences"]}
num("RegDiffOLSBest", pd_["Linear regression"][f"MAE - {r['best_mae']}"], "{:+.2f}")
num("RegDiffOLSBestCI", f"[{pd_['Linear regression']['CI low (best)']:+.2f}, {pd_['Linear regression']['CI high (best)']:+.2f}]")
num("RegCorrGrade", r["corr_prediction_grade"], "{:.2f}")
ob = r["oof_by_grade_best_model"]
num("RegPredLGG", ob["LGG"]["Predicted"], "{:.1f}")
num("RegPredGBM", ob["GBM"]["Predicted"], "{:.1f}")
ols = r["ols"]
num("OLSRTwoIn", ols["r2_in_sample"])
num("OLSIntercept", ols["intercept"], "{:.1f}")
co = ols["coefficients"]
num("OLSIDHOne", co["IDH1"]["Coefficient (years)"], "{:+.1f}")
num("OLSIDHOneCI", f"[{co['IDH1']['CI low']:+.1f}, {co['IDH1']['CI high']:+.1f}]")
num("OLSIDHTwo", co["IDH2"]["Coefficient (years)"], "{:+.1f}")
num("OLSIDHTwoCI", f"[{co['IDH2']['CI low']:+.1f}, {co['IDH2']['CI high']:+.1f}]")
num("OLSATRX", co["ATRX"]["Coefficient (years)"], "{:+.1f}")
num("OLSNative", co["Race_native"]["Coefficient (years)"], "{:+.1f}")
num("OLSNativeCI", f"[{co['Race_native']['CI low']:+.1f}, {co['Race_native']['CI high']:+.1f}]")
num("OLSNSigBH", sum(v["p BH"] < 0.05 for v in co.values()), "{}")
num("OLSMaxVIF", max(v["VIF"] for v in ols["vif"]), "{:.2f}")
num("OLSMaxCook", ols["max_cooks_distance"], "{:.3f}")
num("OLSBPp", ols["breusch_pagan_p"], "{:.3f}")
num("OLSShapirop", ols["shapiro_residuals_p"], "{:.3f}")
num("LassoAlpha", r["final_hyperparameters"]["Lasso"]["alpha"], "{:.3g}")
num("RidgeAlpha", r["final_hyperparameters"]["Ridge"]["alpha"], "{:.3g}")
num("LassoNZero", r["lasso_n_zero"], "{}")
num("LassoNCoef", len(r["penalised_coefficients"]), "{}")
num("RegTreeDepth", r["tree"]["depth"], "{}")
num("RegTreeLeaves", r["tree"]["n_leaves"], "{}")
pim = r["permutation_importance_mean"]["IDH1"]
num("RegPermIDH", sum(pim.values()) / len(pim), "{:.2f}")

# ---------------------------------------------------------------------------
# 05 Classification
# ---------------------------------------------------------------------------
c = load("05_classification")
cs = {m["Model"]: m for m in c["summary"]}
best = c["best_auc"]
num("ClfBest", best)
num("ClfBestAUC", cs[best]["AUC"])
num("ClfBestAUCCI", f"[{cs[best]['AUC CI low']:.3f}, {cs[best]['AUC CI high']:.3f}]")
num("ClfBestSens", cs[best]["Sensitivity"])
num("ClfBestSpec", cs[best]["Specificity"])
num("ClfBestBAcc", cs[best]["Balanced accuracy"])
for model, tag in (("Logistic regression", "LR"), ("Naive Bayes", "NB"), ("Decision tree (pruned)", "Tree"),
                   ("Random forest", "RF"), ("XGBoost", "XGB"), ("SVM", "SVM"), ("Majority baseline", "Base")):
    num(f"ClfAUC{tag}", cs[model]["AUC"])
num("ClfBaseAcc", cs["Majority baseline"]["Accuracy"])
clf_models = [m for m in cs if m != "Majority baseline"]
num("ClfMinAUC", min(cs[m]["AUC"] for m in clf_models))
num("ClfMaxAUC", max(cs[m]["AUC"] for m in clf_models))
num("ClfMinAcc", min(cs[m]["Accuracy"] for m in clf_models))
num("ClfMaxAcc", max(cs[m]["Accuracy"] for m in clf_models))
num("ClfNTies", len(c["ties_auc"]), "{}")
pdc = {p["Model"]: p for p in c["paired_auc_differences"]}
for model, tag in (("Logistic regression", "LR"), ("Naive Bayes", "NB"), ("Decision tree (pruned)", "Tree"),
                   ("XGBoost", "XGB"), ("SVM", "SVM")):
    p = pdc[model]
    num(f"ClfDiff{tag}", f"{p[f'AUC - {best}']:+.3f} [{p['CI low']:+.3f}, {p['CI high']:+.3f}]")
num("ClfNWorse", sum(p["CI high"] < 0 for p in c["paired_auc_differences"]), "{}")
fl = {f["Model"]: f for f in c["final_search_flatness"]}
num("LRFlatWithin", fl["Logistic regression"]["Configurations within 0.005 of the best"], "{}")
num("LRFlatTotal", fl["Logistic regression"]["Configurations"], "{}")
lr = c["logistic"]
num("LRFinalC", lr["params"]["C"], "{:.3g}")
num("LRFinalPenalty", lr["params"]["penalty"].upper())
num("LRCoefIDH", lr["coefficients"]["IDH1"]["Coefficient (log-odds per SD)"], "{:+.2f}")
num("LRORIDH", lr["coefficients"]["IDH1"]["Odds ratio per SD"], "{:.2f}")
num("LRCoefNOTCH", lr["coefficients"]["NOTCH1"]["Coefficient (log-odds per SD)"], "{:+.2f}")
num("LRCoefAge", lr["coefficients"]["Age_at_diagnosis"]["Coefficient (log-odds per SD)"], "{:+.2f}")
num("LRORAge", lr["coefficients"]["Age_at_diagnosis"]["Odds ratio per SD"], "{:.2f}")
num("NBExtreme", pct(c["naive_bayes"]["share_extreme_probabilities"], 0))
num("LRExtreme", pct(lr["share_extreme_probabilities"], 0))
num("ClfTreeDepth", c["tree"]["depth"], "{}")
num("ClfTreeLeaves", c["tree"]["n_leaves"], "{}")
num("ClfTreeAlpha", c["tree"]["params"]["ccp_alpha"], "{:.3g}")
num("ClfTreeCriterion", c["tree"]["params"]["criterion"])
cm = c["confusion_matrices"][best]   # [[TN, FP], [FN, TP]]
num("ClfFN", cm[1][0], "{}")
num("ClfFP", cm[0][1], "{}")
pi = c["permutation_importance_mean"]["IDH1"]
num("ClfPermIDH", sum(pi.values()) / len(pi))

lines = ["% Generated by report/make_numbers.py from results/logs/*.json -- do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(macros)} macros written to {OUT.relative_to(ROOT)}")
