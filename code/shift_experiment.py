"""
ANLY 735 | Replication Laboratory #3 | When the Test Environment Changes

Proxy replication of the WILDS evaluation logic (Koh et al., 2021):
same task + same pipeline + different evaluation environment.

Task: ACSIncome (Ding et al., 2021). Predict whether a working adult earns
more than $50,000 a year from census person records.
Development environment: California, ACS 2018 1-year PUMS.
Shifted environment: Texas, ACS 2018 1-year PUMS.

The script writes every number used in the report to analysis/ and the
figure to figures/. Run from the repository root or from code/.
"""

from pathlib import Path
import json
import platform
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Reproducibility settings
# ------------------------------------------------------------------
SEED = 735
TEST_SIZE = 0.20
N_BOOT = 200
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis"
FIG = ROOT / "figures"
for d in (DATA, OUT, FIG):
    d.mkdir(exist_ok=True)

STATES = {"CA": ("ca", "psam_p06.csv"), "TX": ("tx", "psam_p48.csv")}
URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_p{}.zip"

# ------------------------------------------------------------------
# 2. Acquire data (public Census files, downloaded once)
# ------------------------------------------------------------------
for code, (abbr, csv) in STATES.items():
    if not (DATA / csv).exists():
        zpath = DATA / f"csv_p{abbr}.zip"
        print(f"Downloading {code} 2018 PUMS ...")
        urllib.request.urlretrieve(URL.format(abbr), zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract(csv, DATA)

# ------------------------------------------------------------------
# 3. ACSIncome task definition (Ding et al., 2021)
# ------------------------------------------------------------------
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
CATEGORICAL = ["COW", "MAR", "RELP", "SEX", "RAC1P"]  # small nominal codes
# OCCP and POBP have hundreds of codes; they are kept as numeric codes.
# Census numbers occupations in family blocks, so tree splits on code
# ranges group related occupations.


def load_state(code):
    _, csv = STATES[code]
    cols = FEATURES + ["PINCP", "PWGTP"]
    df = pd.read_csv(DATA / csv, usecols=cols)
    # ACSIncome filter: age > 16, income > 100, works > 0 hours, weight >= 1
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)].copy()
    df = df.dropna(subset=["OCCP", "COW", "SCHL"])
    df["y"] = (df.PINCP > 50000).astype(int)
    df["state"] = code
    return df.reset_index(drop=True)


ca = load_state("CA")
tx = load_state("TX")

# Race groups for subgroup analysis (RAC1P: 1 White, 2 Black, 6 Asian)
RACE = {1: "White", 2: "Black", 6: "Asian"}
for df in (ca, tx):
    df["race_group"] = df.RAC1P.map(RACE).fillna("Other")
    df["sex_group"] = df.SEX.map({1: "Male", 2: "Female"})

# ------------------------------------------------------------------
# 4. Environment split
#    CA -> train / ID test. TX -> shifted test (and a TX train split used
#    ONLY for the target-trained reference model, as WILDS does).
# ------------------------------------------------------------------
ca_tr, ca_te = train_test_split(ca, test_size=TEST_SIZE, random_state=SEED, stratify=ca.y)
tx_tr, tx_te = train_test_split(tx, test_size=TEST_SIZE, random_state=SEED, stratify=tx.y)


def make_model():
    """The one fixed model. Defaults only; no tuning on any environment."""
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    return HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=SEED)


model = make_model().fit(ca_tr[FEATURES], ca_tr.y)

# Reference: identical pipeline trained on Texas (WILDS "in-distribution
# oracle"). It separates "Texas is harder" from "the model did not travel".
ref = make_model().fit(tx_tr[FEATURES], tx_tr.y)


# ------------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, p >= 0.5),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
        "observed_rate": y.mean(),
        "mean_pred": p.mean(),
        "fnr": ((p < 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1),
        "fpr": ((p >= 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1),
    }


def boot_ci(y, p, key, n=N_BOOT):
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(metrics(y[i], p[i])[key])
    return np.percentile(vals, [2.5, 97.5]).tolist()


p_id = model.predict_proba(ca_te[FEATURES])[:, 1]
p_ood = model.predict_proba(tx_te[FEATURES])[:, 1]
p_ref = ref.predict_proba(tx_te[FEATURES])[:, 1]

res = {
    "id_ca_model_on_ca": metrics(ca_te.y, p_id),
    "ood_ca_model_on_tx": metrics(tx_te.y, p_ood),
    "ref_tx_model_on_tx": metrics(tx_te.y, p_ref),
}
for k, (y, p) in {"id_ca_model_on_ca": (ca_te.y, p_id), "ood_ca_model_on_tx": (tx_te.y, p_ood),
                  "ref_tx_model_on_tx": (tx_te.y, p_ref)}.items():
    for m in ("auc", "accuracy", "ece"):
        res[k][m + "_ci"] = boot_ci(y, p, m)

# Bootstrap CI for the ID minus shifted gap (paired over independent sets)
gaps = {m: [] for m in ("auc", "accuracy", "ece", "brier")}
yid, yod = ca_te.y.to_numpy(), tx_te.y.to_numpy()
for _ in range(N_BOOT):
    i = rng.integers(0, len(yid), len(yid))
    j = rng.integers(0, len(yod), len(yod))
    a, b = metrics(yid[i], p_id[i]), metrics(yod[j], p_ood[j])
    for m in gaps:
        gaps[m].append(b[m] - a[m])
res["shift_minus_id_ci"] = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in gaps.items()}

# ------------------------------------------------------------------
# 6. Diagnose before fixing
# ------------------------------------------------------------------
# 6a. Label-shift-only correction: rescale the CA model's odds by the
#     change in base rate. Uses the TX base rate only, no TX features or
#     retraining. If calibration recovers, prevalence change explains it.
pi_s, pi_t = ca_tr.y.mean(), tx_tr.y.mean()
odds = p_ood / (1 - p_ood) * (pi_t / (1 - pi_t)) / (pi_s / (1 - pi_s))
p_ls = odds / (1 + odds)
res["ood_label_shift_adjusted"] = metrics(tx_te.y, p_ls)

# 6b. Covariate shift: standardized mean differences and distribution distances
diag = []
for f in ["AGEP", "SCHL", "WKHP"]:
    a, b = ca[f], tx[f]
    smd = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
    diag.append({"feature": f, "type": "numeric", "CA": a.mean(), "TX": b.mean(), "shift": smd,
                 "measure": "standardized mean difference"})


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


for f in ["COW", "MAR", "RELP", "SEX", "RAC1P", "OCCP", "POBP"]:
    diag.append({"feature": f, "type": "categorical", "CA": np.nan, "TX": np.nan,
                 "shift": tvd(ca[f], tx[f]), "measure": "total variation distance"})
pd.DataFrame(diag).to_csv(OUT / "feature_shift.csv", index=False)

# Notable place-of-birth shares (POBP: 6 California, 48 Texas, 303 Mexico,
# 207 India, 233 Philippines, 210 China)
pob = {}
for code, name in {6: "California", 48: "Texas", 303: "Mexico", 233: "Philippines"}.items():
    pob[name] = {"CA": float((ca.POBP == code).mean()), "TX": float((tx.POBP == code).mean())}
res["place_of_birth_share"] = pob

# 6c. Representation: domain classifier (CA vs TX) on the model's features.
#     High AUC = environments are separable; share of TX with p(TX) > 0.9
#     = TX rows that look unlike anything common in CA.
dom = pd.concat([ca_tr.sample(40000, random_state=SEED), tx_tr.sample(40000, random_state=SEED)])
dom_te = pd.concat([ca_te.sample(10000, random_state=SEED), tx_te.sample(10000, random_state=SEED)])
dclf = make_model().fit(dom[FEATURES], (dom.state == "TX").astype(int))
pd_te = dclf.predict_proba(dom_te[FEATURES])[:, 1]
res["domain_classifier_auc"] = roc_auc_score((dom_te.state == "TX").astype(int), pd_te)


# Place of birth encodes the state itself (born in CA vs born in TX), so it
# makes the environments trivially separable. Repeat without POBP to see how
# different the two workforces look on everything else.
F_NOPOB = [f for f in FEATURES if f != "POBP"]
dclf2 = HistGradientBoostingClassifier(
    categorical_features=[f in CATEGORICAL for f in F_NOPOB], random_state=SEED
).fit(dom[F_NOPOB], (dom.state == "TX").astype(int))
res["domain_classifier_auc_without_pobp"] = roc_auc_score(
    (dom_te.state == "TX").astype(int), dclf2.predict_proba(dom_te[F_NOPOB])[:, 1])
# TX test rows the no-POBP domain classifier is confident are "not CA-like"
p_tx_rows = dclf2.predict_proba(tx_te[F_NOPOB])[:, 1]
poor = p_tx_rows > 0.8
res["tx_share_poorly_represented"] = float(poor.mean())
res["ood_on_poorly_represented"] = metrics(tx_te.y[poor], p_ood[poor])
res["ood_on_well_represented"] = metrics(tx_te.y[~poor], p_ood[~poor])

# 6d. Relationship change: income-by-education curve in each state, and
#     logistic slopes on schooling and hours fitted within each state.
edu = []
for code, df in (("CA", ca), ("TX", tx)):
    g = df.assign(edu=pd.cut(df.SCHL, [0, 15, 19, 20, 21, 24],
                             labels=["No diploma", "HS / some college", "Associate",
                                     "Bachelor", "Graduate"]))
    for lvl, sub in g.groupby("edu", observed=True):
        edu.append({"state": code, "education": str(lvl), "share": len(sub) / len(df),
                    "rate_over_50k": sub.y.mean()})
pd.DataFrame(edu).to_csv(OUT / "education_relationship.csv", index=False)

slopes = {}
for code, df in (("CA", ca_tr), ("TX", tx_tr)):
    Xs = df[["SCHL", "WKHP", "AGEP"]]
    Xs = (Xs - ca_tr[["SCHL", "WKHP", "AGEP"]].mean()) / ca_tr[["SCHL", "WKHP", "AGEP"]].std()
    lr = LogisticRegression(max_iter=1000).fit(Xs, df.y)
    slopes[code] = dict(zip(["SCHL", "WKHP", "AGEP"], lr.coef_[0].round(3).tolist()))
    slopes[code]["intercept"] = round(float(lr.intercept_[0]), 3)
res["logistic_slopes_standardized"] = slopes

# ------------------------------------------------------------------
# 7. Look beneath the average: subgroups
# ------------------------------------------------------------------
rows = []
for gcol in ("sex_group", "race_group"):
    for grp in ["Male", "Female"] if gcol == "sex_group" else ["White", "Black", "Asian", "Other"]:
        mi, mo = (ca_te[gcol] == grp).to_numpy(), (tx_te[gcol] == grp).to_numpy()
        a = metrics(ca_te.y[mi], p_id[mi])
        b = metrics(tx_te.y[mo], p_ood[mo])
        r = metrics(tx_te.y[mo], p_ref[mo])
        rows.append({"variable": gcol.replace("_group", ""), "group": grp,
                     "n_id": a["n"], "n_shift": b["n"],
                     "rate_id": a["observed_rate"], "rate_shift": b["observed_rate"],
                     "acc_id": a["accuracy"], "acc_shift": b["accuracy"], "acc_ref": r["accuracy"],
                     "auc_id": a["auc"], "auc_shift": b["auc"],
                     "ece_id": a["ece"], "ece_shift": b["ece"],
                     "pred_minus_obs_shift": b["mean_pred"] - b["observed_rate"],
                     "fnr_id": a["fnr"], "fnr_shift": b["fnr"]})
sub = pd.DataFrame(rows)
sub.to_csv(OUT / "subgroups.csv", index=False)

# ------------------------------------------------------------------
# 8. Figure: reliability (calibration) + subgroup accuracy change
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
edges = np.linspace(0, 1, 11)
for y, p, lab, st in ((ca_te.y.to_numpy(), p_id, "In-distribution (CA test)", "-o"),
                      (tx_te.y.to_numpy(), p_ood, "Shifted (TX test)", "-s"),
                      (tx_te.y.to_numpy(), p_ls, "Shifted, base-rate adjusted", ":^")):
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
    ax[0].plot(xs, ys, st, label=lab)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0].set_xlabel("Predicted probability of income > $50k")
ax[0].set_ylabel("Observed share > $50k")
ax[0].set_title("A. Calibration of the California model")
ax[0].legend(fontsize=8)

lab = sub.group.tolist()
x = np.arange(len(lab))
ax[1].bar(x - 0.2, sub.acc_id, 0.4, label="In-distribution (CA)")
ax[1].bar(x + 0.2, sub.acc_shift, 0.4, label="Shifted (TX)")
ax[1].scatter(x + 0.2, sub.acc_ref, color="k", marker="_", s=300, zorder=3, label="TX-trained reference")
ax[1].set_xticks(x, lab)
ax[1].set_ylim(0.6, 0.95)
ax[1].set_ylabel("Accuracy")
ax[1].set_title("B. Accuracy by subgroup")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "shift_result.png", dpi=160)

# ------------------------------------------------------------------
# 9. Save
# ------------------------------------------------------------------
res["sizes"] = {"ca_all": len(ca), "tx_all": len(tx), "ca_train": len(ca_tr), "ca_test": len(ca_te),
                "tx_train": len(tx_tr), "tx_test": len(tx_te)}
res["base_rates"] = {"CA": float(ca.y.mean()), "TX": float(tx.y.mean())}
res["session"] = {"python": platform.python_version(), "platform": platform.platform(),
                  "numpy": np.__version__, "pandas": pd.__version__,
                  "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                  "seed": SEED, "bootstrap_reps": N_BOOT}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


(OUT / "results.json").write_text(json.dumps(clean(res), indent=2))
print(json.dumps(clean(res), indent=2))
print(sub.round(3).to_string())
print(pd.DataFrame(diag).round(3).to_string())
print(pd.DataFrame(edu).round(3).to_string())


# ------------------------------------------------------------------
# 10. Markdown tables included by the Quarto report
#     (generated here so every reported number comes from this run)
# ------------------------------------------------------------------
R = clean(res)


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


I, O, F = R["id_ca_model_on_ca"], R["ood_ca_model_on_tx"], R["ref_tx_model_on_tx"]
G = R["shift_minus_id_ci"]
main = [
    ["AUROC (primary, ranking)", f"{I['auc']:.3f}", f"{O['auc']:.3f}", f"{O['auc']-I['auc']:+.3f} {ci(G['auc'])}", f"{F['auc']:.3f}"],
    ["Accuracy at 0.5 (secondary)", f"{I['accuracy']:.3f}", f"{O['accuracy']:.3f}", f"{O['accuracy']-I['accuracy']:+.3f} {ci(G['accuracy'])}", f"{F['accuracy']:.3f}"],
    ["Expected calibration error", f"{I['ece']:.3f}", f"{O['ece']:.3f}", f"{O['ece']-I['ece']:+.3f} {ci(G['ece'])}", f"{F['ece']:.3f}"],
    ["Brier score", f"{I['brier']:.3f}", f"{O['brier']:.3f}", f"{O['brier']-I['brier']:+.3f} {ci(G['brier'])}", f"{F['brier']:.3f}"],
    ["Mean predicted share > $50k", f"{I['mean_pred']:.3f}", f"{O['mean_pred']:.3f}", f"{O['mean_pred']-I['mean_pred']:+.3f}", f"{F['mean_pred']:.3f}"],
    ["Observed share > $50k", f"{I['observed_rate']:.3f}", f"{O['observed_rate']:.3f}", f"{O['observed_rate']-I['observed_rate']:+.3f}", f"{F['observed_rate']:.3f}"],
    ["False positive rate", f"{I['fpr']:.3f}", f"{O['fpr']:.3f}", f"{O['fpr']-I['fpr']:+.3f}", f"{F['fpr']:.3f}"],
    ["False negative rate", f"{I['fnr']:.3f}", f"{O['fnr']:.3f}", f"{O['fnr']-I['fnr']:+.3f}", f"{F['fnr']:.3f}"],
]
(OUT / "table_main.md").write_text(md(
    ["Metric", "In-distribution (CA test)", "Shifted (TX test)", "Change [95% CI]", "TX-trained reference"], main))

srows = [[r.variable, r.group, f"{r.n_id:,} / {r.n_shift:,}", f"{r.rate_id:.2f} / {r.rate_shift:.2f}",
          f"{r.acc_id:.3f} / {r.acc_shift:.3f}", f"{r.acc_ref:.3f}", f"{r.auc_id:.3f} / {r.auc_shift:.3f}",
          f"{r.ece_id:.3f} / {r.ece_shift:.3f}", f"{r.fnr_id:.2f} / {r.fnr_shift:.2f}"] for r in sub.itertuples()]
(OUT / "table_subgroups.md").write_text(md(
    ["Variable", "Group", "n (CA / TX)", "Share > $50k", "Accuracy", "TX reference acc.", "AUROC", "ECE", "False negative rate"], srows))

names = {"AGEP": "Age", "SCHL": "Education level", "WKHP": "Weekly hours worked", "COW": "Class of worker",
         "MAR": "Marital status", "RELP": "Household relationship", "SEX": "Sex", "RAC1P": "Race",
         "OCCP": "Occupation", "POBP": "Place of birth"}
drows = [[names[d["feature"]], d["measure"], f"{d['shift']:.3f}"] for d in diag]
(OUT / "table_feature_shift.md").write_text(md(["Predictor", "Shift measure (CA vs TX)", "Value"], drows))

e = pd.DataFrame(edu)
erows = []
for lvl in e.education.unique():
    a = e[(e.state == "CA") & (e.education == lvl)].iloc[0]
    b = e[(e.state == "TX") & (e.education == lvl)].iloc[0]
    erows.append([lvl, f"{a.share:.3f} / {b.share:.3f}", f"{a.rate_over_50k:.3f} / {b.rate_over_50k:.3f}"])
(OUT / "table_education.md").write_text(md(["Education", "Share of workers (CA / TX)", "Share > $50k (CA / TX)"], erows))

S = R["session"]
(OUT / "session_info.md").write_text(md(["Item", "Value"], [[k, v] for k, v in S.items()]))
