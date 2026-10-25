# ANLY 735 Replication Laboratory #3

When the Test Environment Changes: a WILDS-style proxy replication on ACS Income

Author: Shail Sanjaykumar Raval. Seed: 735.

## Anchor study

Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts.
PMLR 139, 5637 to 5664. https://proceedings.mlr.press/v139/koh21a.html

## Design

One model (scikit-learn HistGradientBoostingClassifier, defaults), one task
(ACSIncome, Ding et al. 2021: income above $50,000), two environments:

- Development / in-distribution: California, ACS 2018 1-year PUMS (80/20 split)
- Shifted: Texas, ACS 2018 1-year PUMS (test split)
- Reference: same pipeline trained on Texas, scored on the same Texas test set

## Layout

```text
replication-lab.qmd        report source (renders to HTML and DOCX)
references.bib             citations
code/shift_experiment.py   full analysis; writes analysis/ and figures/
analysis/results.json      every headline number
analysis/*.md              tables included by the report
analysis/*.csv             subgroup, feature-shift, education tables
figures/shift_result.png   calibration + subgroup accuracy figure
data/                      Census files (downloaded by the script, not committed)
```

## Reproduce

Python 3.12 with numpy, pandas, scikit-learn, matplotlib; Quarto 1.10.

```bash
python3 code/shift_experiment.py
quarto render replication-lab.qmd
```

Deterministic under seed 735; about 35 seconds on CPU after the one-time
download (about 115 MB).
