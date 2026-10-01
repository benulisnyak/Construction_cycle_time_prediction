# Methodology

## Prediction objective

Estimate total construction duration on the day construction starts. The target is
the difference between construction end and construction start dates, measured in
calendar days. A home-specific estimate can capture differences that a single
floor-plan average misses.

## Data preparation

- Remove 25 exact duplicate historical rows.
- Standardize categorical whitespace and capitalization in both datasets.
- Read the literal `None` basement label as a category; distinguish it from truly
  empty values. Retain missing numerical features for CatBoost's native handling.
- Divide square footage above 10,000 by ten. Initial investigation found these
  values were approximately ten times the typical area for their floor plans.
  This affects 13 historical records and 3 prediction records. This rule is a
  source-specific correction, not a universal rule about large homes.
- Compute targets for completed builds only. Exclude missing outcomes and durations
  of one day or less, leaving 4,077 eligible homes. Leave the public input CSVs unchanged during modeling.

Preprocessing is shared between training and prediction. Basement labels receive
identical treatment in both paths; a genuinely missing category has an explicit
sentinel. All categorical model fields, including product line, are normalized
consistently.

## Features and leakage controls

The model uses 22 features across property configuration, geography, permitting,
and operational context. Five features are derived from dates:

| Feature | Definition |
| --- | --- |
| `days_sale_to_start` | Start date minus sale date; missing if the sale occurs later |
| `permit_days` | Permit issue date minus application date |
| `permit_to_start_days` | Construction start date minus permit issue date |
| `start_month` | Calendar month when construction starts |
| `start_season` | Northern Hemisphere season derived from the start month |

Construction end date, final inspection date, construction status, trade invoice
total, and change-order count are excluded from the input features. End date and
status are used only to construct eligible historical targets. Home ID and data
source are excluded from modeling. Lot numbers have been removed from the public
inputs. Plan name is redundant with plan code.

The workflow assumes selected options value, assigned manager, and permit dates
are available as of construction start. Both included datasets have permit
issuance on or before construction start. Historical exports alone do not prove
that every other field reflects its value at that time; point-in-time availability
should be verified before applying this model operationally.

## Model and validation

CatBoost handles mixed numerical and categorical inputs without a separate
one-hot encoding stage. The configuration is 600 boosting iterations, tree depth
5, learning rate 0.03, and random seed 42. No tuning or early stopping occurs in
this pipeline.

Five shuffled folds are stratified by `plan_code` so each fold has a similar
floor-plan mix. Each home receives one out-of-fold estimate from a model that did
not train on it. MAE, RMSE, and R² are computed per fold and averaged without
weighting. Training R² is also reported to show the train-validation gap.

A global median baseline is fitted separately on each training fold. Its MAE and
RMSE use the same held-out homes as CatBoost. Baseline metrics are calculated by
the current pipeline; they are not inferred from previously reported results.

This split is appropriate for comparing homes drawn from a similar observed
population. It does not establish future forecasting performance or performance
for an entirely new community. A chronological holdout and grouped evaluation are
appropriate extensions.

## Interpreting results

See the repository README and `reports/` for the full-data run. The prediction
scatterplot compares actual durations with out-of-fold estimates. Feature
importance comes from the final model fitted on all eligible completed homes;
it measures predictive contribution rather than a causal effect or the direction
of a feature's relationship with duration.

Exported estimates are rounded to whole days and clipped to a minimum of one day.
No upper cap is applied. Validation uses raw estimates before this export formatting.

## Site-manager analysis

With `--manager-analysis`, a second set of models uses the same folds and model
configuration but excludes `site_manager`. For each home, define:

```text
residual = actual cycle days - out-of-fold expected cycle days
```

A negative average means a manager's observed builds completed faster than the
model expected given the other included features. The output reports count, mean,
median, standard deviation, standard error, and descriptive 95% normal intervals
(`mean ± 1.96 × standard error`). A single-home group has no estimable standard
error and is plotted without an interval.

These are observational comparisons. Community can act as a proxy for manager,
assignment practices may differ, and unmeasured project complexity can affect
both assignments and duration. The intervals do not account for model uncertainty,
correlated projects, or multiple comparisons. They should not be read as causal
manager rankings or as future-home prediction intervals.

## Limitations and extensions

Completed-only training is subject to right-censoring: slow recent projects may
still be in progress and absent from the targets. Survival models could incorporate
these partial observations. A temporal holdout would better simulate future use,
while group-based evaluation could test generalization to unseen communities.

Other useful extensions include per-floor-plan baselines, prediction intervals,
and point-in-time weather, labor, supply, and inspection information. Missing beds
and lot area may have systematic causes, so their provenance also merits further
investigation. Cleaning assumptions should be reassessed when the data source changes.
