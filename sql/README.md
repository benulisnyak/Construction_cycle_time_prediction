# SQL case study: data quality and construction prediction errors

**Question:** Are the historical records suitable for model evaluation, and where
does the construction-duration model make its largest errors?

This standalone DuckDB analysis reads the existing historical CSV and out-of-fold
predictions. It adds a SQL perspective to the Python project without retraining
models, changing dependencies in the project requirements, or modifying any data
or reports. All SQL objects are temporary and the default connection is in memory.

## Run the case study

Tested on October 5, 2026 with **DuckDB 1.5.6** and **Python 3.13.7**. Start in the
repository root, not in the `sql/` directory. DuckDB is an optional dependency for
this case study only; the Python modeling pipeline does not require it.

Install it in your preferred Python environment:

```bash
python -m pip install duckdb==1.5.6
```

Then execute this short launcher from the repository root. It runs the SQL
statements in order and prints every result table; all analysis logic is in SQL.
It stops on any failed validation rather than continuing to produce metrics.

```bash
python - <<'PY'
from pathlib import Path
import duckdb

with duckdb.connect(':memory:') as connection:
    for statement in connection.extract_statements(Path('sql/analysis.sql').read_text()):
        result = connection.execute(statement.query)
        if statement.type.name == 'SELECT':
            print('\t'.join(column[0] for column in result.description))
            for row in result.fetchall():
                print('\t'.join(str(value) for value in row))
            print()
PY
```

Alternatively, with the separately installed DuckDB CLI:

```bash
duckdb -bail < sql/analysis.sql
```

The `-bail` option stops the CLI if a join validation fails. The Python package
installation alone does not install the CLI.

## Inputs and analysis grain

| Input | Role |
| --- | --- |
| `data/build_history.csv` | Source dates, eligibility, community, metro, and floor plan |
| `reports/out_of_fold_predictions.csv` | Historical predictions made without training on the corresponding home |

The analysis does not evaluate `homes_to_predict.csv`: those homes do not have
known outcomes. It also does not read archived originals or reverse the project's
pseudonymous identifiers.

The audit starts with raw rows, removes **exact duplicates across all columns**,
then checks whether any conflicting records still share an ID. Conflicting IDs
cause evaluation to fail; the analysis never selects an arbitrary record to make
a join succeed. The final evaluation grain is one eligible completed home per row.

Strings are initially read as text. `TRY_CAST` allows invalid date values to be
counted explicitly, instead of aborting the audit or confusing invalid dates with
valid ones. The literal basement label `None` remains a valid category.

## Queries and SQL techniques

| Section in `analysis.sql` | Purpose | Techniques |
| --- | --- | --- |
| 1 | Count raw rows and exact duplicates | `DISTINCT`, scalar subqueries, aggregates |
| 2 | Audit missing values, malformed dates, and cleaning candidates | CTE, `UNION ALL`, `FILTER`, `TRY_CAST`, `NULLIF` |
| 3 | Reconcile eligibility with the Python pipeline | `CASE`, date arithmetic, `GROUP BY` |
| 4 | Enforce unique IDs, complete coverage, and correct actual durations | `HAVING`, `NOT EXISTS`, joins, failing assertions |
| 5 | Calculate overall prediction errors | MAE, RMSE, median, conditional aggregation |
| 6 | Rank community-level errors | CTE, `DENSE_RANK`, minimum sample threshold |
| 7 | Compare floor plans and their share of total absolute error | CTEs, `SUM(...) OVER ()` |
| 8 | Diagnose error direction across actual-duration bands | `CASE`, grouped and conditional aggregation |

Community groups use **metro plus community**, so the same community name in two
cities is not silently combined. Community and floor-plan tables display groups
with at least 30 homes. The floor-plan error-share denominator includes all homes,
including any groups excluded by that display threshold. Ranks use unrounded MAE,
so displayed values that round to the same number can have different ranks.

## Verified findings

These results were computed from the included public inputs on October 5, 2026.
They are descriptive diagnostics, not claims about causal drivers or performance
on new data.

### Data quality

| Check | Result |
| --- | ---: |
| Raw historical rows | 4,684 |
| Exact duplicate rows removed | 25 |
| Distinct historical homes | 4,659 |
| Homes not marked complete | 575 |
| Completed homes with duration of one day or less | 7 |
| Eligible completed homes | 4,077 |
| Missing sale dates, after deduplication | 1,577 |
| Missing beds, after deduplication | 139 |
| Missing lot areas, after deduplication | 97 |
| Valid literal `None` basement labels, after deduplication | 1,644 |
| Square footage above 10,000, after deduplication | 13 |
| Community labels with extra leading/trailing whitespace | 197 |

No malformed non-empty start/end dates, genuinely missing basement categories, or
completed homes with missing start/end dates were found. All nine join checks
passed, giving **4,077 matched homes without row multiplication**. Audit categories
can overlap; only the eligibility buckets are mutually exclusive. The SQL flags
square-footage candidates but does not apply the Python model's correction.

### Overall model error

| Pooled metric | Value |
| --- | ---: |
| MAE | 30.91 days |
| RMSE | 40.44 days |
| Mean signed error | +0.30 days |
| Median absolute error | 24.71 days |
| Predictions within 30 days of the outcome | 59.11% |

Here, **signed error = predicted minus actual duration**. Negative values indicate
underestimation. This is the opposite sign convention from the project's manager
residual report, which uses actual minus expected duration.

These are metrics pooled over individual homes. The main project reports the
unweighted average of per-fold metrics. In particular, taking the square root
across all squared errors is not the same as averaging five fold RMSEs; the pooled
40.44-day RMSE therefore differs slightly from the reported mean fold RMSE.

### Where the model needs attention

- **Community:** Calgary's Auburn Meadows has the largest community MAE,
  **36.44 days across 282 homes**, with mean signed error of **-5.85 days**.
  Edmonton's Laurel Green has the lowest, **27.13 days across 295 homes**.
  Different project mixes can explain these differences; this does not establish
  that location causes the errors.
- **Floor plan:** `sf-309` has the largest floor-plan MAE, **45.60 days across
  255 homes**, and accounts for **9.23%** of total absolute error. Its near-zero
  mean signed error (**-0.55 days**) shows why average bias alone can hide large
  individual errors.
- **Long builds:** For **234 homes lasting at least 300 days**, MAE is
  **78.81 days**, mean signed error is **-78.18 days**, and **98.29%** are
  underestimated. For **468 homes lasting under 120 days**, mean signed error is
  **+27.85 days**. The small overall bias hides substantial errors in opposite
  directions at the two ends of the duration range.

The duration-band analysis uses the eventual outcome, which is unknown when a
prediction is made. It diagnoses compression toward typical durations; it does
not supply an actionable rule for deciding which future home will be a long build.

## Interpretation and next steps

The immediate analytical priority is understanding extreme-duration errors and
checking whether they persist on a chronological holdout. The segment tables can
guide that investigation, but repeated inspection of the same validation results
must not be presented as independent validation of a newly tuned model.

The 30-home threshold is a reporting rule, not a statistical significance test.
No confidence intervals or multiple-comparison adjustments are estimated here.
Completed-only selection, temporal generalization, missing operational variables,
and the project's existing [privacy limitations](../docs/data_privacy.md) still apply.

## References

- [DuckDB CSV import](https://duckdb.org/docs/current/data/csv/overview)
- [DuckDB casting and TRY_CAST](https://duckdb.org/docs/current/sql/expressions/cast)
- [DuckDB command-line client](https://duckdb.org/docs/current/clients/cli/overview)
