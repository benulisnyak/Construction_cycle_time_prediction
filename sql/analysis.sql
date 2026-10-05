-- Construction data-quality audit and out-of-fold model error analysis.
-- Engine: DuckDB. Run from the repository root; see sql/README.md.
-- All objects are temporary. No source files, model artifacts, or databases are written.

-- Preserve original text for the audit, including literal 'None' basement labels.
CREATE OR REPLACE TEMP VIEW raw_history AS
SELECT * FROM read_csv('data/build_history.csv', header = true, all_varchar = true);

-- Remove exact duplicate rows before joining; never arbitrarily choose between
-- conflicting records that happen to share an ID.
CREATE OR REPLACE TEMP VIEW distinct_history AS
SELECT DISTINCT * FROM raw_history;

CREATE OR REPLACE TEMP VIEW typed_history AS
SELECT
    home_id,
    COALESCE(NULLIF(LOWER(TRIM(community)), ''), '__missing__') AS community,
    COALESCE(NULLIF(LOWER(TRIM(metro)), ''), '__missing__') AS metro,
    COALESCE(NULLIF(LOWER(TRIM(plan_code)), ''), '__missing__') AS plan_code,
    LOWER(TRIM(construction_status)) AS status,
    TRY_CAST(construction_start_date AS DATE) AS start_date,
    TRY_CAST(construction_end_date AS DATE) AS end_date,
    DATE_DIFF('day', TRY_CAST(construction_start_date AS DATE),
                     TRY_CAST(construction_end_date AS DATE)) AS cycle_days
FROM distinct_history;

-- 1. Row counts and exact duplicates (raw history grain).
SELECT
    (SELECT COUNT(*) FROM raw_history) AS raw_rows,
    COUNT(*) AS deduplicated_rows,
    (SELECT COUNT(*) FROM raw_history) - COUNT(*) AS exact_duplicate_rows,
    COUNT(DISTINCT home_id) AS distinct_home_ids
FROM distinct_history;

-- 2. Missing values versus invalid values (deduplicated history grain).
-- TRY_CAST preserves the ability to count malformed source dates explicitly.
-- These issues can overlap; do not sum them to calculate excluded rows.
WITH quality_counts AS (
    SELECT 'missing sale date' AS issue,
           COUNT(*) FILTER (WHERE NULLIF(TRIM(sale_date), '') IS NULL) AS affected_rows
    FROM distinct_history
    UNION ALL
    SELECT 'missing beds', COUNT(*) FILTER (WHERE NULLIF(TRIM(beds), '') IS NULL)
    FROM distinct_history
    UNION ALL
    SELECT 'missing lot area', COUNT(*) FILTER (WHERE NULLIF(TRIM(lot_size_sqft), '') IS NULL)
    FROM distinct_history
    UNION ALL
    SELECT 'missing basement category', COUNT(*) FILTER (WHERE NULLIF(TRIM(basement_type), '') IS NULL)
    FROM distinct_history
    UNION ALL
    SELECT 'literal None basement category (valid)',
           COUNT(*) FILTER (WHERE LOWER(TRIM(basement_type)) = 'none')
    FROM distinct_history
    UNION ALL
    SELECT 'square footage above 10000 (source correction candidates)',
           COUNT(*) FILTER (WHERE TRY_CAST(sqft AS DOUBLE) > 10000)
    FROM distinct_history
    UNION ALL
    SELECT 'community labels changed by whitespace trimming',
           COUNT(*) FILTER (WHERE community <> TRIM(community))
    FROM distinct_history
    UNION ALL
    SELECT 'malformed non-empty start date',
           COUNT(*) FILTER (WHERE NULLIF(TRIM(construction_start_date), '') IS NOT NULL
                           AND TRY_CAST(construction_start_date AS DATE) IS NULL)
    FROM distinct_history
    UNION ALL
    SELECT 'malformed non-empty end date',
           COUNT(*) FILTER (WHERE NULLIF(TRIM(construction_end_date), '') IS NOT NULL
                           AND TRY_CAST(construction_end_date AS DATE) IS NULL)
    FROM distinct_history
)
SELECT issue, affected_rows,
       ROUND(100.0 * affected_rows / NULLIF((SELECT COUNT(*) FROM distinct_history), 0), 2) AS pct_of_history
FROM quality_counts
ORDER BY affected_rows DESC, issue;

-- 3. Mutually exclusive eligibility buckets, matching the Python target rule.
CREATE OR REPLACE TEMP VIEW eligibility_audit AS
SELECT *,
    CASE
        WHEN status IS DISTINCT FROM 'complete' THEN 'not completed'
        WHEN start_date IS NULL OR end_date IS NULL THEN 'completed: missing or invalid dates'
        WHEN cycle_days <= 1 THEN 'completed: duration <= 1 day'
        ELSE 'eligible completed build'
    END AS eligibility
FROM typed_history;

SELECT eligibility, COUNT(*) AS homes
FROM eligibility_audit
GROUP BY eligibility
ORDER BY homes DESC;

CREATE OR REPLACE TEMP VIEW eligible_history AS
SELECT home_id, metro, community, plan_code, cycle_days
FROM eligibility_audit
WHERE eligibility = 'eligible completed build';

-- Use historical out-of-fold predictions, not the unlabelled prediction set.
CREATE OR REPLACE TEMP VIEW oof_predictions AS
SELECT home_id,
       TRY_CAST(actual_cycle_days AS DOUBLE) AS actual_days,
       TRY_CAST(predicted_cycle_days AS DOUBLE) AS predicted_days
FROM read_csv('reports/out_of_fold_predictions.csv', header = true, all_varchar = true);

-- 4. Join checks: fail before producing metrics if the inputs are inconsistent.
CREATE OR REPLACE TEMP VIEW join_checks AS
SELECT 'conflicting historical home IDs' AS check_name, COUNT(*) AS failures
FROM (SELECT home_id FROM distinct_history GROUP BY home_id HAVING COUNT(*) > 1)
UNION ALL
SELECT 'missing historical home IDs', COUNT(*)
FROM distinct_history WHERE NULLIF(TRIM(home_id), '') IS NULL
UNION ALL
SELECT 'duplicate out-of-fold home IDs', COUNT(*)
FROM (SELECT home_id FROM oof_predictions GROUP BY home_id HAVING COUNT(*) > 1)
UNION ALL
SELECT 'missing out-of-fold home IDs', COUNT(*)
FROM oof_predictions WHERE NULLIF(TRIM(home_id), '') IS NULL
UNION ALL
SELECT 'eligible homes without out-of-fold predictions', COUNT(*)
FROM eligible_history h
WHERE NOT EXISTS (SELECT 1 FROM oof_predictions p WHERE p.home_id = h.home_id)
UNION ALL
SELECT 'out-of-fold predictions without eligible history', COUNT(*)
FROM oof_predictions p
WHERE NOT EXISTS (SELECT 1 FROM eligible_history h WHERE h.home_id = p.home_id)
UNION ALL
SELECT 'missing or non-finite prediction values', COUNT(*)
FROM oof_predictions
WHERE actual_days IS NULL OR predicted_days IS NULL
   OR NOT ISFINITE(actual_days) OR NOT ISFINITE(predicted_days)
UNION ALL
SELECT 'reported actual duration disagrees with source dates', COUNT(*)
FROM eligible_history h JOIN oof_predictions p USING (home_id)
WHERE ABS(h.cycle_days - p.actual_days) > 0.000001
UNION ALL
SELECT 'empty eligible history', CASE WHEN COUNT(*) = 0 THEN 1 ELSE 0 END
FROM eligible_history;

SELECT check_name, failures FROM join_checks ORDER BY check_name;
SELECT CASE WHEN SUM(failures) = 0 THEN 'PASS: one-to-one join and complete coverage'
            ELSE ERROR('Join validation failed. Inspect join_checks before interpreting metrics.')
       END AS join_validation
FROM join_checks;

CREATE OR REPLACE TEMP VIEW prediction_errors AS
SELECT h.*, p.predicted_days,
       p.predicted_days - h.cycle_days AS signed_error_days,
       ABS(p.predicted_days - h.cycle_days) AS absolute_error_days,
       POWER(p.predicted_days - h.cycle_days, 2) AS squared_error_days
FROM eligible_history h
JOIN oof_predictions p USING (home_id);

-- 5. Pooled metrics across all homes, not an unweighted mean of fold metrics.
-- Negative signed error = underestimation. This sign differs from the manager
-- report, which defines its residual as actual minus expected duration.
SELECT COUNT(*) AS homes,
       ROUND(AVG(absolute_error_days), 2) AS mae_days,
       ROUND(SQRT(AVG(squared_error_days)), 2) AS rmse_days,
       ROUND(AVG(signed_error_days), 2) AS mean_signed_error_days,
       ROUND(MEDIAN(absolute_error_days), 2) AS median_absolute_error_days,
       ROUND(100.0 * AVG(CASE WHEN absolute_error_days <= 30 THEN 1.0 ELSE 0.0 END), 2) AS pct_within_30_days
FROM prediction_errors;

-- 6. Community diagnostics. Group by metro as well as community so identically
-- named communities in different cities are not merged. Suppress groups < 30.
WITH segments AS (
    SELECT metro, community, COUNT(*) AS homes,
           AVG(absolute_error_days) AS mae_days,
           SQRT(AVG(squared_error_days)) AS rmse_days,
           AVG(signed_error_days) AS mean_signed_error_days
    FROM prediction_errors
    GROUP BY metro, community
    HAVING COUNT(*) >= 30
)
SELECT DENSE_RANK() OVER (ORDER BY mae_days DESC) AS mae_rank,
       metro, community, homes,
       ROUND(mae_days, 2) AS mae_days,
       ROUND(rmse_days, 2) AS rmse_days,
       ROUND(mean_signed_error_days, 2) AS mean_signed_error_days
FROM segments
ORDER BY mae_rank, metro, community;

-- 7. Floor-plan errors and share of total absolute error.
-- Compute the window denominator before filtering small groups, so the share
-- still refers to ALL homes rather than only displayed groups.
WITH plans AS (
    SELECT plan_code, COUNT(*) AS homes,
           AVG(absolute_error_days) AS mae_days,
           AVG(signed_error_days) AS mean_signed_error_days,
           SUM(absolute_error_days) AS total_absolute_error
    FROM prediction_errors
    GROUP BY plan_code
), contributions AS (
    SELECT *, 100.0 * total_absolute_error / NULLIF(SUM(total_absolute_error) OVER (), 0) AS pct_total_absolute_error
    FROM plans
)
SELECT plan_code, homes, ROUND(mae_days, 2) AS mae_days,
       ROUND(mean_signed_error_days, 2) AS mean_signed_error_days,
       ROUND(pct_total_absolute_error, 2) AS pct_total_absolute_error
FROM contributions
WHERE homes >= 30
ORDER BY mae_days DESC, plan_code;

-- 8. Retrospective duration diagnostics; actual duration is unavailable at
-- construction start, so these bands are not deployable routing rules.
WITH duration_bands AS (
    SELECT *, CASE
        WHEN cycle_days < 120 THEN '01: under 120 days'
        WHEN cycle_days < 180 THEN '02: 120-179 days'
        WHEN cycle_days < 240 THEN '03: 180-239 days'
        WHEN cycle_days < 300 THEN '04: 240-299 days'
        ELSE '05: 300+ days'
    END AS duration_band
    FROM prediction_errors
)
SELECT duration_band, COUNT(*) AS homes,
       ROUND(AVG(absolute_error_days), 2) AS mae_days,
       ROUND(AVG(signed_error_days), 2) AS mean_signed_error_days,
       ROUND(100.0 * AVG(CASE WHEN signed_error_days < 0 THEN 1.0 ELSE 0.0 END), 2) AS pct_underestimated
FROM duration_bands
GROUP BY duration_band
ORDER BY duration_band;
