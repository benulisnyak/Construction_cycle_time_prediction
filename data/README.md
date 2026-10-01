# Construction data

Both CSV files are included so the project can run from a fresh clone. These are
public versions of the source inputs that have been changed to synthetic data. Model preprocessing happens in memory and does not modify them.
See [data preparation and privacy](../docs/data_privacy.md) for the remaining
limitations. The other field values and row order are preserved.

| File | Rows | Purpose |
| --- | ---: | --- |
| `build_history.csv` | 4,684 | Completed and in-progress historical builds |
| `homes_to_predict.csv` | 578 | Homes requiring duration estimates |

## Input contract

Both inputs require `home_id`, the four start-time date fields below, and every
raw model feature. History additionally requires `construction_status` and
`construction_end_date`. Extra columns are allowed and ignored by the model.

| Fields | Interpretation |
| --- | --- |
| `home_id` | Home identifier such as `home_000001` |
| `beds`, `baths`, `half_baths`, `stories` | Property configuration |
| `sqft`, `lot_size_sqft` | Above-grade floor area and lot area, in square feet |
| `garage_size`, `basement_type`, `lot_type` | Categorical property attributes |
| `plan_code`, `product_line` | Floor-plan identifier and product category |
| `community`, `metro`, `permit_authority` | Geographic and permitting context |
| `site_manager` | Label such as `manager_01` for the assigned manager |
| `is_spec_home` | Boolean or 0/1 indicator for an unsold home at construction start |
| `selected_options_value` | Buyer-selected upgrades, in dollars, assumed known at start |
| `sale_date` | Sale date; may be missing for unsold properties |
| `permit_application_date`, `permit_issue_date` | Permit application and issuance dates |
| `construction_start_date` | Date when the duration estimate is made |
| `construction_end_date` | Historical outcome date, used only to form the target |
| `construction_status` | Historical eligibility: only completed builds train the model |

Dates use `YYYY-MM-DD`. Numeric features may contain missing values. Empty
categorical fields become `__missing__`; the literal basement label `None` is
preserved and normalized to `none`, rather than treated as missing. Malformed
non-empty dates and numeric values raise errors instead of being silently discarded.

Exact duplicates are removed from history before checking identifier uniqueness.
Conflicting rows with the same identifier require upstream resolution. Prediction
inputs must already have one row per home; they are never deduplicated or reordered.

`plan_name`, `data_source`, `final_inspection_date`,
`change_order_count`, and `trade_invoice_total` are present in the source files but
are not model inputs. The raw inputs do not need `cycle_days` or any engineered features.
