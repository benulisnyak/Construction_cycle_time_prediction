"""Shared input validation, cleaning, and construction-start features."""

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "garage_size", "basement_type", "lot_type", "start_season", "community",
    "metro", "permit_authority", "plan_code", "site_manager", "product_line",
]
FEATURES = [
    "beds", "baths", "half_baths", "stories", "lot_size_sqft", "garage_size",
    "basement_type", "lot_type", "is_spec_home", "start_season", "sqft",
    "community", "metro", "permit_authority", "product_line", "start_month",
    "days_sale_to_start", "permit_days", "plan_code", "site_manager",
    "permit_to_start_days", "selected_options_value",
]
DATE_COLUMNS = [
    "sale_date", "permit_application_date", "permit_issue_date", "construction_start_date",
]
DERIVED_FEATURES = {
    "start_season", "start_month", "days_sale_to_start", "permit_days", "permit_to_start_days",
}
SEASONS = {
    12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall",
}


def read_data(path: Path) -> pd.DataFrame:
    """Preserve literal 'None' basement labels and identifiers when reading CSV."""
    return pd.read_csv(
        path, keep_default_na=False, na_values=["", "NA", "N/A", "NaN", "nan", "null", "NULL"],
        dtype={"home_id": "string", **{c: "string" for c in CATEGORICAL_FEATURES if c != "start_season"}},
    )


def prepare_data(raw: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
    """Apply the same transformations to training and prediction inputs.

    Exact duplicates are removed only from history; prediction row order and
    cardinality are preserved. The square-footage correction is dataset-specific.
    """
    required = (set(FEATURES) - DERIVED_FEATURES) | set(DATE_COLUMNS) | {"home_id"}
    if training:
        required |= {"construction_status", "construction_end_date"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = raw.drop_duplicates().copy() if training else raw.copy()
    if data.empty:
        raise ValueError("Input contains no rows.")
    if data["home_id"].isna().any() or data["home_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every row must have a home_id.")
    if data["home_id"].duplicated().any():
        raise ValueError("home_id must be unique after removing exact historical duplicates.")

    for col in set(CATEGORICAL_FEATURES) - {"start_season"}:
        data[col] = data[col].astype("string").str.strip().str.lower().replace("", pd.NA)
        data[col] = data[col].fillna("__missing__").astype(str)
    for col in set(FEATURES) - set(CATEGORICAL_FEATURES) - DERIVED_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="raise").astype(float)
        if np.isinf(data[col]).any():
            raise ValueError(f"Column {col} contains infinite values.")
    # Observed extra-zero errors were approximately ten times their plan median.
    data.loc[data["sqft"] > 10_000, "sqft"] /= 10
    dates = DATE_COLUMNS + (["construction_end_date"] if training else [])
    for col in dates:
        data[col] = pd.to_datetime(data[col], errors="raise")
    data["days_sale_to_start"] = (data["construction_start_date"] - data["sale_date"]).dt.days
    # A sale after construction starts is unavailable at the prediction date.
    data.loc[data["days_sale_to_start"] < 0, "days_sale_to_start"] = np.nan
    data["permit_days"] = (data["permit_issue_date"] - data["permit_application_date"]).dt.days
    data["permit_to_start_days"] = (data["construction_start_date"] - data["permit_issue_date"]).dt.days
    data["start_month"] = data["construction_start_date"].dt.month
    data["start_season"] = data["start_month"].map(SEASONS).fillna("__missing__")
    if training:
        data["cycle_days"] = (data["construction_end_date"] - data["construction_start_date"]).dt.days
        completed = data["construction_status"].astype("string").str.strip().str.lower().eq("complete")
        data = data.loc[completed & (data["cycle_days"] > 1)].copy()
        if data.empty:
            raise ValueError("No completed builds have a cycle time greater than one day.")
    return data.reset_index(drop=True)
