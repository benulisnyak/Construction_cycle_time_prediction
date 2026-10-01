"""Regression checks for preprocessing and the complete artifact workflow."""

from pathlib import Path
import tempfile
import unittest

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from construction_cycle.data import CATEGORICAL_FEATURES, FEATURES, prepare_data, read_data
from construction_cycle.modeling import make_splits
from pipeline import run


def example_data(rows=12):
    data = pd.DataFrame({col: [1] * rows for col in set(FEATURES) - {"start_season"}})
    for col in CATEGORICAL_FEATURES:
        data[col] = "sample"
    data["home_id"] = [f"H{i:04}" for i in range(rows)]
    data["basement_type"] = "None"
    data["community"] = " Riverbend "
    data["sqft"] = np.arange(rows) * 50 + 1500
    data["sale_date"] = "2024-01-01"
    data["permit_application_date"] = "2024-01-02"
    data["permit_issue_date"] = "2024-01-05"
    data["construction_start_date"] = "2024-02-01"
    data["construction_end_date"] = pd.Timestamp("2024-02-01") + pd.to_timedelta(np.arange(rows) * 7 + 100, unit="D")
    data["construction_status"] = "Complete"
    return data


class PreprocessingTests(unittest.TestCase):
    def test_training_and_prediction_features_match(self):
        raw = example_data()
        raw.loc[0, "sqft"] = 15000
        raw.loc[0, "sale_date"] = "2024-03-01"
        training = prepare_data(raw, training=True)
        prediction = prepare_data(raw.drop(columns=["construction_end_date", "construction_status"]))
        pd.testing.assert_frame_equal(training[FEATURES], prediction[FEATURES])
        self.assertEqual(training.loc[0, "sqft"], 1500)
        self.assertTrue(pd.isna(training.loc[0, "days_sale_to_start"]))
        self.assertEqual(training.loc[0, "community"], "riverbend")

    def test_literal_none_is_distinct_from_missing(self):
        raw = example_data()
        raw.loc[1, "basement_type"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            raw.to_csv(path, index=False)
            clean = prepare_data(read_data(path))
        self.assertEqual(clean.loc[0, "basement_type"], "none")
        self.assertEqual(clean.loc[1, "basement_type"], "__missing__")

    def test_targets_and_duplicates(self):
        raw = example_data()
        raw.loc[0, "construction_status"] = "In Progress"
        raw.loc[1, "construction_end_date"] = pd.Timestamp("2024-01-01")
        raw.loc[2, "construction_end_date"] = pd.Timestamp("2024-02-02")
        raw.loc[3, "construction_end_date"] = pd.NaT
        raw = pd.concat([raw, raw.iloc[[4]]], ignore_index=True)
        clean = prepare_data(raw, training=True)
        self.assertEqual(len(clean), 8)
        self.assertTrue((clean.cycle_days > 1).all())

    def test_prediction_order(self):
        raw = example_data().iloc[::-1]
        self.assertEqual(prepare_data(raw).home_id.tolist(), raw.home_id.tolist())

    def test_missing_columns_and_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            prepare_data(example_data().drop(columns="sqft"))
        raw = example_data()
        raw.loc[1, "home_id"] = raw.loc[0, "home_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_data(raw)

    def test_rare_plans_fail_before_training(self):
        data = prepare_data(example_data(), training=True)
        data.loc[0, "plan_code"] = "rare"
        with self.assertRaises(ValueError):
            make_splits(data)

    def test_each_home_is_validated_once(self):
        data = prepare_data(example_data(), training=True)
        splits = make_splits(data, folds=3)
        indices = np.concatenate([valid for _, valid in splits])
        np.testing.assert_array_equal(np.sort(indices), np.arange(len(data)))
        for train, valid in splits:
            self.assertFalse(set(train) & set(valid))

    def test_outcomes_are_not_features(self):
        self.assertFalse(set(FEATURES) & {
            "construction_end_date", "final_inspection_date", "construction_status",
            "trade_invoice_total", "change_order_count", "home_id", "cycle_days",
        })


class PipelineTests(unittest.TestCase):
    def test_artifacts_and_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = example_data()
            homes = example_data(3).drop(columns=["construction_end_date", "construction_status"])
            history.to_csv(root / "history.csv", index=False)
            homes.to_csv(root / "homes.csv", index=False)
            output = root / "results"
            metrics = run(root / "history.csv", root / "homes.csv", output, folds=3, iterations=5, managers=True)
            self.assertTrue(all(np.isfinite(list(metrics.values()))))
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(predictions.home_id.tolist(), homes.home_id.tolist())
            self.assertTrue((predictions.predicted_cycle_days >= 1).all())
            self.assertEqual(len(pd.read_csv(output / "out_of_fold_predictions.csv")), len(history))
            for name in ["metrics.json", "fold_metrics.csv", "run_config.json", "feature_importance.csv",
                         "feature_importance.png", "predicted_vs_actual.png", "manager_analysis.csv", "manager_analysis.png"]:
                self.assertGreater((output / name).stat().st_size, 0)
            model = CatBoostRegressor()
            model.load_model(str(output / "model.cbm"))
            estimates = model.predict(prepare_data(read_data(root / "homes.csv"))[FEATURES])
            np.testing.assert_array_equal(predictions.predicted_cycle_days, np.maximum(np.rint(estimates), 1).astype(int))


if __name__ == "__main__":
    unittest.main()
