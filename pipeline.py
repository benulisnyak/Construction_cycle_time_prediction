"""Validate, train, and generate construction cycle-time predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from construction_cycle.data import CATEGORICAL_FEATURES, FEATURES, prepare_data, read_data
from construction_cycle.modeling import MODEL_PARAMS, cross_validate, make_splits, manager_analysis, new_model


def run(history: Path, homes: Path, output: Path, *, folds=5, iterations=600, managers=False):
    """Write model, metrics, plots, and predictions to a dedicated run directory."""
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    raw_history = read_data(history)
    data = prepare_data(raw_history, training=True)
    prediction_data = prepare_data(read_data(homes))
    splits = make_splits(data, folds)
    output.mkdir(parents=True, exist_ok=True)

    metrics, oof = cross_validate(data, splits, iterations)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary = metrics.drop(columns="fold").mean().to_dict()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({
        "home_id": data["home_id"], "actual_cycle_days": data["cycle_days"],
        "predicted_cycle_days": oof,
    }).to_csv(output / "out_of_fold_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data["cycle_days"], oof, alpha=0.4, s=14)
    bounds = [min(data["cycle_days"].min(), oof.min()), max(data["cycle_days"].max(), oof.max())]
    ax.plot(bounds, bounds, "k--", linewidth=1)
    ax.set(xlabel="Actual cycle time (days)", ylabel="Predicted cycle time (days)", title="Out-of-fold construction cycle predictions")
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    model = new_model(iterations)
    model.fit(data[FEATURES], data["cycle_days"], cat_features=CATEGORICAL_FEATURES)
    estimates = model.predict(prediction_data[FEATURES])
    if not np.isfinite(estimates).all():
        raise ValueError("The model generated non-finite predictions.")
    predictions = pd.DataFrame({
        "home_id": prediction_data["home_id"],
        "predicted_cycle_days": np.maximum(np.rint(estimates), 1).astype(int),
    })
    predictions.to_csv(output / "predictions.csv", index=False)
    model.save_model(str(output / "model.cbm"))
    importance = pd.Series(model.get_feature_importance(), index=FEATURES, name="importance").sort_values()
    importance.rename_axis("feature").to_csv(output / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    importance.plot.barh(ax=ax)
    ax.set(xlabel="CatBoost feature importance", ylabel="", title="Construction cycle model features")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)

    if managers:
        results = manager_analysis(data, splits, iterations)
        results.to_csv(output / "manager_analysis.csv")
        fig, ax = plt.subplots(figsize=(9, max(4, len(results) * 0.3)))
        ax.scatter(results["mean"], results.index)
        valid = results["se"].notna()
        ax.errorbar(results.loc[valid, "mean"], results.index[valid], xerr=1.96 * results.loc[valid, "se"], fmt="none")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Actual minus expected cycle time (days)", ylabel="Site manager", title="Manager residuals with descriptive 95% intervals")
        fig.tight_layout()
        fig.savefig(output / "manager_analysis.png", dpi=160)
        plt.close(fig)

    config = {
        "history": str(history), "homes": str(homes), "raw_history_rows": len(raw_history),
        "eligible_training_rows": len(data), "prediction_rows": len(predictions),
        "folds": folds, "model_parameters": {**MODEL_PARAMS, "iterations": iterations},
        "features": FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "manager_analysis": managers,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/build_history.csv"))
    parser.add_argument("--predict", type=Path, default=Path("data/homes_to_predict.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--manager-analysis", action="store_true", help="Also fit manager-excluded models and summarize residuals.")
    args = parser.parse_args()
    try:
        summary = run(args.history, args.predict, args.output_dir, folds=args.folds, iterations=args.iterations, managers=args.manager_analysis)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
