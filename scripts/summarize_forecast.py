"""Build a shareable, validation-only report from the four approved pilot runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    ("gated_deltanet", "short"): "GDN · train 64",
    ("gated_deltanet", "mixed"): "GDN · train 64/256",
    ("transformer_nope", "short"): "Transformer · train 64",
    ("transformer_nope", "mixed"): "Transformer · train 64/256",
}
COLORS = {"gated_deltanet": "#146d9c", "transformer_nope": "#cc642d"}


def json_save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(raw, output, manifest_path, repo):
    sys.path.insert(0, str(repo / "src"))
    from forecast_data import score
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    runs, table, data = [], [], None
    for item in manifest["runs"]:
        directory = raw / item["run_name"]
        config = yaml.safe_load((directory / "config.yaml").read_text())
        assert config["training"]["seed"] == 182
        assert config["training"]["steps"] == 2000
        assert config["training"]["total_training_tokens"] == 4096000
        saved = np.load(directory / "data/validation.npz", allow_pickle=False)
        current = {name: saved[name] for name in saved.files}
        if data is None:
            data = current
        else:
            for key in ("values", "theta", "latent_mean", "spec"):
                np.testing.assert_array_equal(data[key], current[key])
            # Derived SciPy moments can differ at floating-point roundoff across
            # worker CPUs; the sampled observations and coefficients must be exact.
            for key in ("bayes_mean", "bayes_risk", "bayes_epistemic_risk"):
                np.testing.assert_allclose(data[key], current[key], rtol=1e-13, atol=1e-13)
        config_out = output / "configs"
        config_out.mkdir(exist_ok=True)
        shutil.copyfile(directory / "config.yaml", config_out / f"{item['run_name']}.yaml")
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        history = json.loads((directory / "training.json").read_text())
        assert history[-1]["step"] == 2000
        results, predictions = {}, {}
        for step in (500, 1000, 2000):
            path = directory / f"validation_predictions_step_{step}.npz"
            with np.load(path) as values:
                pred = values["prediction"]
            rows = score(pred, data)
            recorded = json.loads((directory / f"validation_step_{step}.json").read_text())
            for calculated, stored in zip(rows, recorded):
                for key in ("conditional_mean_error", "excess_vs_bayes", "observed_mse", "expected_predictive_mse"):
                    np.testing.assert_allclose(calculated[key], stored[key], rtol=1e-10, atol=1e-12)
            results[step] = rows
            predictions[step] = pred
            shutil.copyfile(path, prediction_dir / f"{item['run_name']}_step_{step}.npz")
            for row in rows:
                table.append({"run_name": item["run_name"], "family": item["family"],
                              "regime": item["regime"], "training_seed": 182, "training_steps": step, **row})
        runs.append({**item, "label": LABELS[(item["family"], item["regime"])],
                     "history": history, "results": results, "predictions": predictions,
                     "config": config})
    assert len(runs) == 4
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez_compressed(output / "validation_reference.npz",
                        **{k: v for k, v in data.items() if k != "values"})
    lengths = data["lengths"]
    bayes = data["bayes_epistemic_risk"].mean(0)
    sigma2 = json.loads(str(data["spec"]))["noise_std"] ** 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for run in runs:
        rows = run["results"][2000]
        color = COLORS[run["family"]]
        style = "-" if run["regime"] == "short" else "--"
        for axis, metric in zip(axes, ("conditional_mean_error", "excess_vs_bayes", "expected_predictive_mse")):
            means = np.array([row[metric] for row in rows])
            se = np.array([row[metric + "_series_se"] for row in rows])
            axis.plot(lengths, means, style, color=color, marker="o", markersize=3, label=run["label"])
            axis.fill_between(lengths, np.maximum(means - 1.96 * se, 1e-8), means + 1.96 * se, color=color, alpha=.10)
    axes[0].plot(lengths, bayes, "k:", linewidth=2, label="Exact Bayes reference")
    ridge_error = ((data["ridge_mean"] - data["latent_mean"]) ** 2).mean(0)
    axes[0].plot(lengths, ridge_error, color="#777777", linestyle="-.", label="Online ridge reference")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].plot(lengths, sigma2 + bayes, "k:", linewidth=2, label="Exact Bayes reference")
    axes[2].axhline(sigma2, color="#777777", linestyle="-.", label="Innovation-noise floor")
    for axis, title, ylabel in zip(axes,
            ("Reducible forecast error", "Gap to the available Bayes forecast", "Total expected forecast error"),
            ("MSE relative to a × current value", "MSE relative to Bayes mean", "Expected next-value MSE")):
        axis.set_xscale("log", base=2)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Observed transitions")
        axis.set_ylabel(ylabel)
        axis.axvline(64, color="gray", alpha=.45, linewidth=1)
        axis.axvline(256, color="gray", alpha=.45, linewidth=1, linestyle="--")
        axis.grid(alpha=.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("AR(1) streaming pilot · 2,000 updates · one training seed\n512 validation series; bands are ±1.96 series SE, not training-seed uncertainty", fontsize=12)
    fig.savefig(output / "forecast_curves.png", dpi=180)
    fig.savefig(output / "forecast_curves.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for run in runs:
        color, style = COLORS[run["family"]], "-" if run["regime"] == "short" else "--"
        steps = [h["step"] for h in run["history"]]
        losses = [h["loss"] for h in run["history"]]
        axes[0].plot(steps, losses, style, color=color, label=run["label"], alpha=.8)
        for axis, length in zip(axes[1:], (64, 4096)):
            index = lengths.tolist().index(length)
            means = [run["results"][step][index]["conditional_mean_error"] for step in (500, 1000, 2000)]
            axis.plot([500, 1000, 2000], means, style, color=color, marker="o", label=run["label"])
            axis.set_yscale("log")
            axis.set_title(f"Validation after {length:,} observations")
            axis.set_ylabel("Reducible forecast MSE")
    axes[0].set_title("Logged training loss")
    axes[0].set_ylabel("Noisy-target MSE (one logged batch)")
    for axis in axes:
        axis.set_xlabel("Training updates")
        axis.grid(alpha=.15)
    for axis, length in zip(axes[1:], (64, 4096)):
        axis.axhline(bayes[lengths.tolist().index(length)], color="black", linestyle=":")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Convergence checks · checkpoint comparisons are exploratory", fontsize=12)
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)

    summary, contrasts = [], []
    i64, i4096 = lengths.tolist().index(64), lengths.tolist().index(4096)
    for run in runs:
        at64, at4096 = run["results"][2000][i64], run["results"][2000][i4096]
        previous = run["results"][1000][i4096]["conditional_mean_error"]
        row = {"label": run["label"], "run_name": run["run_name"],
               "error_64": at64["conditional_mean_error"], "error_4096": at4096["conditional_mean_error"],
               "total_mse_4096": at4096["expected_predictive_mse"],
               "long_vs_short_error_reduction_percent": 100 * (1 - at4096["conditional_mean_error"] / at64["conditional_mean_error"]),
               "long_history_bayes_gap": at4096["excess_vs_bayes"],
               "change_1000_to_2000_long_error_percent": 100 * (at4096["conditional_mean_error"] / previous - 1),
               "wandb_url": run["wandb_url"]}
        summary.append(row)
        for earlier, later in ((64, 4096), (256, 4096)):
            ix, iy = lengths.tolist().index(earlier), lengths.tolist().index(later)
            errors = (run["predictions"][2000] - data["latent_mean"]) ** 2
            difference = errors[:, iy] - errors[:, ix]
            contrasts.append({"comparison": "long minus short history", "run_name": run["run_name"],
                              "earlier_length": earlier, "later_length": later,
                              "paired_mean_difference": float(difference.mean()),
                              "paired_series_se": float(difference.std(ddof=1) / np.sqrt(len(difference)))})
    for family in COLORS:
        short = next(r for r in runs if r["family"] == family and r["regime"] == "short")
        mixed = next(r for r in runs if r["family"] == family and r["regime"] == "mixed")
        difference = ((mixed["predictions"][2000][:, i4096] - data["latent_mean"][:, i4096]) ** 2 -
                      (short["predictions"][2000][:, i4096] - data["latent_mean"][:, i4096]) ** 2)
        contrasts.append({"comparison": "mixed minus short training at 4096", "family": family,
                          "paired_mean_difference": float(difference.mean()),
                          "paired_series_se": float(difference.std(ddof=1) / np.sqrt(len(difference)))})
    json_save(output / "summary.json", {"runs": summary, "paired_validation_contrasts": contrasts,
               "bayes_epistemic_at_64": float(bayes[i64]), "bayes_epistemic_at_4096": float(bayes[i4096]),
               "confirmation_set_opened": False, "training_seeds": [182],
               "generated_at_utc": datetime.now(timezone.utc).isoformat()})
    json_save(output / "training_histories.json", {r["run_name"]: r["history"] for r in runs})
    json_save(output / "manifest.json", manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(summarize(args.raw, args.output, args.manifest, args.repo), indent=2))
