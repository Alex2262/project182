"""Paired, fixed-origin evaluation of continuous and freshly replayed GDN states.

Run on CPU using the three existing 30k cosine checkpoints. No training, network,
or confirmation-set access. Saved compact predictions suffice to rebuild plots.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

SEEDS = [182, 183, 184]
ORIGINS = [2048, 4096, 8192, 16384]
POLICIES = ["full", "window256", "window2048"]
LABELS = ["Continuous full history", "Fresh state + latest 256", "Fresh state + latest 2,048"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(repo, out):
    import torch
    sys.path.insert(0, str(repo / "src"))
    from forecast_models import Forecaster
    from forecast_runtime import predict
    from forecast_data import sample_streams
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(182)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory, or --summarize-only.")
    source = out / "source"
    source.mkdir()
    shutil.copyfile(__file__, source / "evaluate_forecast_windows.py")
    shutil.copyfile(repo / "src/forecast_models.py", source / "forecast_models.py")
    shutil.copyfile(repo / "src/forecast_data.py", source / "forecast_data.py")
    started = time.monotonic()
    root = repo / "outputs/forecast_optimization"
    first = root / "alex_261123_ar1_opt30k_cosine_s182"
    with np.load(first / "data/validation.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    spec = json.loads(str(data["spec"]))
    assert spec["seed"] == 920000000 and spec["count"] == 512
    indexes = [data["lengths"].tolist().index(t) for t in ORIGINS]
    manifest = {
        "name": "alex_261124_ar1_paired_windows", "status": "running",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seeds": SEEDS, "checkpoint_step": 30000,
        "checkpoint_arm": "cosine", "origins": ORIGINS,
        "primary_origin": 16384, "primary_comparison": "window2048 minus full",
        "policies": POLICIES, "validation_seed": spec["seed"], "series": 512,
        "primary_metric": "squared error against true next-value mean a*y[t]",
        "additional_metrics": ["squared disagreement with full-history Bayes", "total expected MSE"],
        "device": "local CPU", "dtype": "float32", "batch_size": 16,
        "torch_threads": 2, "new_training_updates": 0, "cloud_jobs": 0,
        "cloud_compute_cost_usd": 0, "confirmation_set_opened": False,
        "full_predictions": "Reused saved GPU predictions; CPU replay verifies four series per seed at all origins.",
        "window_definition": "Reset recurrent and convolution states; replay last W observed transitions ending at the same t; predict y[t+1].",
        "uncertainty": "5000 paired bootstrap resamples of independent series, seed 261124; average losses across the three fixed models before resampling. Pointwise conditional intervals, not training-population intervals.",
        "torch_version": torch.__version__,
        "model_source_sha256": digest(repo / "src/forecast_models.py"),
        "data_sha256": digest(first / "data/validation.npz"), "checkpoints": [],
    }
    assert manifest["model_source_sha256"] == digest(repo / "results/2026-11-23-forecast-optimization/source/src/forecast_models.py")
    write_json(out / "manifest.json", manifest)
    # Verify dataset reproduction independently of model evaluation.
    regenerated = sample_streams(spec["seed"], spec["count"], max(ORIGINS), spec["bound"], spec["noise_std"])
    np.testing.assert_allclose(regenerated["values"], data["values"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(regenerated["theta"], data["theta"], rtol=0, atol=1e-15)
    compact = {k: data[k][:, indexes] for k in ["latent_mean", "bayes_mean", "bayes_epistemic_risk", "ridge_mean", "target"]}
    np.savez_compressed(out / "reference.npz", **compact, theta=data["theta"], origins=ORIGINS, spec=data["spec"])
    checks = {"dataset_reproduction": True, "identical_model_source": True,
              "regeneration_values_max_abs_error": float(np.max(np.abs(regenerated["values"]-data["values"]))),
              "regeneration_theta_max_abs_error": float(np.max(np.abs(regenerated["theta"]-data["theta"]))),
              "inference_uses_original_saved_observations": True, "seeds": []}
    values = data["values"]
    for seed in SEEDS:
        run_name = f"alex_261123_ar1_opt30k_cosine_s{seed}"
        directory = root / run_name
        checkpoint = directory / "state_step_30000.pt"
        checksum = digest(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = saved["config"]
        assert saved["completed_steps"] == 30000
        assert cfg["training"]["seed"] == seed and cfg["training"]["lengths"] == [64, 2048]
        assert cfg["training"]["schedule"] == "cosine_decay"
        assert cfg["model"]["family"] == "gated_deltanet"
        with np.load(directory / "data/validation.npz", allow_pickle=False) as other:
            for key in ["values", "theta", "latent_mean", "target", "spec", "lengths"]:
                np.testing.assert_array_equal(data[key], other[key])
        model = Forecaster(cfg["model"]).eval()
        model.load_state_dict(saved["model"])
        with np.load(directory / "validation_predictions_step_30000.npz", allow_pickle=False) as z:
            full = z["prediction"][:, indexes]
        predictions = np.empty((512, len(ORIGINS), len(POLICIES)), dtype=np.float64)
        predictions[:, :, 0] = full
        selected = [0, 1, 255, 511]
        with torch.inference_mode():
            replay = predict(model, {"values": values[selected], "lengths": np.array(ORIGINS)}, "cpu", batch_size=4)
            replay_error = float(np.max(np.abs(replay - full[selected])))
            np.testing.assert_allclose(replay, full[selected], rtol=0, atol=3e-6)
            # Validate complete replay versus carried-state chunking on a real window.
            observed = values[:2, 8192-256:8192+1]
            x = torch.tensor(np.stack((observed[:, :-1], observed[:, 1:]), axis=-1), dtype=torch.float32)
            together = model(x)
            state, parts = None, []
            for offset in range(0, 256, 64):
                y, state = model(x[:, offset:offset+64], state, return_state=True)
                parts.append(y)
            chunk_error = float((together-torch.cat(parts, dim=1)).abs().max())
            torch.testing.assert_close(together, torch.cat(parts, dim=1), rtol=0, atol=3e-6)
            # Target/future values and data before the window are never passed in.
            for j, origin in enumerate(ORIGINS):
                for k, width in [(1, 256), (2, 2048)]:
                    start = origin - width
                    assert start >= 0
                    for b in range(0, 512, 16):
                        observed = values[b:b+16, start:origin+1]
                        tokens = np.stack((observed[:, :-1], observed[:, 1:]), axis=-1)
                        assert tokens.shape[1:] == (width, 2)
                        np.testing.assert_array_equal(tokens[:, -1, 1], values[b:b+16, origin])
                        y = model(torch.as_tensor(tokens, dtype=torch.float32))
                        predictions[b:b+16, j, k] = y[:, -1].double().numpy()
                    print(json.dumps({"seed": seed, "origin": origin, "policy": POLICIES[k], "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
        # At t=2048 the full and window2048 treatments receive identical histories.
        identity_error = float(np.max(np.abs(predictions[:, 0, 2]-predictions[:, 0, 0])))
        np.testing.assert_allclose(predictions[:, 0, 2], predictions[:, 0, 0], rtol=0, atol=3e-6)
        assert np.isfinite(predictions).all()
        for key, value in model.state_dict().items():
            assert torch.equal(value, saved["model"][key]), key
        assert digest(checkpoint) == checksum
        np.savez_compressed(out / f"predictions_s{seed}.npz", prediction=predictions, origins=ORIGINS, policies=POLICIES)
        write_json(out / f"config_s{seed}.json", cfg)
        checks["seeds"].append({"seed": seed, "full_replay_max_abs_error": replay_error,
                                "chunk_equivalence_max_abs_error": chunk_error,
                                "full_equals_window2048_at_2048_max_abs_error": identity_error,
                                "weights_unchanged": True, "checkpoint_unchanged": True})
        manifest["checkpoints"].append({"seed": seed, "path": str(checkpoint.relative_to(repo)), "sha256": checksum})
        write_json(out / "verification.json", checks)
        write_json(out / "manifest.json", manifest)
    manifest.update(status="complete", elapsed_seconds=time.monotonic()-started,
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "manifest.json", manifest)
    summarize(out)


def summarize(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(out / "reference.npz", allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    predictions = np.stack([np.load(out / f"predictions_s{s}.npz")["prediction"] for s in SEEDS])
    assert predictions.shape == (3, 512, 4, 3) and np.isfinite(predictions).all()
    error = (predictions-ref["latent_mean"][None, :, :, None])**2
    gap = (predictions-ref["bayes_mean"][None, :, :, None])**2
    seed_means = error.mean(1)
    rng = np.random.default_rng(261124)
    indexes = rng.integers(0, 512, size=(5000, 512))
    rows, contrasts = [], []
    for j, origin in enumerate(ORIGINS):
        for k, policy in enumerate(POLICIES):
            means = seed_means[:, j, k]
            rows.append({"origin": origin, "policy": policy, "reducible_error": float(means.mean()),
                         "seed_sd": float(means.std(ddof=1)), "total_expected_mse": float(.25+means.mean()),
                         "bayes_disagreement": float(gap[:, :, j, k].mean()),
                         **{f"seed{s}_error": float(means[i]) for i,s in enumerate(SEEDS)}})
            if k:
                differences = error[:, :, j, k]-error[:, :, j, 0]
                conditional = differences.mean(0)
                bounds = np.quantile(conditional[indexes].mean(1), [.025, .975])
                contrasts.append({"origin": origin, "policy": policy,
                                  "mean_window_minus_full": float(conditional.mean()),
                                  "paired_series_bootstrap_95": bounds.tolist(),
                                  "per_seed_differences": differences.mean(1).tolist(),
                                  "bayes_gap_window_minus_full": float((gap[:, :, j, k]-gap[:, :, j, 0]).mean()),
                                  "relative_reducible_change": float(means.mean()/seed_means[:, j, 0].mean()-1),
                                  "relative_total_mse_change": float((means.mean()-seed_means[:, j, 0].mean())/(.25+seed_means[:, j, 0].mean()))})
    for filename, items in [("metrics.csv", rows), ("paired_contrasts.csv", contrasts)]:
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader(); writer.writerows(items)
    summary = {"rows": rows, "contrasts": contrasts, "origins": ORIGINS,
               "bayes_epistemic_risk": ref["bayes_epistemic_risk"].mean(0).tolist(),
               "ridge_reducible_error": ((ref["ridge_mean"]-ref["latent_mean"])**2).mean(0).tolist(),
               "bayes_empirical_reducible_error": ((ref["bayes_mean"]-ref["latent_mean"])**2).mean(0).tolist(),
               "bootstrap_draws": 5000, "bootstrap_seed": 261124,
               "interval_scope": "Pointwise paired series bootstrap, conditional on these three trained models; training seeds not resampled.",
               "confirmation_set_opened": False}
    write_json(out / "summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
    colors = ["#2563a6", "#d95f02", "#298050"]
    for k, label in enumerate(LABELS):
        means = seed_means[:, :, k]
        axes[0].plot(ORIGINS, means.mean(0), "o-", color=colors[k], label=label)
        axes[0].fill_between(ORIGINS, means.min(0), means.max(0), color=colors[k], alpha=.12)
    axes[0].plot(ORIGINS, summary["bayes_epistemic_risk"], "k:", label="Full-history Bayes risk")
    axes[0].plot(ORIGINS, summary["ridge_reducible_error"], "--", color="#777777", label="Full-history online ridge")
    axes[0].set(yscale="log", ylabel="Error against noise-free next-value mean", title="Forecast error; bands show training-seed range")
    axes[0].legend(fontsize=8)
    for k in [1,2]:
        items = [c for c in contrasts if c["policy"] == POLICIES[k]]
        means = np.array([c["mean_window_minus_full"] for c in items])
        ci = np.array([c["paired_series_bootstrap_95"] for c in items])
        axes[1].errorbar(np.array(ORIGINS)*(0.97 if k==1 else 1.03), means,
                         yerr=np.maximum(np.stack((means-ci[:,0], ci[:,1]-means)), 0),
                         fmt="o-", capsize=3, color=colors[k], label=LABELS[k])
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(ylabel="Window error minus continuous-history error",
                title="Paired difference; pointwise 95% series bootstrap")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks(ORIGINS, [f"{t:,}" for t in ORIGINS])
        ax.set_xlabel("Same forecast origin (observed transitions)"); ax.grid(alpha=.15)
    fig.suptitle("Does keeping the old state help?\nThree frozen 30k cosine GDNs · 512 shared validation series", fontsize=12)
    for ext in ["png", "svg"]: fig.savefig(out / f"paired_windows.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"primary_origin": 16384, "rows": rows[-3:], "contrasts": contrasts[-2:]}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    if a.summarize_only: summarize(a.output)
    else: run(a.repo, a.output)
