"""CPU numerical checks, data-leakage checks, and a reference-only pilot figure."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.integrate import quad
import torch

from forecast_data import fixed_dataset, posterior_moments, references, sample_streams, score, transition_tokens
from forecast_models import Forecaster, delta_chunk
from forecast_runtime import load_config, predict, write_json


def check_reference():
    data = sample_streams(18210, 12, 64)
    lengths = [1, 2, 3, 16, 64]
    ref = references(data["values"], lengths)
    error = 0.0
    for series in range(12):
        for column, length in enumerate(lengths):
            x, y = data["values"][series, :length], data["values"][series, 1:length + 1]
            sxx, sxy = x @ x, x @ y
            mode = np.clip(sxy / sxx, -.95, .95) if sxx > 0 else 0.
            def weight(theta):
                return np.exp((sxy * (theta - mode) - 0.5 * sxx * (theta ** 2 - mode ** 2)) / .5 ** 2)
            mass = quad(weight, -.95, .95, epsabs=1e-12)[0]
            mean = quad(lambda theta: theta * weight(theta), -.95, .95, epsabs=1e-12)[0] / mass
            var = quad(lambda theta: (theta - mean) ** 2 * weight(theta), -.95, .95, epsabs=1e-12)[0] / mass
            actual = [ref["bayes_parameter_mean"][series, column], ref["bayes_parameter_variance"][series, column]]
            error = max(error, float(np.max(np.abs(np.array(actual) - [mean, var]))))
            np.testing.assert_allclose(actual, [mean, var], atol=2e-9, rtol=2e-8)
    # Prefix invariance and no future-target leakage in reference forecasts.
    shorter = sample_streams(18210, 12, 16)
    np.testing.assert_array_equal(shorter["values"], data["values"][:, :18])
    changed = data["values"].copy()
    changed[:, 17:] += 100
    before = references(data["values"], [16])
    after = references(changed, [16])
    np.testing.assert_array_equal(before["bayes_mean"], after["bayes_mean"])
    # Check near-uniform posterior calculations separately from random examples.
    mean, variance = posterior_moments(np.array([0., 1e-12]), np.zeros(2), .95, .5)
    np.testing.assert_allclose(mean, 0, atol=1e-12)
    np.testing.assert_allclose(variance, .95 ** 2 / 3, atol=1e-10)
    for sxx in (1e-12, 1e-8, 1e-4):
        sxy = 2 * .5 * np.sqrt(sxx)
        actual_mean, actual_var = posterior_moments(np.array([sxx]), np.array([sxy]), .95, .5)
        density = lambda theta: np.exp((sxy * theta - .5 * sxx * theta ** 2) / .5 ** 2)
        mass = quad(density, -.95, .95, epsabs=1e-12)[0]
        mean = quad(lambda theta: theta * density(theta), -.95, .95, epsabs=1e-12)[0] / mass
        variance = quad(lambda theta: (theta - mean) ** 2 * density(theta), -.95, .95, epsabs=1e-12)[0] / mass
        np.testing.assert_allclose([actual_mean[0], actual_var[0]], [mean, variance], atol=1e-12, rtol=1e-10)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixed.npz"
        first = fixed_dataset(path, 12345, 8, [1, 16])
        second = fixed_dataset(path, 12345, 8, [1, 16])
        np.testing.assert_array_equal(first["bayes_mean"], second["bayes_mean"])
        try:
            fixed_dataset(path, 12346, 8, [1, 16])
        except ValueError:
            pass
        else:
            raise AssertionError("Fixed-data metadata mismatch was not rejected")
    return {"quadrature_max_abs_error": error, "prefix_invariance": True,
            "reference_causality": True, "fixed_data_reload_and_spec_validation": True}


def check_delta():
    torch.manual_seed(132)
    q = torch.randn(2, 17, 2, 4, dtype=torch.float64, requires_grad=True)
    k0 = torch.randn_like(q, requires_grad=True)
    k = torch.nn.functional.normalize(k0, dim=-1)
    v = torch.randn(2, 17, 2, 3, dtype=torch.float64, requires_grad=True)
    log_alpha = (-.1 * torch.rand(2, 17, 2, dtype=torch.float64)).requires_grad_()
    beta = torch.rand_like(log_alpha, requires_grad=True)
    initial = torch.randn(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    all_at_once, full_state = delta_chunk(q, k, v, log_alpha, beta, initial)
    state, outputs = initial, []
    for i in range(q.shape[1]):
        state = log_alpha[:, i].exp()[..., None, None] * state
        residual = v[:, i] - torch.einsum("bhk,bhkv->bhv", k[:, i], state)
        state = state + beta[:, i, :, None, None] * torch.einsum("bhk,bhv->bhkv", k[:, i], residual)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state) / 2)
    explicit = torch.stack(outputs, dim=1)
    torch.testing.assert_close(all_at_once, explicit, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(full_state, state, rtol=1e-10, atol=1e-10)
    inputs = (q, k0, v, log_alpha, beta, initial)
    parallel_grads = torch.autograd.grad(all_at_once.square().sum() + full_state.square().sum(), inputs, retain_graph=True)
    explicit_grads = torch.autograd.grad(explicit.square().sum() + state.square().sum(), inputs, retain_graph=True)
    for actual, expected in zip(parallel_grads, explicit_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    # Carry the state through uneven chunks, preserving its gradient.
    state, parts = initial, []
    for start in range(0, 17, 6):
        sl = slice(start, start + 6)
        part, state = delta_chunk(q[:, sl], k[:, sl], v[:, sl], log_alpha[:, sl], beta[:, sl], state)
        parts.append(part)
    chunked = torch.cat(parts, dim=1)
    torch.testing.assert_close(chunked, explicit, rtol=1e-10, atol=1e-10)
    chunk_grads = torch.autograd.grad(chunked.square().sum() + state.square().sum(), inputs)
    for actual, expected in zip(chunk_grads, explicit_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    return {"forward_max_abs_error": float((all_at_once - explicit).abs().max().detach()),
            "gradient_max_abs_error": max(float((a - b).abs().max()) for a, b in zip(parallel_grads, explicit_grads)),
            "nonzero_initial_state_and_uneven_chunks": True}


def check_models(config):
    torch.set_num_threads(2)
    result = {}
    for family in ("gated_deltanet", "transformer_nope"):
        torch.manual_seed(182)
        model = Forecaster(dict(config["model"], family=family))
        streams = sample_streams(18211, 3, 129)
        tokens = torch.tensor(transition_tokens(streams["values"]), dtype=torch.float32)
        with torch.no_grad():
            expected = model(tokens)
            changed = tokens.clone()
            changed[:, 65:] += 20
            error = float((model(changed)[:, :65] - expected[:, :65]).abs().max())
            assert error < 3e-5, error
            row = {"causality_max_abs_error": error, **model.description()}
            streams["lengths"] = np.array([16, 64, 129])
            prefix_predictions = predict(model, streams, "cpu", batch_size=3, streaming_chunk=17)
            np.testing.assert_allclose(prefix_predictions, expected[:, [15, 63, 128]].numpy(), atol=3e-5, rtol=3e-5)
            row["prefix_evaluation_matches_direct_predictions"] = True
            if family == "gated_deltanet":
                for size in (1, 7, 64):
                    state, parts = None, []
                    for start in range(0, tokens.shape[1], size):
                        output, state = model(tokens[:, start:start + size], state, return_state=True)
                        parts.append(output)
                    actual = torch.cat(parts, dim=1)
                    error = float((actual - expected).abs().max())
                    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
                    row[f"streaming_chunk_{size}_max_abs_error"] = error
                actual_elements = sum(t.numel() // len(tokens) for layer in state for t in layer.values())
                assert actual_elements == model.description()["persistent_state_elements_per_series"]
                row["state_size_verified"] = True
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        targets = torch.tensor(sample_streams(18211, 3, 129)["values"][:, 2:], dtype=torch.float32)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tokens) - targets).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            assert torch.isfinite(loss)
        row["three_cpu_optimizer_steps_finite"] = True
        # Check that a reload gives the same prediction; no stale per-series state
        # is stored on the module itself.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(model.state_dict(), path)
            restored = Forecaster(dict(config["model"], family=family))
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(restored(tokens), model(tokens), atol=0, rtol=0)
        row["checkpoint_prediction_exact"] = True
        result[family] = row
    return result


def reference_report(output, config, series):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = fixed_dataset(output / "validation.npz", config["evaluation"]["validation"]["seed"], series,
                         config["evaluation"]["lengths"], config["data"]["bound"], config["data"]["noise_std"])
    rows = score(data["bayes_mean"], data)
    ridge = score(data["ridge_mean"], data)
    # Known-coefficient conditional means remove unpredictable next-step noise.
    # Their squared difference from the posterior mean should agree in expectation
    # with posterior parameter uncertainty times y[t]^2.
    discrepancy = (data["bayes_mean"] - data["latent_mean"]) ** 2 - data["bayes_epistemic_risk"]
    standardized = discrepancy.mean(0) / (discrepancy.std(0, ddof=1) / np.sqrt(series))
    write_json(output / "reference_curves.json", {"bayes": rows, "ridge": ridge,
               "posterior_risk_identity_z_scores": standardized.tolist(),
               "note": "Reference predictors only. No learned-model forecast results exist yet."})
    with (output / "reference_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["length", "bayes_epistemic_risk", "bayes_risk",
                                                   "bayes_conditional_mean_error", "ridge_conditional_mean_error"])
        writer.writeheader()
        for bayes_row, ridge_row in zip(rows, ridge):
            writer.writerow({"length": bayes_row["length"], "bayes_epistemic_risk": bayes_row["bayes_epistemic_risk"],
                             "bayes_risk": bayes_row["bayes_risk"],
                             "bayes_conditional_mean_error": bayes_row["conditional_mean_error"],
                             "ridge_conditional_mean_error": ridge_row["conditional_mean_error"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = data["lengths"]
    epistemic = data["bayes_epistemic_risk"].mean(0)
    se = data["bayes_epistemic_risk"].std(0, ddof=1) / np.sqrt(series)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].loglog(lengths, epistemic, "o-", label="Exact Bayes reference")
    axes[0].fill_between(lengths, np.maximum(epistemic - 1.96 * se, 1e-12), epistemic + 1.96 * se, alpha=.15)
    axes[0].loglog(lengths, [r["conditional_mean_error"] for r in ridge], "s--", label="Online ridge reference")
    axes[0].set_ylabel("Reducible forecasting error")
    axes[0].set_title("Longer history contains useful evidence")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(lengths, config["data"]["noise_std"] ** 2 + epistemic, "o-")
    axes[1].axhline(config["data"]["noise_std"] ** 2, color="gray", linestyle="--", label="Innovation-noise floor")
    axes[1].set_ylabel("Expected next-observation MSE")
    axes[1].set_title("Show the absolute benefit too")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Observed transitions")
        axis.grid(alpha=.2)
        axis.axvline(64, color="gray", alpha=.5, linestyle=":")
    fig.suptitle(f"AR(1) pilot feasibility — statistical references only ({series:,} series)")
    fig.savefig(output / "reference_curves.png", dpi=180)
    fig.savefig(output / "reference_curves.svg")
    plt.close(fig)
    return {"series": series, "max_risk_identity_abs_z": float(np.max(np.abs(standardized))),
            "risk_at_64": rows[2]["bayes_risk"], "risk_at_4096": rows[-1]["bayes_risk"],
            "epistemic_at_64": rows[2]["bayes_epistemic_risk"],
            "epistemic_at_4096": rows[-1]["bayes_epistemic_risk"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--series", type=int, default=1024)
    args = parser.parse_args()
    config = load_config(ROOT / "src/conf/forecast_pilot.yaml")
    result = {"reference": check_reference(), "delta": check_delta(), "models": check_models(config)}
    result["feasibility"] = reference_report(args.output, config, args.series)
    write_json(Path(args.output) / "checks.json", result)
    write_json(Path(args.output) / "resolved_preflight.json", config)
    print(json.dumps(result, indent=2))
