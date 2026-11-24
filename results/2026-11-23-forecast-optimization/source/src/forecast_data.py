"""AR(1) streams and an exact float64 reference for a bounded uniform prior.

One token is the observed transition (y[t-1], y[t]); its target is y[t+1].
The prior is a ~ Uniform[-bound, bound], y[0]=0, and innovations are N(0,sigma²).
Conditioning on this fixed initial value is essential to the likelihood below.
"""
from pathlib import Path
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm


def sample_streams(seed, count, length, bound=0.95, noise_std=0.5):
    if count < 1 or length < 1 or not 0 < bound < 1 or noise_std <= 0:
        raise ValueError("Require positive sizes/noise and 0 < bound < 1")
    parameter_seed, innovation_seed = np.random.SeedSequence(seed).spawn(2)
    theta = np.random.default_rng(parameter_seed).uniform(-bound, bound, count)
    # Time-major draws preserve prefixes when only length changes.
    noise = np.random.default_rng(innovation_seed).normal(
        scale=noise_std, size=(length + 1, count))
    values = np.zeros((count, length + 2), dtype=np.float64)
    for t in range(length + 1):
        values[:, t + 1] = theta * values[:, t] + noise[t]
    return {"values": values, "theta": theta}


def transition_tokens(values):
    return np.stack((values[:, :-2], values[:, 1:-1]), axis=-1)


def posterior_moments(sxx, sxy, bound, noise_std):
    """Truncated-normal posterior, with the exact uniform-prior limit at Sxx=0."""
    sxx, sxy = np.broadcast_arrays(np.asarray(sxx, dtype=np.float64),
                                  np.asarray(sxy, dtype=np.float64))
    mean = np.zeros_like(sxx)
    variance = np.full_like(sxx, bound * bound / 3)
    informed = sxx > 0
    location = sxy[informed] / sxx[informed]
    scale = noise_std / np.sqrt(sxx[informed])
    left, right = (-bound - location) / scale, (bound - location) / scale
    if np.any(informed):
        mean[informed], variance[informed] = truncnorm.stats(
            left, right, loc=location, scale=scale, moments="mv")
    # Near-uniform intervals and far-tail truncations can lose precision in
    # moment subtraction. A bounded likelihood quadrature is safe there.
    fallback = informed & ((~np.isfinite(mean)) | (~np.isfinite(variance)) |
                           (variance < 0) | (variance > bound * bound) |
                           (np.abs(mean) > bound) | (sxx < noise_std ** 2 * 1e-3))
    if np.any(fallback):
        nodes, weights = np.polynomial.legendre.leggauss(128)
        grid = nodes * bound
        log_weight = (sxy[fallback, None] * grid -
                      0.5 * sxx[fallback, None] * grid ** 2) / noise_std ** 2
        log_weight += np.log(weights)
        probability = np.exp(log_weight - logsumexp(log_weight, axis=-1, keepdims=True))
        means = probability @ grid
        mean[fallback] = means
        variance[fallback] = np.sum(probability * (grid - means[:, None]) ** 2, axis=-1)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise FloatingPointError("Non-finite posterior moment")
    return mean, variance


def references(values, lengths, bound=0.95, noise_std=0.5):
    lengths = np.asarray(lengths, dtype=int)
    if len(lengths) == 0 or np.any(lengths < 1) or np.any(lengths > values.shape[1] - 2):
        raise ValueError("Invalid observation prefix")
    if np.any(values[:, 0] != 0):
        raise ValueError("This reference conditions on y[0]=0; do not add burn-in")
    # Only transitions ending at t <= L enter the prediction of y[L+1].
    sxx = np.cumsum(values[:, :-2] ** 2, axis=1)[:, lengths - 1]
    sxy = np.cumsum(values[:, :-2] * values[:, 1:-1], axis=1)[:, lengths - 1]
    mean, variance = posterior_moments(sxx, sxy, bound, noise_std)
    current = values[:, lengths]
    # This ridge rule uses a Gaussian prior with the same variance; unlike the
    # bounded-uniform Bayes rule, it is only a classical baseline.
    ridge_theta = sxy / (sxx + noise_std ** 2 / (bound ** 2 / 3))
    return {"bayes_mean": current * mean,
            "bayes_parameter_mean": mean,
            "bayes_parameter_variance": variance,
            "bayes_epistemic_risk": current ** 2 * variance,
            "bayes_risk": noise_std ** 2 + current ** 2 * variance,
            "ridge_mean": current * ridge_theta,
            "lengths": lengths}


def fixed_dataset(path, seed, count, lengths, bound=0.95, noise_std=0.5):
    path = Path(path)
    spec = {"version": 1, "seed": seed, "count": count,
            "lengths": list(lengths), "bound": bound, "noise_std": noise_std,
            "initial_value": 0.0, "tokenization": "observed_transition"}
    serialized = json.dumps(spec, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["spec"]) != serialized:
                raise ValueError(f"Fixed dataset specification mismatch: {path}")
            return {key: saved[key] for key in saved.files}
    data = sample_streams(seed, count, max(lengths), bound, noise_std)
    data.update(references(data["values"], lengths, bound, noise_std))
    data["latent_mean"] = data["theta"][:, None] * data["values"][:, lengths]
    data["target"] = data["values"][:, np.asarray(lengths) + 1]
    data["spec"] = np.array(serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, **data)
    path.with_suffix(".tmp").replace(path)
    return data


def score(prediction, data):
    """Independent-series SEs. These are not across-training-seed error bars."""
    spec = json.loads(str(data["spec"]))
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != data["bayes_mean"].shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape/finite-value check failed")
    errors = {
        "conditional_mean_error": (prediction - data["latent_mean"]) ** 2,
        "excess_vs_bayes": (prediction - data["bayes_mean"]) ** 2,
        "observed_mse": (prediction - data["target"]) ** 2,
        "expected_predictive_mse": spec["noise_std"] ** 2 +
                                   (prediction - data["latent_mean"]) ** 2,
    }
    rows = []
    for index, length in enumerate(data["lengths"]):
        row = {"length": int(length), "series": prediction.shape[0],
               "bayes_risk": float(data["bayes_risk"][:, index].mean()),
               "bayes_epistemic_risk": float(data["bayes_epistemic_risk"][:, index].mean())}
        for name, error in errors.items():
            row[name] = float(error[:, index].mean())
            row[name + "_series_se"] = float(error[:, index].std(ddof=1) / np.sqrt(len(error)))
        row["normalized_excess"] = row["excess_vs_bayes"] / row["bayes_risk"]
        rows.append(row)
    return rows
