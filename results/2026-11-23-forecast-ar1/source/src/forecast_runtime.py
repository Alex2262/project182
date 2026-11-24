"""Reproducible AR(1) pilot with explicit dry-run and preflight entry points."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from forecast_data import fixed_dataset, sample_streams, transition_tokens, score
from forecast_models import Forecaster


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path, family=None, regime=None, seed=None):
    config = yaml.safe_load(Path(path).read_text())
    if family is not None:
        config["model"]["family"] = family
    if regime is not None:
        config["training"]["regime"] = regime
    if seed is not None:
        config["training"]["seed"] = seed
    if config["model"]["family"] not in ("gated_deltanet", "transformer_nope"):
        raise ValueError("Unrecognized family")
    if config["training"]["regime"] not in ("short", "mixed"):
        raise ValueError("Unrecognized length regime")
    if config["training"]["seed"] not in (182, 183, 184):
        raise ValueError("Use a registered training seed")
    if config["data"]["family"] != "ar1" or config["data"]["initial_value"] != 0.0:
        raise ValueError("This sampler/reference requires AR(1) with fixed y[0]=0")
    if config["model"]["dtype"] != "float32" or config["training"]["optimizer"] != "Adam":
        raise ValueError("This pilot implements FP32 Adam only")
    if config["training"]["lengths"] != [64, 256]:
        raise ValueError("The preregistered regimes use lengths 64 and 256")
    for length in config["training"]["lengths"]:
        if config["training"]["tokens_per_step"] % length:
            raise ValueError("Training token budget must be divisible by each length")
    sources = {}
    for name in ("forecast_data.py", "forecast_models.py", "forecast_runtime.py"):
        path = Path(__file__).with_name(name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["source_sha256"] = sources
    config["training"]["effective_data_seed"] = config["training"]["data_seed"] + 100000 * (
        config["training"]["seed"] - 182)
    config["training"]["total_training_tokens"] = config["training"]["steps"] * config["training"]["tokens_per_step"]
    torch.manual_seed(config["training"]["seed"])
    config["model_info"] = Forecaster(config["model"]).description()
    cases = []
    for preflight_family in config["preflight"]["families"]:
        torch.manual_seed(config["training"]["seed"])
        description = Forecaster(dict(config["model"], family=preflight_family)).description()
        for length in config["training"]["lengths"]:
            cases.append({"family": preflight_family, "length": length,
                          "batch_size": config["training"]["tokens_per_step"] // length,
                          "optimizer_steps": config["preflight"]["steps_per_case"],
                          "seed": config["training"]["seed"], **description})
    config["preflight"]["resolved_cases"] = cases
    return config


def setup(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def make_optimizer(model, config):
    training = config["training"]
    if training["schedule"] != "constant":
        raise ValueError("Only the constant learning-rate pilot is implemented")
    return torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                            betas=tuple(training["betas"]), eps=training["eps"],
                            weight_decay=training["weight_decay"])


def make_batch(config, step, length, device):
    training, data = config["training"], config["data"]
    streams = sample_streams(training["effective_data_seed"] + step,
                             training["tokens_per_step"] // length, length,
                             data["bound"], data["noise_std"])
    tokens = torch.as_tensor(transition_tokens(streams["values"]), dtype=torch.float32, device=device)
    targets = torch.as_tensor(streams["values"][:, 2:], dtype=torch.float32, device=device)
    return tokens, targets


@torch.no_grad()
def predict(model, data, device, batch_size=8, streaming_chunk=64):
    model.eval()
    lengths = data["lengths"].tolist()
    output = []
    for start in range(0, len(data["values"]), batch_size):
        values = data["values"][start:start + batch_size]
        if model.config["family"] == "gated_deltanet":
            states, predictions = None, {}
            for offset in range(0, max(lengths), streaming_chunk):
                end = min(offset + streaming_chunk, max(lengths))
                # Only the current chunk is transferred to the model device.
                # The offline reference file is not part of its carried state.
                observed = np.stack((values[:, offset:end], values[:, offset + 1:end + 1]), axis=-1)
                chunk = torch.as_tensor(observed, dtype=torch.float32, device=device)
                prediction, states = model(chunk, states, return_state=True)
                for length in lengths:
                    if offset < length <= offset + chunk.shape[1]:
                        predictions[length] = prediction[:, length - offset - 1].cpu().double().numpy()
            output.append(np.stack([predictions[length] for length in lengths], axis=-1))
        else:
            # Full observed prefixes, never a rolling window. No untrained
            # positional embeddings are introduced beyond training length.
            tokens = torch.as_tensor(transition_tokens(values), dtype=torch.float32, device=device)
            output.append(np.stack([model(tokens[:, :length])[:, -1].cpu().double().numpy()
                                    for length in lengths], axis=-1))
    return np.concatenate(output)


def evaluate(model, config, output, device, step, split="validation"):
    if split not in ("validation", "confirmation"):
        raise ValueError(split)
    spec = config["evaluation"][split]
    data = fixed_dataset(Path(output) / "data" / f"{split}.npz", spec["seed"], spec["series"],
                         config["evaluation"]["lengths"], config["data"]["bound"], config["data"]["noise_std"])
    prediction = predict(model, data, device, config["evaluation"]["batch_size"], config["model"]["chunk_size"])
    rows = score(prediction, data)
    write_json(Path(output) / f"{split}_step_{step}.json", rows)
    np.savez_compressed(Path(output) / f"{split}_predictions_step_{step}.npz", prediction=prediction)
    return rows


def preflight(config, output, device="cuda"):
    """Exactly four short checks, then long-prefix inference. No sweep follows."""
    setup(device)
    started = time.monotonic()
    rows = []
    for family in config["preflight"]["families"]:
        for length in config["training"]["lengths"]:
            torch.manual_seed(config["training"]["seed"])
            model_config = dict(config["model"], family=family)
            model = Forecaster(model_config).to(device).float()
            optimizer = make_optimizer(model, config)
            timings, losses = [], []
            for step in range(config["preflight"]["steps_per_case"]):
                tokens, targets = make_batch(config, step, length, device)
                synchronize(device)
                before = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tokens)
                loss = (prediction - targets).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                synchronize(device)
                timings.append(time.monotonic() - before)
                losses.append(float(loss.detach()))
            if not np.isfinite(losses).all():
                raise FloatingPointError("Non-finite preflight loss")
            with torch.no_grad():
                model.eval()
                base = model(tokens[:2])
                changed = tokens[:2].clone()
                changed[:, length // 2:] += 10
                causal_error = float((model(changed)[:, :length // 2] - base[:, :length // 2]).abs().max())
                if causal_error > 3e-5:
                    raise AssertionError(f"Causality error: {causal_error}")
            row = {"family": family, "length": length, "batch_size": tokens.shape[0],
                   "steps": len(losses), "losses": losses, "causality_max_error": causal_error,
                   "median_seconds_per_step_after_warmup": float(np.median(timings[2:])),
                   **model.description()}
            if family == "gated_deltanet":
                with torch.no_grad():
                    state, parts = None, []
                    for offset in range(0, length, 17):
                        part, state = model(tokens[:2, offset:offset + 17], state, return_state=True)
                        parts.append(part)
                    stream_error = float((torch.cat(parts, 1) - base).abs().max())
                    if stream_error > 3e-5:
                        raise AssertionError(f"Streaming-state error: {stream_error}")
                    row["streaming_max_abs_error"] = stream_error
                    del state, parts, part
            # Only the final case per family runs the maximum-length inference check.
            if length == max(config["training"]["lengths"]):
                data = sample_streams(930000001, 2, max(config["evaluation"]["lengths"]),
                                      config["data"]["bound"], config["data"]["noise_std"])
                data["lengths"] = np.array(config["evaluation"]["lengths"])
                synchronize(device)
                before = time.monotonic()
                predicted = predict(model, data, device, batch_size=2)
                synchronize(device)
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("Non-finite long-prefix prediction")
                row["long_prefix_check_seconds"] = time.monotonic() - before
            rows.append(row)
            write_json(Path(output) / "preflight.json", {"cases": rows, "resolved_config": config,
                                                       "elapsed_seconds": time.monotonic() - started})
            del model, optimizer, prediction, loss, base, changed, tokens, targets
            if device == "cuda":
                torch.cuda.empty_cache()
    return {"cases": rows, "elapsed_seconds": time.monotonic() - started}


def train(config, output, device="cuda", max_seconds=1500):
    """Fixed-step, token-matched experiment; validation never changes the schedule."""
    setup(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path, checkpoint_path = output / "config.yaml", output / "state.pt"
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError("Resume configuration or source fingerprint mismatch")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    torch.manual_seed(config["training"]["seed"])
    model = Forecaster(config["model"]).to(device).float()
    optimizer = make_optimizer(model, config)
    completed = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["completed_steps"]
    history_path = output / "training.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    started = time.monotonic()
    run = None
    if config["wandb"]["enabled"]:
        import wandb
        log_dir = Path("/tmp/project182-forecast-wandb") / output.name
        log_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]["project"],
                         name=output.name, id=output.name, resume="allow", config=config,
                         dir=str(log_dir), tags=["forecast-ar1", config["training"]["regime"]],
                         settings=wandb.Settings(disable_git=True, save_code=False))

    def checkpoint():
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "completed_steps": completed, "config": config}, temporary)
        temporary.replace(checkpoint_path)
        write_json(history_path, history)

    try:
        for step in range(completed, config["training"]["steps"]):
            if time.monotonic() - started > max_seconds - 60:
                break
            lengths = config["training"]["lengths"]
            length = lengths[0] if config["training"]["regime"] == "short" else lengths[step % len(lengths)]
            tokens, targets = make_batch(config, step, length, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tokens) - targets).square().mean()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if completed % config["training"]["log_every"] == 0:
                row = {"step": completed, "loss": float(loss.detach()), "length": length,
                       "batch_size": len(tokens), "grad_norm_before_clip": float(grad),
                       "elapsed_seconds": time.monotonic() - started}
                history.append(row)
                if run is not None:
                    run.log({"iteration": completed, **{f"train/{k}": v for k, v in row.items() if k != "step"}})
            if completed % config["training"]["checkpoint_every"] == 0:
                checkpoint()
            if completed in config["training"]["validation_steps"]:
                checkpoint()
                rows = evaluate(model, config, output, device, completed)
                if run is not None:
                    run.log({"iteration": completed, **{
                        f"validation/excess_at_{r['length']}": r["normalized_excess"] for r in rows}})
    finally:
        checkpoint()
        if run is not None:
            run.finish()
    return {"completed_steps": completed, "planned_steps": config["training"]["steps"],
            "elapsed_seconds": time.monotonic() - started, "output": str(output),
            "status": "finished" if completed == config["training"]["steps"] else "partial"}


def default_run_name(config):
    date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%y%m%d")
    return f"alex_{date}_ar1_{config['model']['family']}_{config['training']['regime']}_s{config['training']['seed']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["dry-run", "preflight", "train"])
    parser.add_argument("--config", default=str(Path(__file__).parent / "conf/forecast_pilot.yaml"))
    parser.add_argument("--family", choices=["gated_deltanet", "transformer_nope"])
    parser.add_argument("--regime", choices=["short", "mixed"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    resolved = load_config(args.config, args.family, args.regime, args.seed)
    if args.action == "dry-run":
        print(yaml.safe_dump(resolved, sort_keys=False))
    else:
        directory = args.output or str(Path("outputs/forecast_v1") / default_run_name(resolved))
        result = preflight(resolved, directory, args.device) if args.action == "preflight" else train(resolved, directory, args.device)
        print(json.dumps(result, indent=2))
