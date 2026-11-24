"""Personal Modal launcher. Default action is a local dry-run, never a launch."""
import json
import os
from pathlib import Path
import sys

import modal


def require_personal_launch_context():
    """Reject local account ambiguity before defining images or remote functions."""
    if not modal.is_local():
        return
    from modal import config as modal_config
    if os.environ.get("MODAL_PROFILE") != "alexander-tian" or modal_config._profile != "alexander-tian":
        raise RuntimeError("Set MODAL_PROFILE=alexander-tian explicitly; this project uses only the personal workspace")
    overrides = (
        "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN",
        "MODAL_OAUTH_CLIENT_ID", "MODAL_OAUTH_CLIENT_SECRET", "MODAL_OAUTH_JWT_KEY",
        "MODAL_CONFIG_PATH", "MODAL_SERVER_URL",
    )
    present = [name for name in overrides if os.environ.get(name)]
    if present:
        raise RuntimeError("Remove Modal credential/config overrides before using the personal launcher: " + ", ".join(present))


def require_personal_logging(resolved):
    destination = resolved.get("wandb", {})
    if destination.get("entity") != "alexander-tian-ucb" or destination.get("project") != "project182":
        raise ValueError("Forecast runs must log to the personal alexander-tian-ucb/project182 destination")


require_personal_launch_context()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "src/conf/forecast_horizon.yaml"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy==2.2.6", "scipy==1.16.3", "PyYAML==6.0.3", "wandb==0.31.1")
         .env({"PYTHONPATH": "/root/project/src", "OMP_NUM_THREADS": "2"}))
for module in ("forecast_data.py", "forecast_models.py", "forecast_runtime.py", "forecast_horizon.py"):
    image = image.add_local_file(ROOT / "src" / module, remote_path=f"/root/project/src/{module}")
image = image.add_local_file(CONFIG, remote_path="/root/project/src/conf/forecast_horizon.yaml")
volume = modal.Volume.from_name("project182-artifacts", create_if_missing=False, version=2)
app = modal.App("project182-forecast-optimization")


@app.function(image=image, gpu="A100-40GB", cpu=2, memory=8192, timeout=4800,
              max_containers=1, retries=0, scaledown_window=2, volumes={"/artifacts": volume},
              secrets=[modal.Secret.from_name("project182-wandb", required_keys=["WANDB_API_KEY"])])
def train_gpu(resolved, run_name):
    require_personal_logging(resolved)
    import wandb
    from forecast_runtime import prepare_continuation, train
    wandb.teardown()
    volume.reload()
    try:
        if "continuation" in resolved:
            parent = resolved["continuation"]["parent_run_name"]
            if not parent.startswith("alex_") or "/" in parent or "\\" in parent:
                raise ValueError("Invalid continuation parent name")
            prepare_continuation(resolved, f"/artifacts/forecast_v1/runs/{run_name}",
                                 f"/artifacts/forecast_v1/runs/{parent}")
        return train(resolved, f"/artifacts/forecast_v1/runs/{run_name}", device="cuda", max_seconds=4500)
    finally:
        volume.commit()


def resolve_locally(family, regime, seed):
    # Resolution needs scientific dependencies. The lightweight Modal launcher
    # may instead receive an already-resolved JSON produced by the CPU dry-run.
    from forecast_runtime import load_config
    return load_config(CONFIG, family, regime, seed)


def wait_for_call(call):
    """Retry reads of one existing call; never resubmit a GPU invocation."""
    import time
    from grpclib.const import Status
    from grpclib.exceptions import GRPCError, StreamTerminatedError
    while True:
        try:
            return call.get(timeout=20)
        except (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError):
            raise
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except (ConnectionError, modal.exception.ConnectionError, OSError, StreamTerminatedError) as exc:
            print(json.dumps({"event": "result_poll_retry", "error_type": type(exc).__name__}), flush=True)
            time.sleep(5)
        except GRPCError as exc:
            if exc.status not in (Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED):
                raise
            print(json.dumps({"event": "result_poll_retry", "error_type": str(exc.status)}), flush=True)
            time.sleep(5)


@app.local_entrypoint()
def main(action: str = "dry-run", resolved_file: str = "", family: str = "gated_deltanet",
         regime: str = "short", seed: int = 182, run_name: str = ""):
    require_personal_launch_context()
    if action not in ("dry-run", "train"):
        raise ValueError(action)
    resolved = json.loads(Path(resolved_file).read_text()) if resolved_file else resolve_locally(family, regime, seed)
    require_personal_logging(resolved)
    # Detect changed source after the operator reviews a resolved manifest.
    import hashlib
    for name, expected in resolved["source_sha256"].items():
        if hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since configuration resolution: {name}")
    print(json.dumps(resolved, indent=2))
    if action == "dry-run":
        return
    if action == "train":
        if not run_name or "/" in run_name or "\\" in run_name or not run_name.startswith("alex_"):
            raise ValueError("Supply an explicit alex_YYMMDD_<description> run name")
        call = train_gpu.spawn(resolved, run_name)
        print(json.dumps({"event": "submitted", "run_name": run_name,
                          "function_call_id": call.object_id}), flush=True)
        print(json.dumps(wait_for_call(call), indent=2))
