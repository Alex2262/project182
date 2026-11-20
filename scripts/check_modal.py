"""Check the personal Modal workspace, A100 access, and persistent storage.

Run after activating the intended personal Modal profile:
    MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/check_modal.py
This launches one brief GPU check, not a training experiment.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("project182-setup-check")
volume = modal.Volume.from_name("project182-artifacts", create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/artifacts": volume},
    timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def check():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    result = {"gpu": gpu, "checked_at": datetime.now(timezone.utc).isoformat()}
    target = Path("/artifacts/setup/gpu-check.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    assert json.loads(target.read_text()) == result
    return {**result, "volume": "project182-artifacts", "file": "setup/gpu-check.json"}


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=2))
