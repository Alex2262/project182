"""Verify the personal W&B connection through Modal without exposing credentials.

MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/check_wandb.py
Add --log-test after confirming the destination to write one setup-only run.
"""

import json
import modal

app = modal.App("project182-wandb-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb==0.31.1")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("project182-wandb", required_keys=["WANDB_API_KEY"])],
    timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def check(log_test: bool = False, entity: str = "alexander-tian-ucb", verify_run: str = ""):
    import os
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("The Modal secret is missing WANDB_API_KEY")
    api = wandb.Api(timeout=30)
    viewer = api.viewer
    if viewer.username != "alexander-tian":
        raise RuntimeError("The secret does not belong to the expected personal W&B account")
    result = {"username": viewer.username, "default_entity": api.default_entity,
              "teams": viewer.teams, "client_version": wandb.__version__}
    if log_test:
        launch_date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%y%m%d")
        with wandb.init(
            entity=entity,
            project="project182",
            name=f"alex_{launch_date}_logging_setup",
            job_type="setup-check",
            tags=["setup-only", "no-training"],
            config={"purpose": "Verify Modal to W&B logging; no training"},
            settings=wandb.Settings(init_timeout=60, disable_git=True, save_code=False),
        ) as run:
            run.log({"setup/connection_ok": 1})
            result["run_url"] = run.url
            result["run_path"] = run.path
    if verify_run:
        if not verify_run.startswith(f"{entity}/project182/"):
            raise ValueError("Only this personal project's runs can be verified")
        result["run_path"] = verify_run
    if "run_path" in result:
        import time
        for attempt in range(3):
            api.flush()
            saved = api.run(result["run_path"])
            result["saved_metric"] = saved.summary_metrics.get("setup/connection_ok")
            result["run_state"] = saved.state
            if result["saved_metric"] == 1:
                break
            time.sleep(2)
        else:
            raise RuntimeError("The setup metric was not readable after waiting for sync")
    return result


@app.local_entrypoint()
def main(log_test: bool = False, entity: str = "alexander-tian-ucb", verify_run: str = ""):
    print(json.dumps(check.remote(log_test, entity, verify_run), indent=2))
