import os
from random import randint
import uuid

from config import parse_config
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, grad_clip=None):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, validate=None, max_seconds=None):
    """Upstream sampler/curriculum loop with device-neutral, resumable checkpoints."""
    import time
    from contextlib import nullcontext
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    device = next(model.parameters()).device
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])
        starting_step = state.get("completed_steps", state["train_step"] + 1)
        for _ in range(starting_step):
            curriculum.update()
    data_sampler = get_data_sampler(args.training.data, n_dims=model.n_dims)
    task_sampler = get_task_sampler(args.training.task, model.n_dims, args.training.batch_size,
                                   num_tasks=args.training.num_tasks, **args.training.task_kwargs)
    completed = starting_step
    start = time.monotonic()
    step_times = []
    def save():
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "completed_steps": completed, "train_step": completed - 1,
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                      "data_seed": getattr(args.training, "data_seed", None)}
        temporary = state_path + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, state_path)
        if completed in getattr(args.training, "checkpoint_steps", []):
            from pathlib import Path
            import shutil
            snapshot=Path(args.out_dir)/"checkpoints"/f"step_{completed}.pt"
            snapshot.parent.mkdir(exist_ok=True)
            if not snapshot.exists():shutil.copy2(state_path,snapshot)
    try:
        for i in range(starting_step, args.training.train_steps):
            if max_seconds is not None and time.monotonic() - start >= max_seconds - 30:
                break
            t = time.monotonic()
            data_args, task_args = {}, {}
            if "sparse" in args.training.task:
                task_args["valid_coords"] = curriculum.n_dims_truncated
            if args.training.num_training_examples is not None:
                seeds = sample_seeds(args.training.num_training_examples,args.training.batch_size)
                data_args["seeds"] = seeds
                task_args["seeds"] = [s + 1 for s in seeds]
            # Stateless per-step data randomness: candidates see identical prompts;
            # resuming a checkpoint reproduces the same next training batch.
            with torch.random.fork_rng(devices=[]) if hasattr(args.training, "data_seed") else nullcontext():
                if hasattr(args.training, "data_seed"):
                    torch.random.default_generator.manual_seed(args.training.data_seed + i)
                xs = data_sampler.sample_xs(curriculum.n_points,args.training.batch_size,
                                             curriculum.n_dims_truncated,**data_args)
                task = task_sampler(**task_args)
                ys = task.evaluate(xs)
            model.train()
            loss, output = train_step(model,xs.to(device),ys.to(device),optimizer,
                                      task.get_training_metric(),getattr(args.training,"grad_clip",None))
            if not __import__('math').isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            curriculum.update(); completed = i + 1
            step_times.append(time.monotonic()-t)
            if completed % args.wandb.log_every_steps == 0 and wandb.run is not None:
                wandb.log({"iteration":completed,"train/mse":loss,
                           "train/seconds_per_step":sum(step_times[-25:])/len(step_times[-25:]),
                           "n_points":curriculum.n_points,"n_dims":curriculum.n_dims_truncated})
            if completed % args.training.save_every_steps == 0:
                save()
            if validate is not None and completed % getattr(args.training, "validation_every_steps", 100) == 0:
                validate(completed)
    finally:
        save()
    return {"completed_steps":completed,"new_steps":completed-starting_step,
            "elapsed_seconds":time.monotonic()-start,
            "seconds_per_step":sum(step_times[5:])/len(step_times[5:]) if len(step_times)>5 else None,
            "stopped_for_time":completed<args.training.train_steps}


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.train()

    train(model, args)

    if not args.test_run:
        from eval import get_run_metrics
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    args = parse_config()
    assert args.model.family in ["gpt2", "lstm", "linear_attention", "mamba2", "gated_deltanet"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
