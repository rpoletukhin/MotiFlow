"""Dependency-light MotiFlow training loop."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from motiflow.data.batching import create_data_loader
from motiflow.data.fragment_dataset import FragmentDataset, TrainSampler
from motiflow.modeling import build_model_bundle
from motiflow.training.checkpoints import load_checkpoint, save_checkpoint
from motiflow.training.evaluation import evaluate_checkpoint
from motiflow.training.losses import compute_motif_loss
from motiflow.training.tracking import TrainingTracker, create_tracker, legacy_step_metrics

LOGGER = logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name!r} was requested, but CUDA is unavailable.")
    return requested


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _absolute_data_config(config: DictConfig) -> DictConfig:
    data_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    data_config.processed_data_dir = to_absolute_path(str(data_config.processed_data_dir))
    data_config.library_path = to_absolute_path(str(data_config.library_path))
    return data_config


def _train(config: DictConfig, tracker: TrainingTracker) -> dict[str, int | float | str]:
    """Execute training with an initialized optional experiment tracker."""
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision(str(config.runtime.matmul_precision))
    torch.set_num_threads(int(config.runtime.torch_num_threads))
    device = _device(str(config.runtime.device))
    LOGGER.info("Using device %s", device)

    data_config = _absolute_data_config(config.data)
    bundle = build_model_bundle(
        model_config=config.model,
        flow_config=config.flow,
        data_config=data_config,
        device=device,
    )
    model = bundle.model
    flow_matcher = bundle.flow_matcher
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.learning_rate))

    start_epoch = 0
    step = 0
    resume_path = config.training.get("resume_from")
    if resume_path:
        payload = load_checkpoint(
            to_absolute_path(str(resume_path)),
            model=model,
            optimizer=optimizer,
            fragment_library_path=data_config.library_path,
            map_location=device,
        )
        start_epoch = int(payload["epoch"])
        step = int(payload["step"])
        LOGGER.info("Resumed at epoch=%d step=%d", start_epoch, step)

    dataset = FragmentDataset(split="train", data_conf=data_config, is_training=True)
    sampler = TrainSampler(dataset=dataset, batch_size=int(config.training.batch_size))
    loader = create_data_loader(
        dataset,
        sampler=sampler,
        batch_size=int(config.training.batch_size),
        num_workers=int(config.training.num_loader_workers),
        maximum_squared_size=int(config.training.max_squared_res),
        prefetch_factor=int(config.training.prefetch_factor),
    )

    checkpoint_dir = Path(to_absolute_path(str(config.training.checkpoint_dir)))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_manifest = Path(data_config.processed_data_dir) / "manifest.json"
    log_started = time.monotonic()
    tracking_step_started = time.monotonic()
    tracked_metrics = (
        "total_loss",
        "rot_loss",
        "trans_loss",
        "cat_loss",
        "frag_length",
        "examples_per_step",
    )
    rolling: dict[str, list[float]] = {name: [] for name in tracked_metrics}
    max_steps = config.training.get("max_steps")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("training.max_steps must be positive when set.")
    if int(config.training.num_epochs) <= start_epoch:
        raise ValueError("training.num_epochs must be greater than the resumed epoch.")
    eval_every_steps = config.training.get("eval_every_steps")
    if eval_every_steps is not None and int(eval_every_steps) < 1:
        raise ValueError("training.eval_every_steps must be positive when set.")
    stop = False
    final_epoch = start_epoch

    model.train()
    for epoch in range(start_epoch, int(config.training.num_epochs)):
        final_epoch = epoch
        sampler.set_epoch(epoch)
        for raw_batch in loader:
            if "dummy_batch" in raw_batch:
                continue
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            batch = flow_matcher.sample_times(batch)
            batch = flow_matcher.forward_marginal(batch)
            loss, metrics = compute_motif_loss(
                model=model,
                flow_matcher=flow_matcher,
                batch=batch,
                model_config=config.model,
                flow_config=config.flow,
                loss_config=config.loss,
            )
            if not torch.isfinite(loss):
                tracker.alert(
                    title="Encountered NaN loss",
                    text=f"Loss NaN after {epoch} epochs, {step + 1} steps",
                )
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.gradient_clip_norm))
            optimizer.step()
            step += 1

            for name in rolling:
                rolling[name].append(float(metrics[name].detach().cpu()))
            if step == 1 or step % int(config.training.log_every_steps) == 0:
                elapsed = max(time.monotonic() - log_started, 1e-9)
                window_steps = len(rolling["total_loss"])
                examples = sum(rolling["examples_per_step"])
                averages = {name: float(np.mean(rolling[name])) for name in tracked_metrics[:-1]}
                steps_per_second = window_steps / elapsed
                examples_per_second = examples / elapsed
                values = " ".join(
                    f"{name}={averages[name]:.5f}"
                    for name in ("total_loss", "rot_loss", "trans_loss", "cat_loss", "frag_length")
                )
                LOGGER.info(
                    "step=%d %s steps/s=%.3f examples/s=%.1f",
                    step,
                    values,
                    steps_per_second,
                    examples_per_second,
                )
                for items in rolling.values():
                    items.clear()
                log_started = time.monotonic()

            if step % int(config.training.checkpoint_every_steps) == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pth",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    step=step,
                    fragment_library_path=data_config.library_path,
                    preprocessing_manifest_path=preprocessing_manifest,
                )

            evaluation_metrics = {}
            if eval_every_steps is not None and step % int(eval_every_steps) == 0:
                evaluation_started = time.monotonic()
                evaluation_metrics = evaluate_checkpoint(
                    model=model,
                    flow_matcher=flow_matcher,
                    fragment_library=bundle.fragment_library,
                    data_config=data_config,
                    model_config=config.model,
                    training_config=config.training,
                    device=device,
                    output_dir=Path(to_absolute_path(str(config.training.eval_dir))) / f"step_{step}",
                )
                evaluation_metrics["eval_time"] = time.monotonic() - evaluation_started
                LOGGER.info("step=%d evaluation=%s", step, evaluation_metrics)

            tracking_step_seconds = max(time.monotonic() - tracking_step_started, 1e-9)
            tracking_metrics = legacy_step_metrics(
                loss=loss,
                metrics=metrics,
                batch_t=batch["t"],
                configured_batch_size=int(config.training.batch_size),
                step_seconds=tracking_step_seconds,
                epoch=epoch,
            )
            tracking_metrics.update(evaluation_metrics)
            tracker.log(tracking_metrics, step=step)
            tracking_step_started = time.monotonic()

            if max_steps is not None and step >= int(max_steps):
                stop = True
                break
        if stop:
            break

    final_path = checkpoint_dir / f"step_{step}_final.pth"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=final_epoch,
        step=step,
        fragment_library_path=data_config.library_path,
        preprocessing_manifest_path=preprocessing_manifest,
    )
    return {
        "epoch": final_epoch,
        "step": step,
        "checkpoint": str(final_path),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def train(config: DictConfig) -> dict[str, int | float | str]:
    """Train MotiFlow and always close the configured tracking run."""
    tracker = create_tracker(config)
    try:
        result = _train(config, tracker)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result
