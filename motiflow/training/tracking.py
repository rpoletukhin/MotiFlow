"""Optional experiment tracking with the historical MotiFlow metric contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger(__name__)


class TrainingTracker:
    """Small lifecycle wrapper around an optional tracking run."""

    def __init__(self, run: Any | None = None):
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if self._run is not None:
            self._run.log(dict(metrics), step=int(step))

    def alert(self, *, title: str, text: str) -> None:
        """Send an alert when tracking is enabled."""
        if self._run is not None:
            self._run.alert(title=title, text=text)

    def finish(self, *, exit_code: int = 0) -> None:
        if self._run is not None:
            self._run.finish(exit_code=int(exit_code))
            self._run = None


def _flatten_config(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration keys with the legacy colon separator."""
    flattened = {}
    for name, value in values.items():
        key = f"{prefix}:{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            flattened.update(_flatten_config(value, key))
        else:
            flattened[key] = value
    return flattened


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar tracking value, received shape {array.shape}.")
    return float(array.reshape(-1)[0])


def t_stratified_loss(
    batch_t: Any,
    batch_loss: Any,
    *,
    num_bins: int = 5,
    loss_name: str = "loss",
) -> dict[str, float]:
    """Reproduce the five-bin time-stratified loss used by historical runs."""
    flat_t = _numpy(batch_t).reshape(-1)
    flat_losses = _numpy(batch_loss).reshape(-1)
    if flat_t.shape != flat_losses.shape:
        raise ValueError(
            "Time and per-example loss arrays must have the same flattened shape; "
            f"received {flat_t.shape} and {flat_losses.shape}."
        )
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_indices = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    if np.any(bin_indices < 0) or np.any(bin_indices >= num_bins):
        raise ValueError("Training times must lie in the historical interval [0, 1.001).")
    binned_loss = np.bincount(bin_indices, weights=flat_losses, minlength=num_bins)
    binned_count = np.bincount(bin_indices, minlength=num_bins)
    result = {}
    for bin_index in np.flatnonzero(binned_count):
        start = bin_edges[bin_index]
        end = bin_edges[bin_index + 1]
        key = f"{loss_name} t=[{start:.2f},{end:.2f})"
        result[key] = float(binned_loss[bin_index] / binned_count[bin_index])
    return result


def legacy_step_metrics(
    *,
    loss: Any,
    metrics: Mapping[str, Any],
    batch_t: Any,
    configured_batch_size: int,
    step_seconds: float,
    epoch: int,
) -> dict[str, float | int]:
    """Build the exact standard-trainer payload used by the dirty repository."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive.")
    payload: dict[str, float | int] = {
        "loss": _scalar(loss),
        "rotation_loss": _scalar(metrics["rot_loss"]),
        "translation_loss": _scalar(metrics["trans_loss"]),
        "cat_loss": _scalar(metrics["cat_loss"]),
        "batch_size": _scalar(metrics["examples_per_step"]),
        "frag_length": _scalar(metrics["frag_length"]),
        "examples_per_sec": float(configured_batch_size) / float(step_seconds),
        "num_epochs": int(epoch),
    }
    for metric_name, batch_name in (
        ("rot_loss", "batch_rot_loss"),
        ("trans_loss", "batch_trans_loss"),
        ("cat_loss", "batch_cat_loss"),
    ):
        payload.update(
            t_stratified_loss(
                batch_t,
                metrics[batch_name],
                loss_name=metric_name,
            )
        )
    return payload


def create_tracker(config: DictConfig) -> TrainingTracker:
    """Initialize the configured tracker, or return a no-op tracker."""
    tracking = config.get("tracking")
    if tracking is None or not bool(tracking.get("enabled", False)):
        return TrainingTracker()

    backend = str(tracking.get("backend", "wandb"))
    if backend != "wandb":
        raise ValueError(f"Unsupported tracking backend {backend!r}.")
    mode = str(tracking.get("mode", "online"))
    if mode not in {"online", "offline"}:
        raise ValueError("tracking.mode must be either 'online' or 'offline'.")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB tracking requires the 'tracking' optional dependency. "
            "Install MotiFlow with `pip install -e '.[tracking]'`."
        ) from exc

    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("The resolved training configuration must be a mapping.")
    run_directory = Path(to_absolute_path(str(tracking.dir)))
    run_directory.mkdir(parents=True, exist_ok=True)
    run_id = tracking.get("id")
    resume = tracking.get("resume") if run_id else None
    init_arguments = {
        "project": str(tracking.project),
        "entity": tracking.get("entity"),
        "name": tracking.get("name"),
        "group": tracking.get("group"),
        "tags": list(tracking.get("tags", [])),
        "job_type": str(tracking.get("job_type", "train")),
        "dir": str(run_directory),
        "mode": mode,
        "id": run_id,
        "resume": resume,
        "config": _flatten_config(resolved),
    }
    run = wandb.init(**{key: value for key, value in init_arguments.items() if value is not None})
    if run is None:
        raise RuntimeError("wandb.init returned no run.")
    LOGGER.info("WandB run initialized: id=%s mode=%s dir=%s", run.id, mode, run.dir)
    return TrainingTracker(run)
