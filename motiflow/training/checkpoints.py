"""Atomic, provenance-aware training checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from motiflow.provenance import fingerprint, sha256_file

CHECKPOINT_SCHEMA_VERSION = "2.0"
TIME_CONVENTION = "prior_t0_data_t1"


def _resolved_config(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
    step: int,
    fragment_library_path: str | Path,
    preprocessing_manifest_path: str | Path | None = None,
) -> None:
    """Atomically save model state plus the inputs needed to audit it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(config)
    manifest_hash = None
    if preprocessing_manifest_path is not None:
        manifest = Path(preprocessing_manifest_path)
        if manifest.is_file():
            manifest_hash = sha256_file(manifest)

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "time_convention": TIME_CONVENTION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "config": resolved,
        "config_sha256": fingerprint(resolved),
        "fragment_library_sha256": sha256_file(fragment_library_path),
        "preprocessing_manifest_sha256": manifest_hash,
    }
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    fragment_library_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint, rejecting a mismatched fragment vocabulary."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema {payload.get('schema_version')!r}.")
    if payload.get("time_convention") != TIME_CONVENTION:
        raise ValueError(f"Unsupported checkpoint time convention {payload.get('time_convention')!r}.")
    if fragment_library_path is not None:
        actual_hash = sha256_file(fragment_library_path)
        expected_hash = payload.get("fragment_library_sha256")
        if actual_hash != expected_hash:
            raise ValueError("Checkpoint fragment-library hash does not match the requested library.")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
