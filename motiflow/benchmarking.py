"""Wall-clock generation benchmark used in Appendix A.5.2."""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from motiflow.chem.reconstruction import reconstruct_atoms
from motiflow.data.loaders import load_processed_source
from motiflow.modeling import build_model_bundle
from motiflow.sampling import sample
from motiflow.training.checkpoints import load_checkpoint


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def benchmark_walltime(config: DictConfig) -> Path:
    """Time prior-to-data integration plus motif-to-atom reconstruction."""
    checkpoint_value = config.benchmark.get("checkpoint_path")
    if checkpoint_value is None or str(checkpoint_value).lower() in {"", "none", "null"}:
        raise ValueError("benchmark.checkpoint_path must point to a trained checkpoint.")
    num_samples = int(config.benchmark.num_samples)
    batch_size = int(config.benchmark.batch_size)
    num_steps = int(config.benchmark.num_steps)
    if num_samples <= 0 or batch_size <= 0 or num_steps <= 0:
        raise ValueError("benchmark num_samples, batch_size, and num_steps must all be positive.")
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(str(config.runtime.device))
    data = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
    data.processed_data_dir = to_absolute_path(str(data.processed_data_dir))
    data.library_path = to_absolute_path(str(data.library_path))
    bundle = build_model_bundle(model_config=config.model, flow_config=config.flow, data_config=data, device=device)
    checkpoint = Path(to_absolute_path(str(checkpoint_value)))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    load_checkpoint(
        checkpoint,
        model=bundle.model,
        fragment_library_path=data.library_path,
        map_location=device,
    )

    count_source = Path(to_absolute_path(str(config.benchmark.count_source_path)))
    records = load_processed_source(count_source, "train")
    population = np.asarray([int(item.get("num_frags", len(item["frag_ids"]))) for item in records])
    rng = np.random.default_rng(seed)
    counts = rng.choice(population, size=num_samples, replace=True)
    all_samples = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(counts), batch_size):
        current = counts[start : start + batch_size]
        max_count = int(current.max())
        mask = torch.zeros(len(current), max_count, device=device)
        for row, length in enumerate(current):
            mask[row, : int(length)] = 1.0
        initial = bundle.flow_matcher.sample_ref(
            n_samples=len(current),
            n_fragments=max_count,
            device=device,
            flow_mask=mask,
            impute_mask=None,
            as_tensor_7=True,
        )
        initial["frag_mask"] = mask
        result = sample(
            model=bundle.model,
            flow_matcher=bundle.flow_matcher,
            initial_features=initial,
            num_steps=num_steps,
            vocab_size=len(bundle.fragment_library),
            use_self_conditioning=bool(config.model.use_self_conditioning),
        )
        for row, length in enumerate(current):
            length = int(length)
            all_samples.append(
                {
                    "frag_ids": result.fragment_ids[row, :length].cpu(),
                    "rigids": result.rigids[row, :length].cpu(),
                }
            )
    reconstruct_atoms(all_samples, bundle.fragment_library)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    output = Path(to_absolute_path(str(config.benchmark.output_path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seconds": elapsed,
        "seconds_per_sample": elapsed / len(counts),
        "num_samples": len(counts),
        "batch_size": batch_size,
        "num_steps": num_steps,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "host": platform.node(),
        "includes_reconstruction": True,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output
