"""Configuration-driven generation from MotiFlow checkpoints."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from motiflow.chem.reconstruction import reconstruct_atoms
from motiflow.data.conditioning import validate_conditioning_type
from motiflow.data.loaders import load_processed_source
from motiflow.evaluation.metrics import composition_match, fp2_tanimoto
from motiflow.modeling import build_model_bundle
from motiflow.provenance import sha256_file
from motiflow.sampling import sample
from motiflow.training.checkpoints import load_checkpoint


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name!r} was requested, but CUDA is unavailable.")
    return device


def _generation_specs(config: DictConfig, data_config: DictConfig, rng: np.random.Generator) -> list[dict]:
    """Build aligned size/condition specifications for an entire generation run."""
    count = int(config.sampling.num_samples)
    condition_type = validate_conditioning_type(str(config.conditioning.type))
    condition_path = config.sampling.get("condition_path")
    if condition_type != "none" and not condition_path:
        raise ValueError("Conditional generation requires sampling.condition_path.")

    if condition_path:
        source = torch.load(to_absolute_path(str(condition_path)), map_location="cpu", weights_only=False)
        if not source:
            raise ValueError("The conditional evaluation source is empty.")
        indices = rng.choice(len(source), size=count, replace=count > len(source))
        specs = []
        for index in indices:
            item = source[int(index)]
            item_count = item.get("num_frags")
            if item_count is None:
                item_count = len(item["frag_ids"])
            spec = {"count": int(item_count)}
            if condition_type == "composition":
                target = torch.as_tensor(item.get("target_counts", item.get("composition_counts"))).float()
                spec.update(condition=target / 29.0, target=target)
            elif condition_type == "structure":
                target = torch.as_tensor(item.get("target_fingerprint", item.get("fingerprint"))).float()
                spec.update(condition=target, target=target)
            elif condition_type != "none":
                raise ValueError(f"Unsupported conditioning type {condition_type!r}.")
            specs.append(spec)
        return specs

    fixed_count = config.sampling.get("fragment_count")
    if fixed_count is not None:
        if int(fixed_count) < 1:
            raise ValueError("sampling.fragment_count must be positive.")
        return [{"count": int(fixed_count)} for _ in range(count)]
    count_path = config.sampling.get("count_source_path")
    if not count_path:
        count_path = str(data_config.processed_data_dir)
    source = load_processed_source(to_absolute_path(str(count_path)), "train")
    lengths = np.asarray([int(item.get("num_frags", len(item["frag_ids"]))) for item in source])
    sampled = rng.choice(lengths, size=count, replace=True)
    return [{"count": int(value)} for value in sampled]


def generate(config: DictConfig) -> Path:
    """Generate rigid-motif states and save them with audit metadata."""
    checkpoint_setting = config.sampling.get("checkpoint_path")
    if not checkpoint_setting:
        raise ValueError("Set sampling.checkpoint_path to a trained checkpoint.")

    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(str(config.runtime.device))

    data_config = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
    data_config.processed_data_dir = to_absolute_path(str(data_config.processed_data_dir))
    data_config.library_path = to_absolute_path(str(data_config.library_path))
    bundle = build_model_bundle(
        model_config=config.model,
        flow_config=config.flow,
        data_config=data_config,
        device=device,
    )

    checkpoint_path = Path(to_absolute_path(str(checkpoint_setting)))
    load_checkpoint(
        checkpoint_path,
        model=bundle.model,
        fragment_library_path=data_config.library_path,
        map_location=device,
    )

    specs = sorted(_generation_specs(config, data_config, rng), key=lambda item: item["count"])
    if not specs:
        raise ValueError("sampling.num_samples must be positive.")
    generated_samples, targets = [], []
    batch_size = int(config.sampling.batch_size)
    if batch_size < 1:
        raise ValueError("sampling.batch_size must be positive.")
    for start in range(0, len(specs), batch_size):
        batch_specs = specs[start : start + batch_size]
        max_count = max(item["count"] for item in batch_specs)
        fragment_mask = torch.zeros(len(batch_specs), max_count, device=device)
        for row, item in enumerate(batch_specs):
            fragment_mask[row, : item["count"]] = 1.0
        initial = bundle.flow_matcher.sample_ref(
            n_samples=len(batch_specs),
            n_fragments=max_count,
            device=device,
            flow_mask=fragment_mask,
            impute_mask=None,
            as_tensor_7=True,
        )
        initial["frag_mask"] = fragment_mask
        if str(config.conditioning.type) != "none":
            initial["condition"] = torch.stack([item["condition"] for item in batch_specs]).to(device)
        result = sample(
            model=bundle.model,
            flow_matcher=bundle.flow_matcher,
            initial_features=initial,
            num_steps=int(config.sampling.num_steps),
            vocab_size=len(bundle.fragment_library),
            use_self_conditioning=bool(config.model.use_self_conditioning),
            keep_trajectory=bool(config.sampling.keep_trajectory),
        )
        for row, item in enumerate(batch_specs):
            molecule_count = item["count"]
            generated_sample = {
                "rigids": result.rigids[row, :molecule_count].detach().cpu(),
                "frag_ids": result.fragment_ids[row, :molecule_count].detach().cpu(),
                "num_frags": molecule_count,
            }
            if result.trajectory:
                generated_sample["trajectory"] = torch.stack(
                    [frame[row, :molecule_count] for frame in result.trajectory]
                )
            generated_samples.append(generated_sample)
            if "target" in item:
                targets.append(item["target"].cpu())

    output_path = Path(to_absolute_path(str(config.sampling.output_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": generated_samples,
        "targets": torch.stack(targets) if targets else None,
        "config": OmegaConf.to_container(config, resolve=True),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fragment_library_sha256": sha256_file(data_config.library_path),
    }
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if targets and bool(config.sampling.get("evaluate_condition", False)):
        reconstructed = reconstruct_atoms(generated_samples, bundle.fragment_library)
        condition_type = str(config.conditioning.type)
        if condition_type == "composition":
            score = composition_match([item.atomic_numbers for item in reconstructed], np.asarray(torch.stack(targets)))
            metrics = {"composition_match_rate": score}
        elif condition_type == "structure":
            metrics = {"fp2_tanimoto": fp2_tanimoto(reconstructed, np.asarray(torch.stack(targets)))}
        else:
            raise ValueError(f"Condition evaluation is unavailable for {condition_type!r} generation.")
        output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return output_path
