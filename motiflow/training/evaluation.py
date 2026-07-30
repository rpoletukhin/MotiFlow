"""Periodic generated-sample evaluation used during MotiFlow training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from motiflow.chem.reconstruction import reconstruct_atoms
from motiflow.data.batching import create_data_loader
from motiflow.data.fragment_dataset import FragmentDataset
from motiflow.evaluation.metrics import (
    MoleculeMetricAccumulator,
    composition_match,
    fp2_tanimoto,
)
from motiflow.sampling import sample
from motiflow.utils import rigid_utils as ru


def _legacy_molecule_metrics(accumulator: MoleculeMetricAccumulator) -> dict[str, float]:
    """Return the exact generated-molecule keys used by historical W&B runs."""
    if accumulator.attempted == 0:
        raise ValueError("Cannot evaluate an empty generated sample set.")
    valid_unique = len(set(accumulator.valid_smiles)) / accumulator.attempted
    return {
        "atom_stable": accumulator.stable_atoms / accumulator.total_atoms if accumulator.total_atoms else 0.0,
        "molecule_stable": accumulator.stable_molecules / accumulator.attempted,
        "valid": accumulator.valid / accumulator.attempted,
        "valid_connected": accumulator.connected / accumulator.attempted,
        "valid_unique": valid_unique,
        # Historical runs used an empty reference set, so every unique molecule
        # was also counted as novel.
        "valid_unique_novel": valid_unique,
    }


def _evaluation_loader(data_config, training_config):
    dataset = FragmentDataset(split="val", data_conf=data_config, is_training=False)
    return create_data_loader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=int(training_config.eval_batch_size),
        num_workers=int(training_config.num_loader_workers),
        maximum_squared_size=int(training_config.max_squared_res),
        prefetch_factor=int(training_config.prefetch_factor),
    )


def _legacy_condition_metrics(
    condition_type: str,
    reconstructed,
    targets: np.ndarray | None,
) -> dict[str, float]:
    """Return the conditional metric name used by the matching historical run."""
    if targets is None:
        return {}
    if condition_type == "composition":
        return {
            "composition_match_rate": composition_match(
                [item.atomic_numbers for item in reconstructed],
                targets,
            )
        }
    if condition_type == "structure":
        return {"tanimoto_similarity": fp2_tanimoto(reconstructed, targets)}
    return {}


def evaluate_checkpoint(
    *,
    model: torch.nn.Module,
    flow_matcher,
    fragment_library: dict,
    data_config,
    model_config,
    training_config,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    """Generate from validation sizes/conditions and compute legacy W&B metrics."""
    generated_samples = []
    targets = []
    loader = _evaluation_loader(data_config, training_config)
    maximum_batches = training_config.get("max_eval_batches")

    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= int(maximum_batches):
            break
        if "dummy_batch" in raw_batch:
            continue
        batch = {name: value.to(device) if torch.is_tensor(value) else value for name, value in raw_batch.items()}
        fragment_mask = batch["frag_mask"]
        batch_size, fragment_count = fragment_mask.shape
        initial = flow_matcher.sample_ref(
            n_samples=batch_size,
            n_fragments=fragment_count,
            device=device,
            flow_mask=fragment_mask,
            impute_mask=None,
            impute=ru.Rigid.from_tensor_7(batch["rigids_1"]),
            impute_cat=batch["frag_ids"],
            as_tensor_7=True,
        )
        initial["frag_mask"] = fragment_mask
        if "condition" in batch:
            initial["condition"] = batch["condition"]
        result = sample(
            model=model,
            flow_matcher=flow_matcher,
            initial_features=initial,
            num_steps=int(data_config.num_t),
            vocab_size=len(fragment_library),
            use_self_conditioning=bool(model_config.use_self_conditioning),
        )
        fragment_ids = result.fragment_ids if result.fragment_ids is not None else batch["frag_ids"]
        for row in range(batch_size):
            count = int(fragment_mask[row].sum().item())
            generated_samples.append(
                {
                    "frag_ids": fragment_ids[row, :count].detach().cpu(),
                    "rigids": result.rigids[row, :count].detach().cpu(),
                    "num_frags": count,
                }
            )
            targets.append(batch["eval_target"][row].detach().cpu())

    reconstructed = reconstruct_atoms(generated_samples, fragment_library)
    accumulator = MoleculeMetricAccumulator(
        dataset=str(data_config.dataset_name),
        retain_molecules=True,
    )
    accumulator.update(reconstructed)
    metrics = _legacy_molecule_metrics(accumulator)
    condition_type = str(data_config.conditioning_type)
    target_array = np.asarray(torch.stack(targets)) if targets else None
    metrics.update(_legacy_condition_metrics(condition_type, reconstructed, target_array))

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generated_samples, output_dir / "generated_rigids.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics
