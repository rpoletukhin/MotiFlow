"""Fragment and dataset statistics used in Appendix A.5.5--A.5.6."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from motiflow.data.loaders import load_processed_data


def _fragment_ids(payload: Any) -> Iterable[np.ndarray]:
    if isinstance(payload, Mapping) and "samples" in payload:
        yield from _fragment_ids(payload["samples"])
        return
    if isinstance(payload, Mapping) and "rigids" in payload:
        ids = payload.get("fragment_ids", payload.get("frag_ids"))
        if ids is None:
            raise ValueError("Generated payload has rigids but no fragment IDs.")
        ids = torch.as_tensor(ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        masks = payload.get("fragment_mask")
        for index in range(ids.shape[0]):
            count = ids.shape[1] if masks is None else int(torch.as_tensor(masks[index]).sum())
            yield ids[index, :count].cpu().numpy().astype(np.int64)
        return
    if not isinstance(payload, (list, tuple)):
        raise TypeError("Samples must be a clean payload dictionary or a list of records.")
    for item in payload:
        ids = item.get("frag_ids", item.get("fragment_ids"))
        if ids is None:
            raise ValueError("Sample record has no fragment IDs.")
        count = int(item.get("num_frags", len(ids)))
        yield torch.as_tensor(ids)[:count].cpu().numpy().astype(np.int64)


def _distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": int(array.max()),
    }


def _count_ids(samples: Iterable[np.ndarray], vocab_size: int) -> tuple[np.ndarray, list[int]]:
    counts = np.zeros(vocab_size, dtype=np.int64)
    lengths = []
    for ids in samples:
        if len(ids) == 0:
            raise ValueError("Encountered a molecule with zero fragments.")
        if np.any(ids < 0) or np.any(ids >= vocab_size):
            raise ValueError("Encountered a fragment ID outside the configured vocabulary.")
        counts += np.bincount(ids, minlength=vocab_size)
        lengths.append(len(ids))
    if not lengths:
        raise ValueError("No molecule records were supplied for fragment analysis.")
    return counts, lengths


def _aligned_atom_counts(path: Path) -> tuple[list[int], list[int], int]:
    files = sorted(path.rglob("shard_*.pt")) if path.is_dir() else [path]
    if path.is_dir() and not files:
        files = sorted(path.rglob("chunk_*.pt"))
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(f"Aligned data source does not exist or has no chunks: {path}")
    total_atoms: list[int] = []
    heavy_atoms: list[int] = []
    for file in files:
        for item in torch.load(file, map_location="cpu", weights_only=False):
            numbers = item["h"] if isinstance(item, Mapping) else item.h
            numbers = torch.as_tensor(numbers)
            total_atoms.append(len(numbers))
            heavy_atoms.append(int((numbers > 1).sum()))
    if not total_atoms:
        raise ValueError(f"Aligned data source contains no molecules: {path}")
    return total_atoms, heavy_atoms, len(total_atoms)


def analyze_fragment_statistics(config: DictConfig) -> Path:
    """Write Table 11/12 statistics and Figure 10--12 source data."""
    library_path = Path(to_absolute_path(str(config.data.library_path)))
    library = torch.load(library_path, map_location="cpu", weights_only=False)
    vocab_size = len(library)
    configured_vocab_size = config.data.get("vocab_size")
    if configured_vocab_size is not None and vocab_size != int(configured_vocab_size):
        raise ValueError(f"Configured vocabulary has {configured_vocab_size} classes, library has {vocab_size}.")

    class_sizes = np.asarray([len(torch.as_tensor(library[index]["exemplar_z"])) for index in range(vocab_size)])
    real_sizes = np.asarray(
        [int((torch.as_tensor(library[index]["exemplar_z"]) != 0).sum()) for index in range(vocab_size)]
    )
    heavy_sizes = np.asarray(
        [int((torch.as_tensor(library[index]["exemplar_z"]) > 1).sum()) for index in range(vocab_size)]
    )
    symmetry_sizes = np.asarray(
        [
            len(torch.as_tensor(library[index].get("symmetries", torch.eye(3).unsqueeze(0))))
            for index in range(vocab_size)
        ]
    )

    processed_dir = Path(to_absolute_path(str(config.data.processed_data_dir)))
    training_ids = []
    for split in config.analysis.splits:
        training_ids.extend(_fragment_ids(load_processed_data(processed_dir, str(split))))
    train_counts, fragment_lengths = _count_ids(training_ids, vocab_size)
    fragmented_total_atoms = [int(real_sizes[ids].sum()) for ids in training_ids]
    fragmented_heavy_atoms = [int(heavy_sizes[ids].sum()) for ids in training_ids]
    aligned_value = config.analysis.get("aligned_path")
    if aligned_value is None or str(aligned_value).lower() in {"", "none", "null"}:
        total_atoms = fragmented_total_atoms
        heavy_atoms = fragmented_heavy_atoms
        atom_statistics_source = str(processed_dir)
        atom_statistics_molecules = len(training_ids)
    else:
        aligned_path = Path(to_absolute_path(str(aligned_value)))
        total_atoms, heavy_atoms, atom_statistics_molecules = _aligned_atom_counts(aligned_path)
        atom_statistics_source = str(aligned_path)

    rows: dict[int, dict[str, Any]] = {
        index: {
            "class_id": index,
            "size_with_dummies": int(class_sizes[index]),
            "real_atoms": int(real_sizes[index]),
            "heavy_atoms": int(heavy_sizes[index]),
            "symmetry_order": int(symmetry_sizes[index]),
            "training_count": int(train_counts[index]),
            "training_frequency": float(train_counts[index] / train_counts.sum()),
        }
        for index in range(vocab_size)
    }

    generated_runs = []
    for run_index, raw_path in enumerate(config.analysis.generated_paths):
        generated_path = Path(to_absolute_path(str(raw_path)))
        payload = torch.load(generated_path, map_location="cpu", weights_only=False)
        generated_counts, generated_lengths = _count_ids(_fragment_ids(payload), vocab_size)
        train_frequency = train_counts / train_counts.sum()
        generated_frequency = generated_counts / generated_counts.sum()
        ratios = np.divide(
            generated_frequency,
            train_frequency,
            out=np.full(vocab_size, np.nan),
            where=train_frequency > 0,
        )
        label = f"generated_{run_index + 1}"
        for class_id in range(vocab_size):
            rows[class_id][f"{label}_count"] = int(generated_counts[class_id])
            rows[class_id][f"{label}_frequency"] = float(generated_frequency[class_id])
            rows[class_id][f"{label}_to_training_ratio"] = float(ratios[class_id])
        generated_runs.append(
            {
                "path": str(generated_path),
                "molecules": len(generated_lengths),
                "fragments_per_molecule": _distribution(generated_lengths),
            }
        )

    size_class_histogram = dict(sorted(Counter(class_sizes.tolist()).items()))
    size_usage_histogram = {int(size): int(train_counts[class_sizes == size].sum()) for size in np.unique(class_sizes)}
    summary = {
        "dataset": str(config.data.dataset_name),
        "method_version": str(config.data.method_version),
        "splits": list(config.analysis.splits),
        "fragmented_molecules": len(training_ids),
        "atom_statistics_molecules": atom_statistics_molecules,
        "atom_statistics_source": atom_statistics_source,
        "vocabulary_size": vocab_size,
        "maximum_fragment_size_with_dummies": int(class_sizes.max()),
        "maximum_symmetry_order": int(symmetry_sizes.max()),
        "fragments_per_molecule": _distribution(fragment_lengths),
        "total_atoms_per_molecule": _distribution(total_atoms),
        "heavy_atoms_per_molecule": _distribution(heavy_atoms),
        "fragmented_total_atoms_per_molecule": _distribution(fragmented_total_atoms),
        "fragmented_heavy_atoms_per_molecule": _distribution(fragmented_heavy_atoms),
        "vocabulary_fragment_size_histogram": size_class_histogram,
        "data_weighted_fragment_size_histogram": size_usage_histogram,
        "generated_runs": generated_runs,
    }

    output_dir = Path(to_absolute_path(str(config.analysis.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows.values()).to_csv(output_dir / "per_fragment.csv", index=False)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path
