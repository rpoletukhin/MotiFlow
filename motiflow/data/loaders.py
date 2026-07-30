import glob
import hashlib
import json
import os
import pickle
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

QM9_REFERENCE_SPLIT_SHA256 = "04fc491236b080b80819a88c18b4909134a5bbb6f41cad6791b0703351b0f3c2"
QM9_REFERENCE_SPLIT_URL = (
    "https://raw.githubusercontent.com/frcnt/equivariant-neural-diffusion/main/data/qm9/splits.json"
)


class ShardedDataSequence(Sequence):
    """Replayable, indexable dataset that retains at most one tensor shard."""

    def __init__(self, shard_paths: Sequence[str]):
        self.shard_paths = tuple(shard_paths)
        self._lengths = tuple(self._shard_length(path) for path in self.shard_paths)
        cumulative = []
        total = 0
        for length in self._lengths:
            total += length
            cumulative.append(total)
        self._cumulative = tuple(cumulative)
        self._cache_index = None
        self._cache_records = None

    @staticmethod
    def _shard_length(path: str) -> int:
        manifest_path = path.removesuffix(".pt") + ".manifest.json"
        if os.path.isfile(manifest_path):
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            record_count = manifest.get("record_count")
            if record_count is not None:
                return int(record_count)
            succeeded_ids = manifest.get("succeeded_ids")
            if succeeded_ids is not None:
                return len(succeeded_ids)
        return len(torch.load(path, map_location="cpu"))

    def _load_shard(self, shard_index: int):
        if self._cache_index != shard_index:
            self._cache_records = torch.load(self.shard_paths[shard_index], map_location="cpu")
            self._cache_index = shard_index
        return self._cache_records

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __iter__(self) -> Iterator[Any]:
        for shard_index in range(len(self.shard_paths)):
            yield from self._load_shard(shard_index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative, index)
        shard_start = 0 if shard_index == 0 else self._cumulative[shard_index - 1]
        return self._load_shard(shard_index)[index - shard_start]


def load_processed_data(processed_root: str | os.PathLike[str], split_name: str) -> Sequence[Any]:
    """Load a canonical processed split, preferring bounded-memory shards."""
    root = os.fspath(processed_root)
    split_dir = os.path.join(root, split_name)
    shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
    if shard_paths:
        return ShardedDataSequence(shard_paths)

    monolithic = os.path.join(root, f"{split_name}_processed.pt")
    if os.path.isfile(monolithic):
        return torch.load(monolithic, map_location="cpu", weights_only=False)
    raise FileNotFoundError(
        f"No processed split {split_name!r}: expected shards under {split_dir!r} or the file {monolithic!r}."
    )


def load_processed_source(source: str | os.PathLike[str], split_name: str = "train") -> Sequence[Any]:
    """Load a processed split from either an explicit file or an artifact root."""
    path = os.fspath(source)
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    direct_shards = sorted(glob.glob(os.path.join(path, "shard_*.pt")))
    if direct_shards:
        return ShardedDataSequence(direct_shards)
    return load_processed_data(path, split_name)


def processed_split_exists(processed_root: str | os.PathLike[str], split_name: str) -> bool:
    """Return whether either supported processed-split layout exists."""
    root = os.fspath(processed_root)
    return bool(glob.glob(os.path.join(root, split_name, "shard_*.pt"))) or os.path.isfile(
        os.path.join(root, f"{split_name}_processed.pt")
    )


def get_qm9_splits(cfg):
    """Load and verify the published equivariant-neural-diffusion QM9 split."""
    splits_path = os.path.join(cfg["data_root"], cfg["dataset"]["splits_path"])
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"Missing the published QM9 split file at {splits_path!r}. "
            f"Download it without modification from {QM9_REFERENCE_SPLIT_URL}. "
            "MotiFlow does not create or modify the QM9 split."
        )

    with open(splits_path, encoding="utf-8") as handle:
        splits = json.load(handle)

    canonical_payload = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    split_hash = hashlib.sha256(canonical_payload).hexdigest()
    if split_hash != QM9_REFERENCE_SPLIT_SHA256:
        raise ValueError(
            f"QM9 split file {splits_path!r} does not match the published "
            "equivariant-neural-diffusion split. "
            f"Expected canonical SHA-256 {QM9_REFERENCE_SPLIT_SHA256}, got {split_hash}. "
            f"Download the reference file from {QM9_REFERENCE_SPLIT_URL}."
        )
    return splits


def load_qm9_tasks(cfg, indices):
    tar_path = os.path.join(cfg["data_root"], cfg["dataset"]["tar_path"])
    return [{"dataset": "qm9", "sample_id": str(idx), "tar_index": idx, "tar_path": tar_path} for idx in indices]


def load_geom_tasks(cfg, split_name):
    """Load every record from a configured, provenance-fixed GEOM split."""
    relative_path = cfg["dataset"]["split_files"][split_name]
    pkl_path = os.path.join(cfg["data_root"], relative_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Missing the GeoDiff-preprocessed GEOM-Drugs split at {pkl_path!r}. "
            "Download the published GeoDiff artifacts and preserve their filenames; "
            "see the data setup section in README.md."
        )
    with open(pkl_path, "rb") as handle:
        raw_data = pickle.load(handle)

    tasks = []
    for index, datum in enumerate(raw_data):
        task = {"dataset": "geom", "sample_id": str(index)}
        if not hasattr(datum, "rdmol") or datum.rdmol is None:
            task["load_error"] = "GEOM record has no RDKit molecule"
        else:
            task.update(
                mol_bytes=datum.rdmol.ToBinary(),
                pos=datum.pos,
                smiles=datum.smiles,
                aux={
                    "totalenergy": getattr(datum, "totalenergy", None),
                    "boltzmannweight": getattr(datum, "boltzmannweight", None),
                },
            )
        tasks.append(task)
    return tasks


def load_qmugs_tasks(cfg, target_ids):
    """Load selected QMugs conformers without materializing the full summary."""
    dataset = cfg["dataset"]
    base = dataset["root"]
    summary_csv = os.path.join(base, dataset["summary_csv"])
    structures_dir = os.path.join(base, dataset["structures_dir"])
    ordered_targets = None if target_ids is None else [str(value) for value in target_ids]
    if ordered_targets is not None and len(set(ordered_targets)) != len(ordered_targets):
        raise ValueError("QMugs split contains duplicate molecule IDs.")
    target_set = None if ordered_targets is None else set(ordered_targets)
    columns = ["chembl_id", "conf_id", "smiles", "DFT_TOTAL_ENERGY", "atoms"]

    selected = []
    for chunk in pd.read_csv(
        summary_csv,
        usecols=columns,
        chunksize=int(dataset.get("summary_chunk_size", 100000)),
        dtype={"chembl_id": str, "conf_id": str},
    ):
        if target_set is not None:
            chunk = chunk[chunk["chembl_id"].isin(target_set)]
        if not chunk.empty:
            selected.append(chunk)

    if not selected:
        return []
    frame = pd.concat(selected, ignore_index=True)
    if target_set is not None:
        missing = sorted(target_set - set(frame["chembl_id"]))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"QMugs split contains {len(missing)} IDs absent from summary.csv: {preview}")

    if ordered_targets is not None:
        molecule_order = {molecule_id: index for index, molecule_id in enumerate(ordered_targets)}
        frame["_molecule_order"] = frame["chembl_id"].map(molecule_order)
        frame.sort_values("_molecule_order", kind="stable", inplace=True)

    thermal_energy = 0.00094
    minimum_energy = frame.groupby("chembl_id")["DFT_TOTAL_ENERGY"].transform("min")
    unnormalized = np.exp(-(frame["DFT_TOTAL_ENERGY"] - minimum_energy) / thermal_energy)
    frame["boltzmannweight"] = unnormalized / unnormalized.groupby(frame["chembl_id"]).transform("sum")

    tasks = []
    for row in frame.itertuples(index=False):
        tasks.append(
            {
                "dataset": "qmugs",
                "sample_id": f"{row.chembl_id}_{row.conf_id}",
                "sdf_path": os.path.join(structures_dir, row.chembl_id, f"{row.conf_id}.sdf"),
                "smiles": row.smiles,
                "aux": {
                    "chembl_id": row.chembl_id,
                    "conf_id": row.conf_id,
                    "totalenergy": row.DFT_TOTAL_ENERGY,
                    "boltzmannweight": row.boltzmannweight,
                    "source_atom_count": row.atoms,
                },
            }
        )
    return tasks


def load_aligned_data(aligned_root: str, dataset_name: str, split_name: str) -> Sequence[Any]:
    """
    Uniformly loads aligned PyG Data tensors.
    Auto-detects if the data is a single file (QM9/GEOM) or a directory of chunks (QMugs).
    """
    ds_dir = os.path.join(aligned_root, dataset_name)

    # Prefer the canonical deterministic shard layout. A nested shard group is
    # used when a large source-ID chunk is aligned independently.
    split_dir = os.path.join(ds_dir, split_name)
    if os.path.isdir(split_dir):
        direct_shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
        nested_shards = sorted(glob.glob(os.path.join(split_dir, "*", "shard_*.pt")))
        if direct_shards and nested_shards:
            raise RuntimeError(
                f"Mixed direct and grouped alignment shards under {split_dir}. Keep only one canonical layout."
            )
        chunk_files = direct_shards or nested_shards
        if not chunk_files:
            # Accept the earlier sequential chunk layout without mixing layouts.
            chunk_files = sorted(glob.glob(os.path.join(split_dir, "chunk_*.pt")))

        if chunk_files:
            print(f"Using {len(chunk_files)} lazy shards from {split_dir}...")
            return ShardedDataSequence(chunk_files)

    # Check for a sequential single-file split.
    # Expected format: <aligned_root>/<dataset_name>/<split_name>_aligned.pt
    file_path = os.path.join(ds_dir, f"{split_name}_aligned.pt")
    if os.path.isfile(file_path):
        print(f"Loading {file_path}...")
        return torch.load(file_path, map_location="cpu")

    print(f"Warning: Neither single file '{file_path}' nor chunk directory '{split_dir}' found.")
    return []
