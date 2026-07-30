"""Deterministic dataset split construction and verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

QMUGS_PAPER_TARGET_CONFORMERS = {
    "train": 300_000,
    "val": 30_000,
    "test": 30_000,
}
QMUGS_PAPER_SEED = 42
QMUGS_PAPER_MW_QUANTILES = 20
QMUGS_PAPER_CHUNK_SIZE = 1_000
QMUGS_PAPER_MANIFEST_PATH = Path(__file__).with_name("manifests") / "qmugs_paper_split.json"
GEOM_PAPER_MANIFEST_PATH = Path(__file__).with_name("manifests") / "geom_paper_split.json"


@dataclass(frozen=True)
class QmugsSplitResult:
    """A molecule-disjoint QMugs split and the statistics used to audit it."""

    splits: dict[str, list[str]]
    source_conformer_count: int
    source_molecule_count: int
    average_conformers_per_molecule: float
    selected_conformer_counts: dict[str, int]
    unused_molecule_count: int
    molecular_weight_bin_counts: dict[str, int]
    seed: int
    molecular_weight_quantiles: int
    target_conformer_counts: dict[str, int]

    @property
    def selected_molecule_counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self.splits.items()}


def hash_id_sequence(ids: Iterable[str]) -> str:
    """Hash IDs in order using the exact newline-delimited on-disk encoding."""
    digest = hashlib.sha256()
    for molecule_id in ids:
        digest.update(str(molecule_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_targets(target_conformer_counts: Mapping[str, int]) -> dict[str, int]:
    required = {"train", "val", "test"}
    if set(target_conformer_counts) != required:
        raise ValueError(f"QMugs targets must contain exactly {sorted(required)}.")
    targets = {name: int(target_conformer_counts[name]) for name in ("train", "val", "test")}
    if any(value <= 0 for value in targets.values()):
        raise ValueError("QMugs conformer targets must be positive.")
    return targets


def create_qmugs_splits(
    summary_csv: str | Path,
    *,
    target_conformer_counts: Mapping[str, int] = QMUGS_PAPER_TARGET_CONFORMERS,
    molecular_weight_quantiles: int = QMUGS_PAPER_MW_QUANTILES,
    seed: int = QMUGS_PAPER_SEED,
) -> QmugsSplitResult:
    """Reproduce the paper's molecule-level, molecular-weight-stratified split.

    The requested sizes are expressed in conformers. The original paper pipeline
    converted them to molecule counts using the dataset-wide mean number of
    conformers, then split unique molecules. Consequently, the realized
    conformer counts are close to, but not exactly equal to, the targets.
    """
    targets = _validate_targets(target_conformer_counts)
    if molecular_weight_quantiles < 2:
        raise ValueError("At least two molecular-weight quantiles are required.")

    summary = pd.read_csv(summary_csv, usecols=["chembl_id", "conf_id", "mw"])
    molecule_stats = (
        summary.groupby("chembl_id")
        .agg({"conf_id": "count", "mw": "first"})
        .reset_index()
        .rename(columns={"conf_id": "n_confs"})
    )
    if molecule_stats.empty:
        raise ValueError("The QMugs summary contains no molecules.")
    if molecule_stats["mw"].isna().any():
        raise ValueError("Molecular weights must be present for every QMugs molecule.")

    average_conformers = float(molecule_stats["n_confs"].mean())
    molecule_targets = {name: int(value / average_conformers) for name, value in targets.items()}
    molecule_stats["strat_bin"] = pd.qcut(
        molecule_stats["mw"],
        q=molecular_weight_quantiles,
        labels=False,
    )

    train_molecules, remainder = train_test_split(
        molecule_stats,
        train_size=molecule_targets["train"],
        stratify=molecule_stats["strat_bin"],
        random_state=seed,
    )
    validation_ratio = molecule_targets["val"] / len(remainder)
    validation_molecules, remainder = train_test_split(
        remainder,
        train_size=validation_ratio,
        stratify=remainder["strat_bin"],
        random_state=seed,
    )
    test_ratio = molecule_targets["test"] / len(remainder)
    test_molecules, unused = train_test_split(
        remainder,
        train_size=test_ratio,
        stratify=remainder["strat_bin"],
        random_state=seed,
    )

    frames = {
        "train": train_molecules,
        "val": validation_molecules,
        "test": test_molecules,
    }
    splits = {name: frame["chembl_id"].astype(str).tolist() for name, frame in frames.items()}
    selected_conformer_counts = {name: int(frame["n_confs"].sum()) for name, frame in frames.items()}
    bin_counts = {
        str(int(bin_index)): int(count)
        for bin_index, count in molecule_stats["strat_bin"].value_counts().sort_index().items()
    }
    return QmugsSplitResult(
        splits=splits,
        source_conformer_count=len(summary),
        source_molecule_count=len(molecule_stats),
        average_conformers_per_molecule=average_conformers,
        selected_conformer_counts=selected_conformer_counts,
        unused_molecule_count=len(unused),
        molecular_weight_bin_counts=bin_counts,
        seed=seed,
        molecular_weight_quantiles=molecular_weight_quantiles,
        target_conformer_counts=targets,
    )


def write_qmugs_split_chunks(
    result: QmugsSplitResult,
    output_dir: str | Path,
    *,
    chunk_size: int = QMUGS_PAPER_CHUNK_SIZE,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Write ordered molecule IDs as ``<split>_ids/chunk_NNN.txt`` files."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    root = Path(output_dir)
    written: dict[str, list[Path]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = root / f"{split_name}_ids"
        existing = sorted(split_dir.glob("chunk_*.txt")) if split_dir.is_dir() else []
        if existing and not overwrite:
            raise FileExistsError(f"{split_dir} already contains split chunks; pass overwrite=True to replace them.")
        if overwrite:
            for path in existing:
                path.unlink()
        split_dir.mkdir(parents=True, exist_ok=True)

        split_paths = []
        ids = result.splits[split_name]
        for start in range(0, len(ids), chunk_size):
            chunk_index = start // chunk_size
            chunk_path = split_dir / f"chunk_{chunk_index:03d}.txt"
            payload = "".join(f"{molecule_id}\n" for molecule_id in ids[start : start + chunk_size])
            chunk_path.write_text(payload, encoding="utf-8")
            split_paths.append(chunk_path)
        written[split_name] = split_paths
    return written


def build_qmugs_split_manifest(result: QmugsSplitResult, *, chunk_size: int) -> dict:
    """Build compact, content-addressed metadata for a QMugs split."""
    return {
        "schema_version": "1.0",
        "dataset": "qmugs",
        "split_unit": "chembl_id",
        "stratification": {
            "column": "mw",
            "method": "pandas.qcut",
            "quantiles": result.molecular_weight_quantiles,
        },
        "random_seed": result.seed,
        "target_conformer_counts": result.target_conformer_counts,
        "source": {
            "conformer_count": result.source_conformer_count,
            "molecule_count": result.source_molecule_count,
            "average_conformers_per_molecule": result.average_conformers_per_molecule,
        },
        "unused_molecule_count": result.unused_molecule_count,
        "molecular_weight_bin_counts": result.molecular_weight_bin_counts,
        "chunk_size": chunk_size,
        "splits": {
            name: {
                "molecule_count": len(ids),
                "conformer_count": result.selected_conformer_counts[name],
                "chunk_count": (len(ids) + chunk_size - 1) // chunk_size,
                "ordered_ids_sha256": hash_id_sequence(ids),
            }
            for name, ids in result.splits.items()
        },
    }


def load_qmugs_paper_split_manifest() -> dict:
    """Load the frozen manifest for the split used in the paper."""
    with open(QMUGS_PAPER_MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def compare_qmugs_split_manifests(actual: Mapping, expected: Mapping) -> list[str]:
    """Return human-readable differences in reproducibility-critical fields."""
    differences = []
    scalar_paths: Sequence[tuple[str, ...]] = (
        ("random_seed",),
        ("target_conformer_counts",),
        ("source", "conformer_count"),
        ("source", "molecule_count"),
        ("unused_molecule_count",),
        ("molecular_weight_bin_counts",),
        ("chunk_size",),
    )
    for path in scalar_paths:
        actual_value = actual
        expected_value = expected
        for component in path:
            actual_value = actual_value.get(component)
            expected_value = expected_value.get(component)
        if actual_value != expected_value:
            dotted = ".".join(path)
            differences.append(f"{dotted}: expected {expected_value!r}, got {actual_value!r}")

    for split_name in ("train", "val", "test"):
        for field in ("molecule_count", "conformer_count", "chunk_count", "ordered_ids_sha256"):
            actual_value = actual["splits"][split_name][field]
            expected_value = expected["splits"][split_name][field]
            if actual_value != expected_value:
                differences.append(f"splits.{split_name}.{field}: expected {expected_value!r}, got {actual_value!r}")
    return differences


def load_geom_paper_split_manifest() -> dict:
    """Load the content-addressed GEOM split manifest used by the paper."""
    with open(GEOM_PAPER_MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def verify_geom_split_files(data_root: str | Path, manifest: Mapping | None = None) -> list[str]:
    """Return differences between local GEOM artifacts and the paper manifest."""
    from motiflow.provenance import sha256_file

    expected = dict(manifest or load_geom_paper_split_manifest())
    differences = []
    root = Path(data_root)
    for split_name, split in expected["splits"].items():
        source = root / split["file"]
        if not source.is_file():
            differences.append(f"{split_name}: missing {source}")
            continue
        actual_hash = sha256_file(source)
        if actual_hash != split["sha256"]:
            differences.append(f"{split_name}.sha256: expected {split['sha256']!r}, got {actual_hash!r}")
    return differences
