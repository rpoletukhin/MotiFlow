"""Configuration-driven evaluation of MotiFlow generation files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from rdkit import Chem

from motiflow.chem.bonding import openbabel_molecule, sanitize, table_molecule
from motiflow.chem.reconstruction import reconstruct_atoms
from motiflow.evaluation.metrics import evaluate_molecules, reference_statistics
from motiflow.modeling import load_fragment_library


def _reference_molecules(path: str, dataset: str):
    source = Path(to_absolute_path(path))
    files = sorted(source.rglob("chunk_*.pt")) if source.is_dir() else [source]
    for file in files:
        for item in torch.load(file, map_location="cpu", weights_only=False):
            positions = torch.as_tensor(item.pos).cpu().numpy()
            numbers = torch.as_tensor(item.h).cpu().numpy().astype(int)
            molecule = (
                table_molecule(positions, numbers)[0] if dataset == "qm9" else openbabel_molecule(positions, numbers)
            )
            sanitized, _, valid, _ = sanitize(molecule)
            if valid:
                yield sanitized


def evaluate(config: DictConfig) -> Path:
    library = load_fragment_library(to_absolute_path(str(config.evaluation.library_path)))
    reference = None
    if config.evaluation.get("reference_path"):
        reference = reference_statistics(
            _reference_molecules(str(config.evaluation.reference_path), str(config.evaluation.dataset)),
            str(config.evaluation.dataset),
        )
    results, connected = [], []
    for configured_path in config.evaluation.generated_paths:
        generated_path = Path(to_absolute_path(str(configured_path)))
        payload = torch.load(generated_path, map_location="cpu", weights_only=False)
        reconstructed = reconstruct_atoms(payload, library)
        metrics, molecules = evaluate_molecules(
            reconstructed,
            dataset=str(config.evaluation.dataset),
            reference=reference,
            compute_properties=bool(config.evaluation.compute_properties),
            compute_strain=bool(config.evaluation.compute_strain),
        )
        metrics["generated_path"] = str(generated_path)
        results.append(metrics)
        connected.extend(molecules)
    if not results:
        raise ValueError("evaluation.generated_paths did not contain any runs.")
    output = Path(to_absolute_path(str(config.evaluation.output_dir)))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "runs.csv", index=False)
    numeric = frame.select_dtypes(include="number")
    summary = {
        column: {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std(ddof=1)) if len(frame) > 1 else 0.0,
        }
        for column in numeric
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if bool(config.evaluation.save_sdf):
        writer = Chem.SDWriter(str(output / "connected.sdf"))
        for molecule in connected:
            writer.write(molecule)
        writer.close()
    return output / "summary.json"
