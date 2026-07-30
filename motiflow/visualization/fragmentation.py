"""Post-processing visualization of saved rigid-motif decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from motiflow.data.loaders import load_aligned_data, load_processed_data


@dataclass(frozen=True)
class FragmentationExample:
    """One aligned molecule together with its saved decomposition and library."""

    dataset: str
    split: str
    sample_id: str
    molecule: Any
    processed: Mapping[str, Any]
    library: Mapping[int, Mapping[str, Any]]
    processed_index: int
    source_index: int


@lru_cache(maxsize=12)
def _load_aligned_cached(aligned_root: str, dataset: str, split: str):
    return load_aligned_data(aligned_root, dataset, split)


@lru_cache(maxsize=12)
def _load_processed_cached(fragmented_dir: str, split: str):
    return load_processed_data(fragmented_dir, split)


@lru_cache(maxsize=12)
def _load_library_cached(fragmented_dir: str):
    return torch.load(
        Path(fragmented_dir) / "fragment_library.pt",
        map_location="cpu",
        weights_only=False,
    )


def load_fragmentation_example(
    *,
    aligned_root: str | Path,
    fragmented_dir: str | Path,
    dataset: str,
    split: str,
    processed_index: int,
) -> FragmentationExample:
    """Load one saved decomposition by its index in a processed split."""
    if dataset not in {"qm9", "geom", "qmugs"}:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from 'qm9', 'geom', and 'qmugs'.")
    aligned_root = str(Path(aligned_root).resolve())
    fragmented_dir = str(Path(fragmented_dir).resolve())
    processed = _load_processed_cached(fragmented_dir, split)
    if not 0 <= processed_index < len(processed):
        raise IndexError(f"Processed index {processed_index} is outside [0, {len(processed)}).")
    library = _load_library_cached(fragmented_dir)
    aligned = _load_aligned_cached(aligned_root, dataset, split)
    record = processed[processed_index]
    source_index = int(record.get("orig_mol_index", processed_index))
    if not 0 <= source_index < len(aligned):
        raise IndexError(f"Source index {source_index} is outside the aligned {split} split.")
    molecule = aligned[source_index]
    sample_id = str(record.get("sample_id", source_index))
    aligned_sample_id = str(getattr(molecule, "sample_id", source_index))
    if sample_id != aligned_sample_id:
        raise ValueError(
            f"Processed sample {sample_id!r} maps to aligned sample {aligned_sample_id!r}; "
            "the selected artifacts do not share the same source ordering."
        )
    return FragmentationExample(
        dataset=dataset,
        split=split,
        sample_id=sample_id,
        molecule=molecule,
        processed=record,
        library=library,
        processed_index=processed_index,
        source_index=source_index,
    )


def fragmentation_summary(example: FragmentationExample) -> dict[str, Any]:
    """Return concise metadata for the selected molecule and decomposition."""
    fragment_sizes = [len(indices) for indices in example.processed["original_indices"]]
    return {
        "dataset": example.dataset,
        "split": example.split,
        "sample_id": example.sample_id,
        "processed_index": example.processed_index,
        "aligned_index": example.source_index,
        "atoms": len(example.molecule.h),
        "fragments": int(example.processed["num_frags"]),
        "unique_fragment_classes": len({int(class_id) for class_id in example.processed["frag_ids"]}),
        "largest_fragment_atoms": max(fragment_sizes, default=0),
        "smiles": str(example.molecule.smiles),
    }


def fragmentation_table(example: FragmentationExample) -> pd.DataFrame:
    """Return a readable fragment-to-vocabulary mapping for one molecule."""
    rows = []
    for fragment_index, (class_id, atom_indices) in enumerate(
        zip(example.processed["frag_ids"], example.processed["original_indices"])
    ):
        class_id = int(class_id)
        entry = example.library[class_id]
        numbers = torch.as_tensor(entry["exemplar_z"])
        rows.append(
            {
                "fragment": fragment_index,
                "class_id": class_id,
                "smiles": entry["smiles"],
                "source_atom_indices": list(map(int, atom_indices)),
                "real_atoms": int((numbers != 0).sum()),
                "dummy_atoms": int((numbers == 0).sum()),
                "symmetry_order": len(entry["symmetries"]),
            }
        )
    return pd.DataFrame(rows)


def _bond_segments(first: np.ndarray, second: np.ndarray, bond_type) -> list[tuple[np.ndarray, np.ndarray]]:
    direction = second - first
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return []
    direction /= length
    reference = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perpendicular = np.cross(direction, reference)
    perpendicular /= np.linalg.norm(perpendicular)
    offset = 0.15 * perpendicular
    if bond_type == Chem.BondType.DOUBLE:
        return [(first + offset, second + offset), (first - offset, second - offset)]
    if bond_type == Chem.BondType.TRIPLE:
        return [(first, second), (first + offset, second + offset), (first - offset, second - offset)]
    return [(first, second)]


def _append_segments(coordinates: tuple[list, list, list], segments) -> None:
    x_values, y_values, z_values = coordinates
    for first, second in segments:
        x_values.extend((first[0], second[0], None))
        y_values.extend((first[1], second[1], None))
        z_values.extend((first[2], second[2], None))


def fragmentation_figure(example: FragmentationExample, *, show_atom_labels: bool | None = None):
    """Build an interactive 3D figure colored by saved fragment assignment."""
    try:
        import plotly.colors as colors
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Fragment visualization requires the 'visualization' optional dependencies.") from exc

    positions = torch.as_tensor(example.molecule.pos).cpu().numpy()
    atomic_numbers = torch.as_tensor(example.molecule.h).cpu().numpy()
    symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(number)) for number in atomic_numbers]
    molecule = Chem.MolFromSmiles(str(example.molecule.smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse the aligned SMILES for sample {example.sample_id!r}.")
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() != len(positions):
        raise ValueError("Aligned coordinates and stored SMILES have different atom counts.")
    if show_atom_labels is None:
        show_atom_labels = len(positions) <= 30

    atom_to_fragment = {}
    for fragment_index, atom_indices in enumerate(example.processed["original_indices"]):
        for atom_index in atom_indices:
            atom_to_fragment[int(atom_index)] = fragment_index

    fragment_count = int(example.processed["num_frags"])
    palette = colors.qualitative.Dark24
    internal = {index: ([], [], []) for index in range(fragment_count)}
    cut = ([], [], [])
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        segments = _bond_segments(positions[first], positions[second], bond.GetBondType())
        first_fragment = atom_to_fragment.get(first)
        second_fragment = atom_to_fragment.get(second)
        target = internal[first_fragment] if first_fragment is not None and first_fragment == second_fragment else cut
        _append_segments(target, segments)

    figure = go.Figure()
    if cut[0]:
        figure.add_trace(
            go.Scatter3d(
                x=cut[0],
                y=cut[1],
                z=cut[2],
                mode="lines",
                line={"color": "rgb(120, 120, 120)", "width": 4, "dash": "dot"},
                hoverinfo="none",
                name="Inter-fragment bonds",
            )
        )
    for fragment_index in range(fragment_count):
        color = palette[fragment_index % len(palette)]
        coordinates = internal[fragment_index]
        if coordinates[0]:
            figure.add_trace(
                go.Scatter3d(
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    mode="lines",
                    line={"color": color, "width": 5},
                    hoverinfo="none",
                    showlegend=False,
                    legendgroup=f"fragment_{fragment_index}",
                )
            )
        atom_indices = list(map(int, example.processed["original_indices"][fragment_index]))
        class_id = int(example.processed["frag_ids"][fragment_index])
        fragment_positions = positions[atom_indices]
        fragment_symbols = [symbols[index] for index in atom_indices]
        fragment_smiles = str(example.library[class_id]["smiles"])
        compact_smiles = fragment_smiles if len(fragment_smiles) <= 36 else f"{fragment_smiles[:33]}..."
        hover_text = [
            f"Atom {atom_index}: {symbol}<br>Fragment {fragment_index}<br>Class {class_id}: {fragment_smiles}"
            for atom_index, symbol in zip(atom_indices, fragment_symbols)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=fragment_positions[:, 0],
                y=fragment_positions[:, 1],
                z=fragment_positions[:, 2],
                mode="markers+text" if show_atom_labels else "markers",
                marker={
                    "size": [4 if symbol == "H" else 9 for symbol in fragment_symbols],
                    "color": color,
                    "line": {"color": "white", "width": 1},
                },
                text=fragment_symbols if show_atom_labels else None,
                textposition="middle right",
                name=f"F{fragment_index} · class {class_id} · {compact_smiles}",
                legendgroup=f"fragment_{fragment_index}",
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    figure.update_layout(
        title=(
            f"{example.dataset.upper()} {example.split} · sample {example.sample_id} · {fragment_count} rigid motifs"
        ),
        scene={"xaxis_visible": False, "yaxis_visible": False, "zaxis_visible": False, "aspectmode": "data"},
        legend={
            "x": 1.02,
            "y": 1,
            "bgcolor": "rgba(255, 255, 255, 0.85)",
            "font": {"size": 10},
            "itemsizing": "constant",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        height=720,
    )
    return figure
