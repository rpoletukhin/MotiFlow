"""Dataset-specific molecular metrics without experiment-specific paths."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Descriptors, RDConfig
from scipy.spatial.distance import cdist

from motiflow.chem.bonding import DRUG_SYMBOLS, QM9_SYMBOLS, openbabel_molecule, sanitize, stability, table_molecule
from motiflow.chem.reconstruction import ReconstructedMolecule
from motiflow.utils.output import suppress_native_output


@dataclass
class MoleculeMetricAccumulator:
    """Accumulate the core paper metrics without retaining every molecule."""

    dataset: str
    retain_molecules: bool = False
    attempted: int = 0
    valid: int = 0
    connected: int = 0
    stable_atoms: int = 0
    stable_molecules: int = 0
    total_atoms: int = 0
    unique_smiles: set[str] = field(default_factory=set)
    valid_smiles: list[str] = field(default_factory=list)
    perceived_molecules: list[Chem.Mol] = field(default_factory=list)
    connected_molecules: list[Chem.Mol] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dataset not in {"qm9", "geom", "qmugs"}:
            raise ValueError(f"Unknown evaluation dataset {self.dataset!r}.")

    def update(self, reconstructed: Iterable[ReconstructedMolecule]) -> None:
        """Add reconstructed molecules using the dataset-specific bond protocol."""
        for item in reconstructed:
            self.attempted += 1
            if not np.isfinite(item.positions).all():
                continue
            with suppress_native_output():
                table, valences = table_molecule(
                    item.positions,
                    item.atomic_numbers,
                    item.known_bonds,
                    conformer=True,
                )
                molecule = table if self.dataset == "qm9" else openbabel_molecule(item.positions, item.atomic_numbers)
                sanitized, canonical, is_valid, is_connected = sanitize(molecule)
            molecule_stable, atom_count, molecule_atoms = stability(item.atomic_numbers, valences)
            self.stable_atoms += atom_count
            self.total_atoms += molecule_atoms
            self.stable_molecules += int(molecule_stable)
            if molecule is not None and self.retain_molecules:
                self.perceived_molecules.append(molecule)
            if not is_valid:
                continue
            self.valid += 1
            self.valid_smiles.append(canonical)
            if is_connected:
                self.connected += 1
                self.unique_smiles.add(canonical)
                if self.retain_molecules:
                    self.connected_molecules.append(sanitized)

    def metrics(self) -> dict[str, float]:
        """Return normalized metrics with the same keys as generated evaluation."""
        if self.attempted == 0:
            raise ValueError("Cannot evaluate an empty reconstructed run.")
        result = {
            "validity": self.valid / self.attempted,
            "connectivity_given_valid": self.connected / self.valid if self.valid else 0.0,
            "valid_and_connected": self.connected / self.attempted,
            "uniqueness": len(self.unique_smiles) / self.connected if self.connected else 0.0,
            "atom_stability": self.stable_atoms / self.total_atoms if self.total_atoms else 0.0,
            "molecule_stability": self.stable_molecules / self.attempted,
        }
        result["validity_times_uniqueness"] = result["validity"] * result["uniqueness"]
        return result


def _sa_scorer():
    contribution = str(Path(RDConfig.RDContribDir) / "SA_Score")
    if contribution not in sys.path:
        sys.path.append(contribution)
    import sascorer

    return sascorer


def normalized_histogram(molecules: Iterable[Chem.Mol], symbols: tuple[str, ...]) -> np.ndarray:
    counts = np.zeros(len(symbols), dtype=np.float64)
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            index = mapping.get(atom.GetSymbol())
            if index is not None:
                counts[index] += 1
    return counts / counts.sum() if counts.sum() else counts


def bond_histogram(molecules: Iterable[Chem.Mol]) -> np.ndarray:
    counts = np.zeros(3, dtype=np.float64)
    mapping = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
    for molecule in molecules:
        for bond in molecule.GetBonds():
            index = mapping.get(bond.GetBondType())
            if index is not None:
                counts[index] += 1
    return counts / counts.sum() if counts.sum() else counts


def bond_lengths(molecules: Iterable[Chem.Mol]) -> dict[tuple[str, str, float], np.ndarray]:
    values = defaultdict(list)
    for molecule in molecules:
        if molecule.GetNumConformers() == 0:
            continue
        distances = Chem.Get3DDistanceMatrix(molecule)
        for bond in molecule.GetBonds():
            first, second = bond.GetBeginAtom(), bond.GetEndAtom()
            symbols = tuple(sorted((first.GetSymbol(), second.GetSymbol())))
            values[(*symbols, bond.GetBondTypeAsDouble())].append(distances[first.GetIdx(), second.GetIdx()])
    return {key: np.asarray(value) for key, value in values.items()}


def multi_kernel_mmd(first, second, max_samples=10_000) -> float:
    first, second = np.asarray(first).reshape(-1), np.asarray(second).reshape(-1)
    if len(first) < 5 or len(second) < 5:
        return 0.0
    rng = np.random.default_rng(42)
    if len(first) > max_samples:
        first = rng.choice(first, max_samples, replace=False)
    if len(second) > max_samples:
        second = rng.choice(second, max_samples, replace=False)
    xx = cdist(first[:, None], first[:, None], "sqeuclidean")
    yy = cdist(second[:, None], second[:, None], "sqeuclidean")
    xy = cdist(first[:, None], second[:, None], "sqeuclidean")
    terms = []
    for gamma in np.logspace(-3, 3, 30):
        terms.append(abs(np.exp(-gamma * xx).mean() + np.exp(-gamma * yy).mean() - 2 * np.exp(-gamma * xy).mean()))
    return float(np.sqrt(np.sum(terms)))


def strain_energy(molecule: Chem.Mol) -> float | None:
    try:
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
        if properties is None:
            return None
        generated = AllChem.MMFFGetMoleculeForceField(molecule, properties)
        relaxed_molecule = Chem.Mol(molecule)
        relaxed = AllChem.MMFFGetMoleculeForceField(relaxed_molecule, properties)
        if generated is None or relaxed is None:
            return None
        initial_energy = generated.CalcEnergy()
        relaxed.Minimize(maxIts=2000)
        return float(initial_energy - relaxed.CalcEnergy())
    except (ValueError, RuntimeError):
        return None


def reference_statistics(molecules: Iterable[Chem.Mol], dataset: str) -> dict:
    molecules = list(molecules)
    symbols = QM9_SYMBOLS if dataset == "qm9" else DRUG_SYMBOLS
    result = {"atom_histogram": normalized_histogram(molecules, symbols)}
    if dataset == "qm9":
        result["bond_histogram"] = bond_histogram(molecules)
        result["bond_lengths"] = bond_lengths(molecules)
    return result


def evaluate_molecules(
    reconstructed: list[ReconstructedMolecule],
    *,
    dataset: str,
    reference: dict | None = None,
    compute_properties: bool = True,
    compute_strain: bool = True,
) -> tuple[dict[str, float], list[Chem.Mol]]:
    """Evaluate one generated run with the protocol used for a paper dataset."""
    accumulator = MoleculeMetricAccumulator(dataset=dataset, retain_molecules=True)
    accumulator.update(reconstructed)
    metrics = accumulator.metrics()
    perceived = accumulator.perceived_molecules
    connected = accumulator.connected_molecules

    if compute_properties:
        property_values = defaultdict(list)
        scorer = _sa_scorer()
        for molecule in connected:
            property_values["qed"].append(QED.qed(molecule))
            property_values["logp"].append(Descriptors.MolLogP(molecule))
            property_values["molecular_weight"].append(Descriptors.MolWt(molecule))
            raw_sa = float(scorer.calculateScore(molecule))
            property_values["sa"].append(np.clip((10.0 - raw_sa) / 9.0, 0.0, 1.0))
        for name in ("qed", "logp", "molecular_weight", "sa"):
            metrics[name] = float(np.mean(property_values[name])) if property_values[name] else 0.0

    if compute_strain:
        values = [value for molecule in connected if (value := strain_energy(molecule)) is not None]
        metrics["strain_energy"] = float(np.mean(values)) if values else 0.0

    if reference is not None:
        symbols = QM9_SYMBOLS if dataset == "qm9" else DRUG_SYMBOLS
        generated_atoms = normalized_histogram(perceived, symbols)
        metrics["tv_atom"] = float(0.5 * np.abs(generated_atoms - reference["atom_histogram"]).sum())
        if dataset == "qm9":
            generated_bonds = bond_histogram(perceived)
            metrics["tv_bond"] = float(0.5 * np.abs(generated_bonds - reference["bond_histogram"]).sum())
            generated_lengths = bond_lengths(connected)
            targets = (
                ("C", "H", 1.0),
                ("C", "C", 1.0),
                ("C", "O", 1.0),
                ("C", "N", 1.0),
                ("H", "N", 1.0),
                ("C", "O", 2.0),
                ("C", "N", 1.5),
                ("H", "O", 1.0),
                ("C", "C", 1.5),
                ("C", "N", 2.0),
            )
            mmd_values = []
            for target in targets:
                key = (*sorted(target[:2]), target[2])
                value = multi_kernel_mmd(reference["bond_lengths"].get(key, ()), generated_lengths.get(key, ()))
                metrics[f"mmd_{target[0]}-{target[1]}:{target[2]:g}"] = value
                mmd_values.append(value)
            metrics["mmd_average"] = float(np.mean(mmd_values))
    return metrics, connected


def composition_match(atomic_numbers: list[np.ndarray], targets: np.ndarray) -> float:
    mapping = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    matches = 0
    for numbers, target in zip(atomic_numbers, targets):
        counts = np.zeros(5, dtype=np.int64)
        for number in numbers:
            if int(number) in mapping:
                counts[mapping[int(number)]] += 1
        matches += int(np.array_equal(counts, np.asarray(target).astype(np.int64)))
    return matches / len(atomic_numbers) if atomic_numbers else 0.0


def fp2_tanimoto(reconstructed: list[ReconstructedMolecule], targets: np.ndarray) -> float:
    try:
        from openbabel import pybel
    except ImportError as exc:
        raise RuntimeError("Structure conditioning evaluation requires openbabel.pybel.") from exc
    values = []
    for item, target in zip(reconstructed, targets):
        lines = [str(len(item.atomic_numbers)), "generated"]
        for number, position in zip(item.atomic_numbers, item.positions):
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(number))
            lines.append(f"{symbol} {position[0]} {position[1]} {position[2]}")
        try:
            with suppress_native_output():
                fingerprint = pybel.readstring("xyz", "\n".join(lines)).calcfp("FP2")
            generated = np.zeros(1024, dtype=np.int8)
            for bit in fingerprint.bits:
                if 0 <= bit < 1024:
                    generated[bit] = 1
            target = np.asarray(target, dtype=np.int8)
            union = np.logical_or(generated, target).sum()
            values.append(np.logical_and(generated, target).sum() / union if union else 0.0)
        except (OSError, RuntimeError, ValueError):
            values.append(0.0)
    return float(np.mean(values)) if values else 0.0
