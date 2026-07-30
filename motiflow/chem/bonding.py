"""Bond perception protocols used by the paper evaluations."""

from __future__ import annotations

import ase
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

from motiflow.utils.output import suppress_native_output

QM9_SYMBOLS = ("H", "C", "N", "O", "F")
DRUG_SYMBOLS = ("H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi")
ATOM_DECODER = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    33: "As",
    35: "Br",
    53: "I",
    80: "Hg",
    83: "Bi",
}
ALLOWED_BONDS = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": (3, 5),
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": (1, 2),
    "Bi": (3, 5),
}

# Picometre tables and margins from the EDM/GeoDiff evaluation lineage.
BONDS1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136, "Cl": 175, "Br": 214, "S": 168, "I": 222, "P": 177},
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {"Si": 233, "H": 148, "C": 185, "O": 163, "S": 200, "F": 160, "Cl": 202, "Br": 215, "I": 243},
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {"Br": 228, "H": 141, "C": 194, "O": 172, "N": 214, "Si": 215, "S": 225, "F": 178, "Cl": 214, "P": 222},
    "P": {"P": 221, "H": 144, "C": 184, "O": 163, "Cl": 203, "S": 210, "F": 156, "N": 177, "Br": 222},
    "I": {"H": 161, "C": 214, "Si": 243, "N": 222, "O": 194, "S": 234, "F": 187, "I": 266},
    "As": {"H": 152},
}
BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 126, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}
BONDS3 = {"C": {"C": 120, "N": 116, "O": 113}, "N": {"C": 116, "N": 110}, "O": {"C": 113}}
BOND_TYPES = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)


def bond_order(first: str, second: str, distance_angstrom: float) -> int:
    if first not in BONDS1 or second not in BONDS1[first]:
        return 0
    distance = 100.0 * distance_angstrom
    if distance >= BONDS1[first][second] + 10:
        return 0
    if first in BONDS2 and second in BONDS2[first] and distance < BONDS2[first][second] + 5:
        if first in BONDS3 and second in BONDS3[first] and distance < BONDS3[first][second] + 3:
            return 3
        return 2
    return 1


def table_molecule(positions: np.ndarray, atomic_numbers: np.ndarray, known_bonds=None, conformer=True):
    atoms = ase.Atoms(numbers=atomic_numbers, positions=positions)
    distances = atoms.get_all_distances()
    molecule = Chem.RWMol()
    for symbol in atoms.symbols:
        molecule.AddAtom(Chem.Atom(symbol))
    valences = np.zeros(len(atoms), dtype=np.int64)
    known_bonds = known_bonds or {}
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            order = int(known_bonds.get((first, second), 0))
            if not order:
                order = bond_order(atoms.symbols[first], atoms.symbols[second], distances[first, second])
            valences[first] += order
            valences[second] += order
            if order:
                molecule.AddBond(first, second, BOND_TYPES[order])
    if conformer:
        geometry = Chem.Conformer(len(atoms))
        for index, position in enumerate(positions):
            geometry.SetAtomPosition(index, Point3D(*map(float, position)))
        molecule.AddConformer(geometry)
    return molecule.GetMol(), valences


def openbabel_molecule(positions: np.ndarray, atomic_numbers: np.ndarray):
    """Perceive drug-like bonds with OpenBabel and return an RDKit molecule."""
    try:
        from openbabel import openbabel
    except ImportError as exc:
        raise RuntimeError("GEOM/QMugs evaluation requires the optional openbabel package.") from exc
    lines = [str(len(positions)), ""]
    for position, number in zip(positions, atomic_numbers):
        symbol = ATOM_DECODER.get(int(number))
        if symbol is None:
            return None
        lines.append(f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}")
    conversion = openbabel.OBConversion()
    conversion.SetInAndOutFormats("xyz", "sdf")
    molecule = openbabel.OBMol()
    with suppress_native_output():
        if not conversion.ReadString(molecule, "\n".join(lines)):
            return None
        molecule.ConnectTheDots()
        molecule.PerceiveBondOrders()
        block = conversion.WriteString(molecule)
    return Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)


def sanitize(molecule):
    if molecule is None:
        return None, None, False, False
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True)
    except (ValueError, RuntimeError):
        return None, None, False, False
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    connected = len(fragments) == 1
    largest = max(fragments, key=lambda item: item.GetNumAtoms(), default=molecule)
    return largest, smiles, bool(smiles), connected


def stability(atomic_numbers: np.ndarray, valences: np.ndarray) -> tuple[bool, int, int]:
    stable = 0
    for number, value in zip(atomic_numbers, valences):
        allowed = ALLOWED_BONDS.get(ATOM_DECODER.get(int(number), ""))
        stable += int(value in allowed if isinstance(allowed, tuple) else value == allowed)
    return stable == len(atomic_numbers), stable, len(atomic_numbers)
