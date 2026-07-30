"""Conditioning targets derived from aligned molecular structures."""

from __future__ import annotations

import torch

from motiflow.utils.output import suppress_native_output

QM9_ATOMIC_NUMBERS = (1, 6, 7, 8, 9)
QM9_ATOM_TO_INDEX = {number: index for index, number in enumerate(QM9_ATOMIC_NUMBERS)}
QM9_ATOM_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
SUPPORTED_CONDITIONING_TYPES = frozenset({"none", "composition", "structure"})


def validate_conditioning_type(condition_type: str) -> str:
    """Validate the conditioning modes supported by the paper implementation."""
    if condition_type not in SUPPORTED_CONDITIONING_TYPES:
        choices = ", ".join(sorted(SUPPORTED_CONDITIONING_TYPES))
        raise ValueError(f"Unsupported conditioning type {condition_type!r}; choose from {choices}.")
    return condition_type


def qm9_conditioning_targets(data) -> dict[str, torch.Tensor]:
    """Compute the composition and OpenBabel FP2 targets used in the paper."""
    atomic_numbers = torch.as_tensor(data.h).detach().cpu().long().reshape(-1)
    positions = torch.as_tensor(data.pos).detach().cpu().float()
    if positions.shape != (len(atomic_numbers), 3):
        raise ValueError(
            f"QM9 conditioning requires aligned positions with shape [num_atoms, 3]; received {tuple(positions.shape)}."
        )
    unknown = sorted(set(atomic_numbers.tolist()) - set(QM9_ATOMIC_NUMBERS))
    if unknown:
        raise ValueError(f"QM9 conditioning encountered unsupported atomic numbers {unknown}.")

    counts = torch.zeros(len(QM9_ATOMIC_NUMBERS), dtype=torch.float32)
    for number in atomic_numbers.tolist():
        counts[QM9_ATOM_TO_INDEX[number]] += 1.0
    proportions = counts / len(atomic_numbers)

    try:
        from openbabel import pybel
    except ImportError as exc:
        raise RuntimeError("QM9 substructure conditioning requires openbabel.pybel.") from exc

    lines = [str(len(atomic_numbers)), "Conditioning"]
    for number, position in zip(atomic_numbers.tolist(), positions.tolist()):
        symbol = QM9_ATOM_TO_SYMBOL[number]
        lines.append(f"{symbol} {position[0]:.5f} {position[1]:.5f} {position[2]:.5f}")
    with suppress_native_output():
        molecule = pybel.readstring("xyz", "\n".join(lines))
        fingerprint = molecule.calcfp("FP2")

    fingerprint_vector = torch.zeros(1024, dtype=torch.float32)
    for bit in fingerprint.bits:
        # Preserve the historical implementation's direct OpenBabel bit indexing.
        if 0 <= bit < 1024:
            fingerprint_vector[bit] = 1.0

    return {
        "composition_counts": counts,
        "composition_proportions": proportions,
        "fingerprint": fingerprint_vector,
    }
