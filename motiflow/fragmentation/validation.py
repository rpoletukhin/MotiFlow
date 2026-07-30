"""Post-fragmentation reconstruction validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from motiflow.chem.reconstruction import reconstruct_atoms
from motiflow.evaluation.metrics import MoleculeMetricAccumulator
from motiflow.utils.output import suppress_native_output
from motiflow.utils.rigid_helpers import assemble_rigid_mat


def _reconstruction_payload(records: Sequence[Mapping]) -> list[dict]:
    payload = []
    for record in records:
        rotations = torch.as_tensor(record["rots"], dtype=torch.float32)
        translations = torch.as_tensor(record["trans"], dtype=torch.float32)
        payload.append(
            {
                "frag_ids": torch.as_tensor(record["frag_ids"], dtype=torch.long),
                "rigids": assemble_rigid_mat(rotations, translations).to_tensor_7(),
                "num_frags": int(record["num_frags"]),
            }
        )
    return payload


def evaluate_ground_truth_fragments(
    records: Sequence[Mapping],
    fragment_library: Mapping[int, Mapping],
    *,
    dataset: str,
    batch_size: int = 1024,
) -> dict[str, float | int]:
    """Reconstruct true motif targets and evaluate them with paper metrics."""
    if batch_size < 1:
        raise ValueError("Ground-truth reconstruction batch size must be positive.")
    accumulator = MoleculeMetricAccumulator(dataset=dataset, retain_molecules=False)
    for start in range(0, len(records), batch_size):
        payload = _reconstruction_payload(records[start : start + batch_size])
        reconstructed = reconstruct_atoms(payload, fragment_library)
        with suppress_native_output():
            accumulator.update(reconstructed)
    return {"molecules": len(records), **accumulator.metrics()}
