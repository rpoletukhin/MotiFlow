"""Reconstruct atom coordinates and known topology from generated motif frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from motiflow.utils import rigid_utils as ru


@dataclass(frozen=True)
class ReconstructedMolecule:
    positions: np.ndarray
    atomic_numbers: np.ndarray
    known_bonds: dict[tuple[int, int], int]
    fragment_ids: np.ndarray


def _samples(payload: Any) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(fragment_ids, rigids)`` from MotiFlow payloads."""
    if isinstance(payload, Mapping) and "samples" in payload:
        yield from _samples(payload["samples"])
        return
    if isinstance(payload, Mapping) and "rigids" in payload:
        ids = payload.get("fragment_ids", payload.get("frag_ids"))
        if ids is None:
            raise ValueError("Generated payload has rigids but no fragment IDs.")
        rigids = torch.as_tensor(payload["rigids"])
        ids = torch.as_tensor(ids)
        if rigids.ndim == 2:
            rigids, ids = rigids.unsqueeze(0), ids.unsqueeze(0)
        masks = payload.get("fragment_mask")
        for index in range(rigids.shape[0]):
            count = rigids.shape[1] if masks is None else int(torch.as_tensor(masks[index]).sum())
            yield ids[index, :count], rigids[index, :count]
        return
    if not isinstance(payload, (list, tuple)):
        raise TypeError("Generated data must be a payload dictionary or a list of sample mappings.")
    for entry in payload:
        ids = entry.get("frag_ids", entry.get("fragment_ids"))
        if ids is None or "rigids" not in entry:
            raise ValueError("Each sample needs frag_ids and rigids.")
        count = int(entry.get("num_frags", len(ids)))
        yield torch.as_tensor(ids)[:count], torch.as_tensor(entry["rigids"])[:count]


def reconstruct_atoms(payload: Any, library: Mapping[int, Mapping[str, Any]]) -> list[ReconstructedMolecule]:
    """Apply generated SE(3) frames to fragment exemplars.

    Dummy atoms (atomic number zero) define local frames and are discarded.
    Intra-fragment bonds are retained as overrides; inter-fragment bonds are
    deliberately left to the dataset-specific bond perception protocol.
    """
    reconstructed = []
    valid_ids = {int(key) for key in library}
    for sample_index, (fragment_ids, rigid_values) in enumerate(_samples(payload)):
        fragment_ids = fragment_ids.detach().cpu().long()
        rigid_values = rigid_values.detach().cpu().float()
        if rigid_values.ndim != 2 or rigid_values.shape[-1] != 7:
            raise ValueError(f"Sample {sample_index} has invalid rigid shape {tuple(rigid_values.shape)}.")
        if len(fragment_ids) != len(rigid_values) or len(fragment_ids) == 0:
            raise ValueError(f"Sample {sample_index} has inconsistent or empty fragments.")
        unknown = sorted(set(fragment_ids.tolist()) - valid_ids)
        if unknown:
            raise ValueError(f"Sample {sample_index} references unknown fragment classes {unknown}.")
        if not torch.isfinite(rigid_values).all():
            raise ValueError(f"Sample {sample_index} contains non-finite rigid values.")

        rigid = ru.Rigid.from_tensor_7(rigid_values)
        rotations = rigid.get_rots().get_rot_mats().numpy()
        translations = rigid.get_trans().numpy()
        all_positions: list[np.ndarray] = []
        all_numbers: list[np.ndarray] = []
        known_bonds: dict[tuple[int, int], int] = {}
        atom_offset = 0

        for local_fragment, class_value in enumerate(fragment_ids.tolist()):
            entry = library[int(class_value)]
            positions = np.asarray(torch.as_tensor(entry["exemplar_pos"]).cpu(), dtype=np.float64)
            numbers = np.asarray(torch.as_tensor(entry["exemplar_z"]).cpu(), dtype=np.int64)
            if positions.shape != (len(numbers), 3):
                raise ValueError(f"Fragment class {class_value} has inconsistent exemplar arrays.")
            world = positions @ rotations[local_fragment].T + translations[local_fragment]
            real_indices = np.flatnonzero(numbers != 0)
            if len(real_indices) == 0:
                continue
            local_to_global = {int(index): atom_offset + rank for rank, index in enumerate(real_indices)}
            all_positions.append(world[real_indices])
            all_numbers.append(numbers[real_indices])

            if "exemplar_edge_index" in entry and "exemplar_edge_attr" in entry:
                edge_index = np.asarray(torch.as_tensor(entry["exemplar_edge_index"]).cpu())
                edge_attr = np.asarray(torch.as_tensor(entry["exemplar_edge_attr"]).cpu()).reshape(-1)
                if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != len(edge_attr):
                    raise ValueError(f"Fragment class {class_value} has invalid exemplar topology.")
                for edge, raw_order in zip(edge_index.T, edge_attr):
                    first, second = int(edge[0]), int(edge[1])
                    if first not in local_to_global or second not in local_to_global or first == second:
                        continue
                    order = int(round(float(raw_order)))
                    if order not in {1, 2, 3}:
                        continue
                    pair = tuple(sorted((local_to_global[first], local_to_global[second])))
                    previous = known_bonds.get(pair)
                    if previous is not None and previous != order:
                        raise ValueError(f"Conflicting bond orders for reconstructed pair {pair}.")
                    known_bonds[pair] = order
            atom_offset += len(real_indices)

        if not all_positions:
            raise ValueError(f"Sample {sample_index} reconstructs to zero real atoms.")
        reconstructed.append(
            ReconstructedMolecule(
                positions=np.concatenate(all_positions).astype(np.float32),
                atomic_numbers=np.concatenate(all_numbers).astype(np.int64),
                known_bonds=known_bonds,
                fragment_ids=fragment_ids.numpy(),
            )
        )
    return reconstructed
