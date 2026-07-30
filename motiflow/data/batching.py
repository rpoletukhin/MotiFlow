"""Length-aware batching for variable-size motif sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from motiflow.utils import rigid_utils

UNPADDED_FEATURES = {
    "condition",
    "eval_target",
    "rot_vectorfield_scaling",
    "t",
    "t_seq",
    "t_struct",
    "trans_vectorfield_scaling",
}
RIGID_FEATURES = {"rigids_1", "rigids_t"}
PAIR_FEATURES = {"rel_rots"}


def _pad_array(value, maximum_length: int, dimension: int = 0):
    """Right-pad one NumPy array or tensor along a selected dimension."""
    padding = maximum_length - value.shape[dimension]
    if padding < 0:
        raise ValueError(f"Cannot pad length {value.shape[dimension]} to {maximum_length}.")
    widths = [(0, 0)] * value.ndim
    widths[dimension] = (0, padding)
    if torch.is_tensor(value):
        shape = list(value.shape)
        shape[dimension] = padding
        zeros = torch.zeros(shape, dtype=value.dtype, device=value.device)
        return torch.cat((value, zeros), dim=dimension)
    return np.pad(value, widths)


def _pad_rigid(value: torch.Tensor, maximum_length: int) -> torch.Tensor:
    padding = maximum_length - value.shape[0]
    if padding < 0:
        raise ValueError(f"Cannot pad {value.shape[0]} rigid frames to {maximum_length}.")
    identities = rigid_utils.Rigid.identity(
        (padding,),
        dtype=value.dtype,
        device=value.device,
        requires_grad=False,
    )
    return torch.cat((value, identities.to_tensor_7()), dim=0)


def _pad_features(features: Mapping[str, object], maximum_length: int) -> dict[str, object]:
    padded = {}
    for name, value in features.items():
        if name in UNPADDED_FEATURES:
            padded[name] = value
        elif name in RIGID_FEATURES:
            padded[name] = _pad_rigid(value, maximum_length)
        else:
            padded[name] = _pad_array(value, maximum_length)
    for name in PAIR_FEATURES & padded.keys():
        padded[name] = _pad_array(padded[name], maximum_length, dimension=1)
    return padded


def length_batch(samples: Sequence[Mapping[str, object]], maximum_squared_size: int) -> dict[str, object]:
    """Collate similarly sized motif sets under a quadratic memory budget."""
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return {"dummy_batch": torch.rand(100)}
    ordered = sorted(samples, key=lambda sample: len(sample["frag_mask"]), reverse=True)
    maximum_length = len(ordered[0]["frag_mask"])
    maximum_examples = max(int(maximum_squared_size // maximum_length**2), 1)
    padded = [_pad_features(sample, maximum_length) for sample in ordered[:maximum_examples]]
    return torch.utils.data.default_collate(padded)


def create_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: Sampler,
    num_workers: int,
    maximum_squared_size: int,
    prefetch_factor: int,
) -> DataLoader:
    """Build the length-aware loader used by MotiFlow training."""
    multiprocessing = num_workers > 0
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(length_batch, maximum_squared_size=maximum_squared_size),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if multiprocessing else None,
        persistent_workers=multiprocessing,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context="fork" if multiprocessing else None,
    )
