"""Paper-time prior-to-data sampling for MotiFlow."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from motiflow.utils import rigid_utils as ru


@dataclass(frozen=True)
class SamplingResult:
    """Final generated state and optional trajectory."""

    rigids: torch.Tensor
    fragment_ids: torch.Tensor | None
    trajectory: tuple[torch.Tensor, ...]


def _center_rigids(rigids: torch.Tensor, fragment_mask: torch.Tensor) -> torch.Tensor:
    rigid_object = ru.Rigid.from_tensor_7(rigids)
    mask = fragment_mask[..., None]
    translations = rigid_object.get_trans()
    center = torch.sum(translations * mask, dim=1, keepdim=True)
    center /= torch.sum(mask, dim=1, keepdim=True) + 1e-6
    centered = (translations - center) * mask
    return ru.Rigid(rots=rigid_object.get_rots(), trans=centered).to_tensor_7()


def _clone_features(features: dict) -> dict:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in features.items()
    }


def sample(
    *,
    model: torch.nn.Module,
    flow_matcher,
    initial_features: dict[str, torch.Tensor],
    num_steps: int,
    vocab_size: int,
    use_self_conditioning: bool,
    keep_trajectory: bool = False,
) -> SamplingResult:
    """Integrate the generative process from the prior at t=0 to data at t=1."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    features = _clone_features(initial_features)
    device = features["rigids_t"].device
    features["rigids_t"] = _center_rigids(features["rigids_t"], features["frag_mask"])
    batch_size = features["rigids_t"].shape[0]
    placeholders = torch.ones(batch_size, device=device)
    time_edges = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float64)
    trajectory = [features["rigids_t"].detach().cpu()] if keep_trajectory else []
    self_conditioning_logits = None
    final_output = None

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for time_value, next_time_value in zip(time_edges[:-1], time_edges[1:]):
                time = float(time_value)
                dt = float(next_time_value - time_value)
                features["t"] = time * placeholders
                effective_time = getattr(flow_matcher, "effective_time", lambda value, _: value)
                spatial_time = effective_time(time, "se3")
                categorical_time = effective_time(time, "cat")
                features["t_se3"] = spatial_time * placeholders
                features["t_cat"] = categorical_time * placeholders
                rotation_scaling, translation_scaling = flow_matcher.vectorfield_scaling(spatial_time)
                features["rot_vectorfield_scaling"] = rotation_scaling * placeholders
                features["trans_vectorfield_scaling"] = translation_scaling * placeholders
                features["sc_logits"] = self_conditioning_logits
                final_output = model(features)

                if use_self_conditioning:
                    self_conditioning_logits = final_output.get("cat_logits")

                categorical_state = features.get("cat_t") if flow_matcher._do_fm_cat else None
                _, _, next_rigids, next_categories = flow_matcher.step(
                    rigid_t=ru.Rigid.from_tensor_7(features["rigids_t"]),
                    cat_t=categorical_state,
                    rot_vectorfield=final_output["rot_vectorfield"],
                    trans_vectorfield=final_output["trans_vectorfield"],
                    cat_vectorfield=final_output.get("cat_logits"),
                    flow_mask=features["frag_mask"],
                    t=time,
                    dt=dt,
                )
                features["rigids_t"] = next_rigids.to_tensor_7().to(device)
                if flow_matcher._do_fm_cat:
                    features["cat_t"] = next_categories
                if keep_trajectory:
                    trajectory.append(features["rigids_t"].detach().cpu())
    finally:
        model.train(model_was_training)

    fragment_ids = None
    if flow_matcher._do_fm_cat:
        fragment_ids = features["cat_t"]
        still_masked = fragment_ids == int(vocab_size)
        if still_masked.any():
            if final_output is None:
                raise RuntimeError("Sampling produced no model output.")
            best_guess = torch.argmax(final_output["cat_logits"], dim=-1)
            fragment_ids = torch.where(still_masked, best_guess, fragment_ids)

    return SamplingResult(
        rigids=features["rigids_t"],
        fragment_ids=fragment_ids,
        trajectory=tuple(trajectory),
    )
