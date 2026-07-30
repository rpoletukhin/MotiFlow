"""MotiFlow training objective.

The objective couples SE(3) conditional flow matching with masked discrete
flow matching under the prior-at-zero, data-at-one convention.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from einops import rearrange

from motiflow.utils import rigid_utils as ru
from motiflow.utils.so3_helpers import hat_inv, pt_to_identity


def _get(config: object, name: str):
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


def compute_motif_loss(
    *,
    model: torch.nn.Module,
    flow_matcher,
    batch: dict[str, torch.Tensor],
    model_config: object,
    flow_config: object,
    loss_config: object,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the joint MotiFlow translation, rotation, and categorical losses.

    The input batch must already have been passed through
    :meth:`SE3FlowMatcher.sample_times` and
    :meth:`SE3FlowMatcher.forward_marginal`.
    """
    batch["sc_logits"] = None
    use_self_conditioning = bool(_get(model_config, "use_self_conditioning"))
    self_conditioning_probability = float(_get(model_config, "sc_training_prob"))
    if use_self_conditioning and model.training:
        if torch.rand((), device=batch["frag_mask"].device) < self_conditioning_probability:
            with torch.no_grad():
                preliminary_output = model(batch)
                batch["sc_logits"] = preliminary_output["cat_logits"].detach()

    model_output = model(batch)
    loss_mask = batch["frag_mask"]
    batch_size, _ = loss_mask.shape
    nonempty_examples = torch.any(loss_mask.bool(), dim=-1)

    translation_loss = torch.zeros(batch_size, device=loss_mask.device)
    rotation_loss = torch.zeros(batch_size, device=loss_mask.device)
    categorical_loss = torch.zeros(batch_size, device=loss_mask.device)

    if bool(_get(flow_config, "flow_trans")):
        target = batch["trans_vectorfield"]
        scaling = batch["trans_vectorfield_scaling"]
        prediction = model_output["trans_vectorfield"] * loss_mask[..., None]
        squared_error = (target - prediction) ** 2 * loss_mask[..., None]
        translation_loss = torch.sum(
            squared_error / (scaling**2 + 1e-10),
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        translation_loss *= float(_get(loss_config, "trans_loss_weight"))

    if bool(_get(flow_config, "flow_rot")):
        target_candidates = flow_matcher._so3_fm.compute_symmetric_target_vectors(
            rot_1=ru.Rigid.from_tensor_7(batch["rigids_1"]).get_rots().get_rot_mats(),
            rot_t=batch["rot_t"],
            t=batch.get("t_se3", batch["t"]),
            symmetries=batch["symmetries"],
        )

        predicted_matrices = model_output["rot_vectorfield"].double()
        noisy_rotations = batch["rot_t"].double()
        _, num_fragments, _, _ = predicted_matrices.shape
        predicted_flat = rearrange(predicted_matrices, "b n r c -> (b n) r c")
        noisy_flat = rearrange(noisy_rotations, "b n r c -> (b n) r c")
        predicted_at_identity = pt_to_identity(noisy_flat, predicted_flat)
        predicted_vectors = rearrange(
            hat_inv(predicted_at_identity).float(),
            "(b n) c -> b n c",
            b=batch_size,
            n=num_fragments,
        )
        predicted_vectors *= loss_mask[..., None]

        loss_arguments = {
            "gt_candidates": target_candidates,
            "pred_vec": predicted_vectors,
            "sym_mask": batch["sym_mask"],
            "scaling": batch["rot_vectorfield_scaling"],
            "separate_rot_loss": bool(_get(loss_config, "separate_rot_loss")),
        }
        rotation_style = str(_get(loss_config, "rotation_loss_style"))
        loss_functions = {
            "af3": flow_matcher._so3_fm.compute_loss_af3,
            "geodiff": flow_matcher._so3_fm.compute_loss_geodiff,
            "naive": flow_matcher._so3_fm.compute_loss_naive,
        }
        if rotation_style not in loss_functions:
            choices = ", ".join(loss_functions)
            raise ValueError(f"Unknown rotation_loss_style {rotation_style!r}; choose from {choices}.")
        per_fragment_rotation_loss = loss_functions[rotation_style](**loss_arguments)
        invalid_present = torch.isinf(per_fragment_rotation_loss) & loss_mask.bool()
        if torch.any(invalid_present):
            raise ValueError("All symmetries are masked for at least one present fragment.")
        per_fragment_rotation_loss = torch.nan_to_num(per_fragment_rotation_loss, posinf=0.0)
        rotation_loss = torch.sum(per_fragment_rotation_loss * loss_mask, dim=-1)
        rotation_loss /= loss_mask.sum(dim=-1) + 1e-10
        rotation_loss *= float(_get(loss_config, "rot_loss_weight"))

    if bool(_get(flow_config, "flow_cat")):
        categorical_loss = flow_matcher._cat_fm.compute_loss(model_output, batch)
        categorical_loss *= float(_get(loss_config, "cat_loss_weight"))

    per_example_loss = rotation_loss + translation_loss + categorical_loss

    def normalize(values: torch.Tensor) -> torch.Tensor:
        return values.sum() / (nonempty_examples.sum() + 1e-10)

    metrics = {
        "batch_train_loss": per_example_loss,
        "batch_rot_loss": rotation_loss,
        "batch_trans_loss": translation_loss,
        "batch_cat_loss": categorical_loss,
        "total_loss": normalize(per_example_loss),
        "rot_loss": normalize(rotation_loss),
        "trans_loss": normalize(translation_loss),
        "cat_loss": normalize(categorical_loss),
        "examples_per_step": torch.tensor(batch_size, device=loss_mask.device),
        "frag_length": torch.mean(torch.sum(loss_mask, dim=-1)),
    }
    return metrics["total_loss"], metrics
