"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

import logging
import math

import torch

from motiflow.utils import rigid_utils as ru
from motiflow.utils.rigid_helpers import (
    assemble_rigid_mat,
    extract_trans_rots_mat,
)

from .discrete_fm import DiscreteFM
from .r3_fm import R3FM
from .so3_fm import SO3FM


class SE3FlowMatcher:
    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf
        self._do_fm_rot = se3_conf.flow_rot
        self._so3_fm = SO3FM(self._se3_conf.so3)
        self._flow_trans = se3_conf.flow_trans
        self._r3_fm = R3FM(self._se3_conf.r3)
        self._do_fm_cat = se3_conf.flow_cat
        self._cat_fm = DiscreteFM(self._se3_conf.cat)

    def effective_time(self, t, modality: str):
        """Map base paper time to a modality-specific interpolation time."""
        key = {"se3": "nu_se3", "cat": "nu_cat"}[modality]
        nu = self._se3_conf.get(key)
        if nu is None:
            return t
        if torch.is_tensor(t):
            base = torch.clamp(t, 0.0, 1.0)
            return torch.sin(math.pi * 0.5 * (base ** float(nu))) ** 2
        base = max(0.0, min(1.0, float(t)))
        return math.sin(math.pi * 0.5 * (base ** float(nu))) ** 2

    def sample_times(self, batch: dict):
        B, _ = batch["frag_ids"].shape
        endpoint_epsilon = float(self._se3_conf.endpoint_epsilon)
        t_batch = torch.rand((B,), device=batch["frag_ids"].device, dtype=torch.float32) * (1.0 - endpoint_epsilon)
        batch["t"] = t_batch
        return batch

    def forward_marginal(
        self,
        batch: dict,
    ):
        """
        Args:
            batch: dict with keys:
                - "rigids_1": torch.Tensor with shape [B, N, 7] (quat + trans)
                - "frag_ids": torch.LongTensor [B, N]
                - "frag_mask": torch.Tensor [B, N], float32, mask

            rigids_1: [..., N] OpenFold Rigid objects
            t: continuous time in [0, 1].
            flow_mask: [..., N] indicating which fragments to flow.
            as_tensor_7:
            rigids_1: [..., N] data rigid objects at time t=1.

        Returns:
            batch: updated dict
        """
        B, N = batch["frag_ids"].shape

        rigids_1 = batch["rigids_1"]
        frag_mask = batch["frag_mask"]
        frag_ids = batch["frag_ids"]

        assert rigids_1.shape == (B, N, 7)
        assert frag_mask.shape == (B, N)
        assert frag_ids.shape == (B, N)

        rigids_1_obj = ru.Rigid.from_tensor_7(rigids_1)
        # Extract the translations, rotations, and categorical fragment types
        trans_1 = rigids_1_obj.get_trans()  # [B, N, 3]
        rot_1 = rigids_1_obj.get_rots().get_rot_mats()  # [B, N, 3, 3]
        cat_1 = frag_ids.long()  # do not one-hot encode for discrete flow
        assert trans_1.shape == (B, N, 3)
        assert rot_1.shape == (B, N, 3, 3)

        t_se3 = self.effective_time(batch["t"], "se3")
        t_cat = self.effective_time(batch["t"], "cat")

        # Flow on rotations
        if not self._do_fm_rot:
            rot_t = rot_1
            rot_vectorfield = torch.zeros_like(rot_1)
            rot_vectorfield_scaling = torch.ones((B,), device=rot_1.device, dtype=torch.float32)
        else:
            rot_t, rot_vectorfield = self._so3_fm.forward_marginal(rot_1, t_se3, frag_mask=frag_mask)
            rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t_se3)

        # Flow on translations
        if not self._flow_trans:
            trans_t = trans_1
            trans_vectorfield = torch.zeros_like(trans_1)
            trans_vectorfield_scaling = 1
        else:
            trans_t, trans_vectorfield = self._r3_fm.forward_marginal(trans_1, t_se3, frag_mask=frag_mask)
            trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t_se3)

        # Flow on categorical fragment types
        if not self._do_fm_cat:
            cat_t = cat_1
            cat_cond_flow = torch.zeros_like(cat_t)
            trans_vectorfield_scaling = 1
        else:
            # DiscreteFlowMatcher expects indices, returns indices
            cat_t = self._cat_fm.forward_marginal(x_1=cat_1, t=t_cat, frag_mask=frag_mask)
            cat_cond_flow = None

        # Not sure if that is needed anymore, but won't hurt
        if frag_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_1, frag_mask[..., None, None])
            trans_t = self._apply_mask(trans_t, trans_1, frag_mask[..., None])
            trans_vectorfield = self._apply_mask(
                trans_vectorfield,
                torch.zeros_like(trans_vectorfield),
                frag_mask[..., None],
            )
            rot_vectorfield = self._apply_mask(
                rot_vectorfield, torch.zeros_like(rot_vectorfield), frag_mask[..., None, None]
            )

        rigids_t = ru.Rigid(
            rots=ru.Rotation(rot_mats=rot_t),
            trans=trans_t,
        )
        rigids_t = rigids_t.to_tensor_7()

        batch.update(
            {
                "rigids_t": rigids_t,
                "trans_vectorfield": trans_vectorfield,
                "rot_vectorfield": rot_vectorfield,
                "rot_t": rot_t,
                "trans_vectorfield_scaling": trans_vectorfield_scaling,
                "rot_vectorfield_scaling": rot_vectorfield_scaling,
                "cat_t": cat_t,
                "cat_cond_flow": cat_cond_flow,
                "t_se3": t_se3,
                "t_cat": t_cat,
            }
        )

        return batch

    def calc_trans_vectorfield(self, trans_1, trans_t, t, scale=True):
        return self._r3_fm.vectorfield(trans_1, trans_t, t, scale=scale)

    def calc_rot_vectorfield(self, rot_1, rot_t, t):
        return self._so3_fm.vectorfield(rot_1, rot_t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def vectorfield_scaling(self, t):
        rot_vectorfield_scaling = self._so3_fm.vectorfield_scaling(t)
        trans_vectorfield_scaling = self._r3_fm.vectorfield_scaling(t)
        return rot_vectorfield_scaling, trans_vectorfield_scaling

    def scheduled_step(self, t: float, dt: float, modality: str) -> tuple[float, float]:
        current = float(self.effective_time(t, modality))
        following = float(self.effective_time(min(t + dt, 1.0), modality))
        return current, max(following - current, 0.0)

    def step(
        self,
        rigid_t: ru.Rigid,
        cat_t: torch.Tensor,
        rot_vectorfield: torch.Tensor,
        trans_vectorfield: torch.Tensor,
        cat_vectorfield: torch.Tensor,  # This will actually be the logits [B, N, V]
        t: float,
        dt: float,
        flow_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, ru.Rigid, torch.Tensor]:
        """
        Advance the paper-time generative process from ``t`` to ``t + dt``.

        Args:
            rigid_t: [B, N] molecular rigid objects at time t.
            rot_vectorfield: [B, N, 3] rotation vector field.
            trans_vectorfield: [B, N, 3] translation vector field.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [B, N] which rigid fragment to update.

        Returns:
            rot_next: [B, N, 3, 3] rotations at time t + dt.
            trans_next: [B, N, 3] translations at time t + dt.
            rigid_next: [B, N] molecular rigid objects at time t + dt.
            cat_next: [B, N, V] categorical features at time t + dt.
        """
        trans_t, rot_t = extract_trans_rots_mat(rigid_t)
        t_se3, dt_se3 = self.scheduled_step(t, dt, "se3")
        t_cat, dt_cat = self.scheduled_step(t, dt, "cat")

        # Advance rotations.
        if not self._do_fm_rot:
            rot_next = rot_t
        else:
            rot_next = self._so3_fm.step(
                rot_t=rot_t,
                v_t=rot_vectorfield,
                t=t_se3,
                dt=dt_se3,
                flow_mask=flow_mask,
            )
        # Advance translations.
        if not self._flow_trans:
            trans_next = trans_t
        else:
            trans_next = self._r3_fm.step(
                x_t=trans_t,
                v_t=trans_vectorfield,
                t=t_se3,
                dt=dt_se3,
                flow_mask=flow_mask,
            )
        # Advance categorical fragment types.
        if not self._do_fm_cat:
            cat_next = cat_t
        else:
            # Note: cat_vectorfield here is actually the logits passed from inference_fn
            cat_next = self._cat_fm.step(s_t=cat_t, logits=cat_vectorfield, t=t_cat, dt=dt_cat, flow_mask=flow_mask)

        if flow_mask is not None:
            trans_next = self._apply_mask(trans_next, trans_t, flow_mask[..., None])
            rot_next = self._apply_mask(rot_next, rot_t, flow_mask[..., None, None])
        return (rot_next, trans_next, assemble_rigid_mat(rot_next, trans_next), cat_next)

    def sample_ref(
        self,
        n_samples: int,
        n_fragments: int,
        device: torch.device,
        flow_mask: torch.Tensor,
        impute_mask: torch.Tensor,
        impute: ru.Rigid = None,
        impute_cat: torch.Tensor = None,
        as_tensor_7: bool = False,
    ):
        """
        Sample the reference state at ``t=0``.

        Args:
            n_samples: Batch size.
            n_fragments: Number of fragments.
            device: Torch device.
            flow_mask: [B, N] mask indicating which fragments to flow.
            impute: [B, N] Rigid object for fixed fragments.
            impute_cat: [B, N, V] categorical features for fixed fragments.
            impute_mask: [B, N] mask (1.0 = sample noise, 0.0 = keep impute).
            as_tensor_7: Return rigid as tensor.

        Returns:
            Dict containing "rigids_t" (Rigid or Tensor7) and "cat_t" (Tensor).
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = extract_trans_rots_mat(impute)
            assert trans_impute.shape == (n_samples, n_fragments, 3)
            assert rot_impute.shape == (n_samples, n_fragments, 3, 3)
            trans_impute = self._r3_fm._scale(trans_impute)

        if ((not self._do_fm_rot) or (not self._flow_trans) or (not self._do_fm_cat)) and impute is None:
            raise ValueError("Must provide imputation values.")

        if self._do_fm_rot:
            rot_ref = self._so3_fm.sample_ref(
                n_samples=n_samples, n_fragments=n_fragments, device=device, frag_mask=flow_mask
            )
        else:
            rot_ref = rot_impute

        if self._flow_trans:
            trans_ref = self._r3_fm.sample_ref(
                n_samples=n_samples, n_fragments=n_fragments, device=device, frag_mask=flow_mask
            )
        else:
            trans_ref = trans_impute

        if self._do_fm_cat:
            cat_ref = self._cat_fm.sample_ref(n_samples, n_fragments, device, flow_mask)
        else:
            cat_ref = impute_cat

        if impute_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, impute_mask[..., None])
            trans_ref = self._apply_mask(trans_ref, trans_impute, impute_mask[..., None])
            if self._do_fm_cat:
                # For discrete, impute_cat is indices
                mask_bool = impute_mask.bool()
                cat_ref = torch.where(mask_bool, cat_ref, impute_cat.long())
        trans_ref = self._r3_fm._unscale(trans_ref)
        rigids_t = assemble_rigid_mat(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        out = {"rigids_t": rigids_t, "cat_t": cat_ref}
        return out
