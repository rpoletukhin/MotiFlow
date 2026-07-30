"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

import logging
from typing import Tuple, Union

import numpy as np
import torch
from einops import rearrange

from motiflow.utils import rigid_utils as ru
from motiflow.utils.so3_condflowmatcher import SO3ConditionalFlowMatcher
from motiflow.utils.so3_helpers import expmap, hat_inv, log, pt_to_identity


class SO3FM:
    def __init__(self, so3_conf):
        self._log = logging.getLogger(__name__)
        self.so3_cfm = SO3ConditionalFlowMatcher()
        self.endpoint_epsilon = float(so3_conf.endpoint_epsilon)

    def sample_ref(
        self, n_samples: int, n_fragments: int, frag_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        # Noise rotation matrix from uniform quaternions
        q = torch.randn((n_samples * n_fragments, 4), device=device, dtype=torch.float32)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        rot_mats = ru.Rotation(quats=q.float()).get_rot_mats()  # [B*N, 3, 3]
        noise_rot = rot_mats.view(n_samples, n_fragments, 3, 3)
        # Enforce identity noise where there is no fragment
        if frag_mask is not None:
            noise_rot = torch.where(
                frag_mask[..., None, None].bool(),
                noise_rot,
                torch.eye(3, device=device, dtype=torch.float32)[None, None, :, :],
            )
        return noise_rot

    def forward_marginal(
        self,
        rot_1: torch.Tensor,
        t: torch.Tensor,
        frag_mask: torch.Tensor,
        rot_0: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the forward diffusion process at time index t.

        Args:
            rot_1: [B, N, 3, 3] data rotations
            t: scalar float or torch.Tensor of shape [B]
            rot_0: optional prior rotations [B, N, 3, 3]

        Returns:
            rot_t: [B, N, 3, 3] (float32, on same device)
            u_t: [B, N, 3, 3] data-directed matrix vector field.
        """
        B, N, _, _ = rot_1.shape
        device = rot_1.device
        if rot_0 is None:
            rot_0 = self.sample_ref(n_samples=B, n_fragments=N, frag_mask=frag_mask, device=device)
        # Flatten for sampling
        rot_0_flat = rearrange(rot_0, "b n r c -> (b n) r c", r=3, c=3).double()
        rot_1_flat = rearrange(rot_1, "b n r c -> (b n) r c", r=3, c=3).double()
        clamped_t = torch.clamp(t, min=0.0, max=1.0 - self.endpoint_epsilon)
        flat_t = clamped_t.repeat_interleave(N).double()
        rot_t = self.so3_cfm.sample_xt(rot_0_flat, rot_1_flat, flat_t)
        rot_t = rearrange(rot_t, "(b n) r c -> b n r c", b=B, r=3, c=3).float().to(device)
        _, vectorfield = self.vectorfield(rot_1, rot_t, clamped_t)
        return rot_t, vectorfield.float()

    def step(
        self,
        rot_t: torch.Tensor,
        v_t: torch.Tensor,
        t: float,
        dt: float,
        flow_mask: torch.Tensor,
    ):
        """
        Integrate the rotational ODE one step from ``t`` to ``t + dt``.

        Args:
            rot_t: [..., 3] current rotations at time t.
            v_t: [..., 3] rotation vector field at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            flow_mask: 1 indicates which fragments to flow.

        Returns:
            [..., 3] rotation vector at next step.
        """
        device = rot_t.device
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        perturb = v_t * dt

        if flow_mask is not None:
            perturb *= flow_mask[..., None, None]

        rot_t_1 = expmap(rot_t.double(), perturb.double()).float().to(device)
        rot_t_1 = rot_t_1.reshape(rot_t.shape)
        return rot_t_1

    def vectorfield(self, rot_1: torch.Tensor, rot_t: torch.Tensor, t: torch.Tensor):
        """
        Compute ``u_t(R_t | R_1) = log_{R_t}(R_1) / (1 - t)``.
        """
        B, N, _, _ = rot_1.shape

        t = torch.clamp(t, min=0.0, max=1.0 - self.endpoint_epsilon).repeat_interleave(N).double()
        rot_1 = rearrange(rot_1, "b n r c -> (b n) r c", r=3, c=3).double()
        rot_t = rearrange(rot_t, "b n r c -> (b n) r c", r=3, c=3).double()

        relative_to_data = rot_t.transpose(-1, -2) @ rot_1
        denominator = torch.clamp(1.0 - t[:, None, None], min=self.endpoint_epsilon)
        u_t = rot_t @ (log(relative_to_data) / denominator)
        u_t = rearrange(u_t, "(b n) r c -> b n r c", b=B, r=3, c=3)
        return None, u_t

    def compute_symmetric_target_vectors(
        self, rot_1: torch.Tensor, rot_t: torch.Tensor, t: torch.Tensor, symmetries: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the target vector fields (in vector form, shape [B, N, S, 3])
        for all possible symmetry alignments of the ground truth.

        Args:
            rot_1: [B, N, 3, 3] ground-truth canonical orientation.
            rot_t: [B, N, 3, 3] Noisy input orientation.
            t: [B] Time.
            symmetries: [B, N, S, 3, 3] Symmetry transformations.

        Returns:
            gt_rot_vecs: [B, N, S, 3] Vector fields in tangent space.
        """
        # Ensure double precision for stability
        rot_1 = rot_1.double()
        rot_t = rot_t.double()
        symmetries = symmetries.double()

        B, N, S, _, _ = symmetries.shape

        # Expand Inputs
        # T_expanded: [B*N*S]
        t_expanded = t.view(B, 1, 1).expand(-1, N, S).reshape(-1)

        # Generate Candidates: R_canon @ Symmetry_k
        # Shape: [B, N, S, 3, 3]
        rot_1_candidates = torch.matmul(rot_1.unsqueeze(2), symmetries)

        # Flatten for Vector Field Calculation
        # Rot_0_flat: [B*N*S, 1, 3, 3] (1 corresponds to the 'n' dim in vectorfield logic)
        rot_1_flat = rearrange(rot_1_candidates, "b n s r c -> (b n s) 1 r c")

        # Expand noisy rot_t to match candidates
        rot_t_expanded = rot_t.unsqueeze(2).expand(-1, -1, S, -1, -1)
        rot_t_flat = rearrange(rot_t_expanded, "b n s r c -> (b n s) 1 r c")

        # Calculate Matrix Vector Fields (u_t)
        # Output u_t_flat: [B*N*S, 1, 3, 3]
        _, u_t_flat = self.vectorfield(rot_1_flat, rot_t_flat, t_expanded)

        # Convert to Vector Form (tangent space at Identity)
        # Transport: R_t^T @ u_t
        at_id_flat = pt_to_identity(rot_t_flat.squeeze(1), u_t_flat.squeeze(1))
        # Hat Inverse: Skew-Matrix -> Vector
        gt_rot_vecs_flat = hat_inv(at_id_flat)  # [B*N*S, 3]

        # Reshape and return
        gt_rot_u_t_candidates = rearrange(gt_rot_vecs_flat, "(b n s) c -> b n s c", b=B, n=N, s=S).float()

        return gt_rot_u_t_candidates

    def _compute_per_symmetry_loss(
        self, gt_vecs: torch.Tensor, pred_vec: torch.Tensor, scaling: torch.Tensor, separate_rot_loss: bool
    ) -> torch.Tensor:
        """Helper to compute loss for each symmetry candidate individually."""
        # Gt_vecs: [B, N, S, 3]
        # Pred_vec: [B, N, 1, 3] (already expanded)
        # Scaling: [B, 1, 1]

        if separate_rot_loss:
            gt_angle = torch.norm(gt_vecs, dim=-1)  # [B, N, S]
            pred_angle = torch.norm(pred_vec, dim=-1).squeeze(-1)  # [B, N, 1] -> [B, N] (broadcasting handles dim 2)

            # Re-expand pred_angle for correct broadcasting: [B, N, 1]
            pred_angle = pred_angle.unsqueeze(-1)

            gt_axis = gt_vecs / (gt_angle.unsqueeze(-1) + 1e-6)
            pred_axis = pred_vec / (pred_angle.unsqueeze(-1) + 1e-6)

            angle_sq_diff = (gt_angle - pred_angle) ** 2
            angle_loss = angle_sq_diff / (scaling**2 + 1e-10)

            axis_sq_diff = torch.sum((gt_axis - pred_axis) ** 2, dim=-1)

            return angle_loss + axis_sq_diff
        else:
            sq_diff = (gt_vecs - pred_vec) ** 2
            mse_per_sym = torch.sum(sq_diff, dim=-1)
            return mse_per_sym / (scaling**2 + 1e-10)

    def compute_loss_geodiff(
        self,
        gt_candidates: torch.Tensor,
        pred_vec: torch.Tensor,
        sym_mask: torch.Tensor,
        scaling: torch.Tensor,
        separate_rot_loss: bool = False,
    ) -> torch.Tensor:
        """
        Aligns target to the noisy state (chooses the closest ground truth symmetry to x_t).
        Equivalent to choosing the symmetry with the smallest flow vector magnitude.
        """
        # Selection Metric: Magnitude of the flow vector (distance to target)
        # [B, N, S]
        dist_to_targets = torch.norm(gt_candidates, dim=-1)

        # Mask invalid symmetries
        dist_masked = dist_to_targets.masked_fill(~sym_mask.bool(), float("inf"))

        # Select index k that is closest to noisy state
        best_indices = torch.argmin(dist_masked, dim=2, keepdim=True)  # [B, N, 1]

        # Gather the specific target vector
        # [B, N, 1, 3]
        best_target_vec = torch.gather(gt_candidates, dim=2, index=best_indices.unsqueeze(-1).expand(-1, -1, -1, 3))

        # Compute Loss only on this target
        # Expand pred for helper compatibility: [B, N, 1, 3]
        pred_vec_expanded = pred_vec.unsqueeze(2)

        loss = self._compute_per_symmetry_loss(best_target_vec, pred_vec_expanded, scaling, separate_rot_loss)
        # Loss result is [B, N, 1], squeeze it
        return loss.squeeze(-1)

    def compute_loss_af3(
        self,
        gt_candidates: torch.Tensor,
        pred_vec: torch.Tensor,
        sym_mask: torch.Tensor,
        scaling: torch.Tensor,
        separate_rot_loss: bool = False,
    ) -> torch.Tensor:
        """
        Aligns target to the prediction (Min-of-N loss).
        Calculates loss for all symmetries and takes the minimum.
        """
        # Expand pred: [B, N, 1, 3]
        pred_vec_expanded = pred_vec.unsqueeze(2)

        # Compute loss for all symmetries
        # [B, N, S]
        all_losses = self._compute_per_symmetry_loss(gt_candidates, pred_vec_expanded, scaling, separate_rot_loss)

        # Mask invalid symmetries
        all_losses_masked = all_losses.masked_fill(~sym_mask.bool(), float("inf"))

        # Take Minimum
        best_loss, _ = torch.min(all_losses_masked, dim=2)
        return best_loss

    def compute_loss_naive(
        self,
        gt_candidates: torch.Tensor,
        pred_vec: torch.Tensor,
        sym_mask: torch.Tensor,  # not used, just to match the signature
        scaling: torch.Tensor,
        separate_rot_loss: bool = False,
    ) -> torch.Tensor:
        """
        Computes direct loss against the canonical target (S=0) without alignment.
        Assumes the first symmetry in gt_candidates corresponds to the Identity/Canonical frame.
        """
        # Select the canonical target (index 0)
        # Gt_candidates is [B, N, S, 3]. Slice to keep dims: [B, N, 1, 3]
        target_vec = gt_candidates[:, :, 0:1, :]

        # Expand prediction to match: [B, N, 1, 3]
        pred_vec_expanded = pred_vec.unsqueeze(2)

        # Compute loss
        # Result is [B, N, 1]
        loss = self._compute_per_symmetry_loss(target_vec, pred_vec_expanded, scaling, separate_rot_loss)
        return loss.squeeze(-1)  # [B, N]

    def vectorfield_scaling(self, t: torch.Tensor) -> float:
        return 1
