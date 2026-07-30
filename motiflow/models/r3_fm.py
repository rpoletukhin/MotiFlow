"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

from typing import Union

import numpy as np
import torch
from einops import rearrange

from motiflow.utils.condflowmatcher import ConditionalFlowMatcher


class R3FM:
    """Flow matcher for translations in R3.
    Args:
        r3_conf: R3 configuration.
    """

    def __init__(self, r3_conf):
        self._r3_conf = r3_conf
        self.r3_cfm = ConditionalFlowMatcher()

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def sample_ref(
        self, n_samples: int, n_fragments: int, frag_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        # Centered (w.r.t. actually present fragments) translation noise
        noise_trans = torch.randn((n_samples, n_fragments, 3), device=device, dtype=torch.float32)
        # Set noise to zero where there is no fragment
        noise_trans *= frag_mask[..., None]  # [B, N, 3]
        # Compute center of mass (only considering actual fragments) of the noise over the fragment
        com = noise_trans.sum(dim=1, keepdim=True) / frag_mask.sum(dim=1, keepdim=True)[..., None]  # [B, 1, 3]
        noise_trans -= com
        # Mask again to account for the shift in center of mass for masked fragments
        noise_trans *= frag_mask[..., None]  # [B, N, 3]
        # Assert that the noise is centered
        assert (noise_trans * frag_mask[..., None]).sum(dim=1).abs().max() < 1e-3
        return noise_trans

    def forward_marginal(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        frag_mask: torch.Tensor,
        x_0: Union[torch.Tensor, None] = None,
    ):
        """Sample the prior-to-data marginal ``p(x_t | x_1)``.

        Args:
            x_1: [B, N, 3] data positions.
            t: [B], tensor with continuous time in [0, 1].
            frag_mask: [B, N] mask indicating which fragments are present.
            x_0: Optional [B, N, 3] prior translation.

        Returns:
            x_t: [B, N, 3] positions at time t.
            u_t: [B, N, 3] data-directed vector field at time t.
        """
        B, N, _ = x_1.shape
        device = x_1.device

        if x_0 is None:
            x_0 = self.sample_ref(B, N, frag_mask, device=device)

        # Flatten x_0 and x_1
        x_0_flat = rearrange(x_0, "B N C -> (B N) C")
        x_1_flat = rearrange(x_1, "B N C -> (B N) C")
        t = torch.clamp(t, min=0.0, max=1.0 - 1e-4).repeat_interleave(N).to(x_1_flat)
        x_1_scaled = self._scale(x_1_flat)

        x_t_flat = self.r3_cfm.sample_xt(x_0_flat, x_1_scaled, t, epsilon=0)
        x_t = rearrange(x_t_flat, "(b n) d -> b n d", b=B, n=N)
        # Assert it is centered
        assert (x_t * frag_mask[..., None]).sum(dim=1).abs().max() < 1e-3

        u_t_flat = self.r3_cfm.compute_conditional_flow(x_0_flat, x_1_scaled, t, x_t_flat)
        u_t = rearrange(u_t_flat, "(b n) d -> b n d", b=B, n=N)
        # Assert it is centered as well
        assert (u_t * frag_mask[..., None]).sum(dim=1).abs().max() < 1e-3

        # Unscale x_t
        x_t = self._unscale(x_t)

        # Assert the masked positions are zero
        assert (x_t * (1 - frag_mask[..., None])).abs().max() < 1e-3
        assert (u_t * (1 - frag_mask[..., None])).abs().max() < 1e-3

        return x_t, u_t

    def step(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        t: float,
        dt: float,
        flow_mask: torch.Tensor,
    ):
        """Integrate the centered ODE one step from ``t`` to ``t + dt``.

        Args:
            x_t: [B, N, 3], current positions at time t in angstroms.
            vt: [B, N, 3], translation vectorfield at time t.
            t: [B], continuous time in [0, 1].
            dt: [B], continuous step size in [0, 1].
            flow_mask: [B, N], 1 indicates which fragment to update.

        Returns:
            [B, N, 3] positions at the next time.
        """
        B, N, _ = x_t.shape
        if not np.isscalar(t):
            raise ValueError(f"{t} must be a scalar.")

        x_t = self._scale(x_t)
        perturb = v_t * dt
        perturb *= flow_mask[..., None]
        x_t_1 = x_t + perturb
        com = torch.sum(x_t_1, axis=-2) / torch.sum(flow_mask, axis=-1)[..., None]
        x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def vectorfield_scaling(self, t: torch.Tensor):
        return 1

    def vectorfield(self, x_1, x_t, t, scale=False):
        """Return the data-directed vector field at ``x_t``."""
        if scale:
            x_t = self._scale(x_t)
            x_1 = self._scale(x_1)
        return (x_1 - x_t) / (1.0 - t + 1e-10)
