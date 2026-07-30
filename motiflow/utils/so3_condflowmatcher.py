import torch
from einops import rearrange
from functorch import vmap

from motiflow.utils.so3_helpers import log


class SO3ConditionalFlowMatcher:
    def __init__(self):
        self.sigma = None

    def sample_xt(self, x0, x1, t):
        """Interpolate from ``x0`` to ``x1`` along the SO(3) geodesic."""
        relative = x0.transpose(-1, -2) @ x1
        relative_log = log(relative)
        return x0 @ torch.linalg.matrix_exp(t.reshape(-1, 1, 1) * relative_log)

    def compute_conditional_flow_simple(self, t, xt):
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[0]

        xt_dot = vmap(index_time_der, in_dims=1)(torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1))
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    def sample_location_and_conditional_flow_simple(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
        t.requires_grad = True
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow_simple(t, xt)

        return t, xt, ut
