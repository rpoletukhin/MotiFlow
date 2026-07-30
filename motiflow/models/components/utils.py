import math
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy.stats import truncnorm
from torch import nn
from torch.nn import functional as F

from motiflow.data.conditioning import validate_conditioning_type

# Common and conditioning utils


def get_fragment_composition_tensor(fragment_library, vocab_size, num_atom_types=5):
    """
    Returns a tensor [vocab_size + 1, num_atom_types] containing the atom counts
    for each fragment class.
    """
    # Mapping for QM9: H=1, C=6, N=7, O=8, F=9 -> 0,1,2,3,4
    z_to_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    # +1 for the MASK token (which is empty/zero)
    counts_tensor = torch.zeros((vocab_size + 1, num_atom_types), dtype=torch.float32)

    for cid, entry in fragment_library.items():
        if cid >= vocab_size:
            continue

        # Entry['exemplar_z'] contains atomic numbers
        z_list = entry["exemplar_z"].tolist()
        for z in z_list:
            if z in z_to_idx:
                counts_tensor[cid, z_to_idx[z]] += 1.0

    return counts_tensor


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ConditioningEncoder(nn.Module):
    def __init__(self, cond_conf, out_dim):
        super().__init__()
        self.cond_type = validate_conditioning_type(str(cond_conf.type))

        if self.cond_type == "composition":
            self.num_types = cond_conf.num_atom_types
            self.emb_dim = cond_conf.atom_type_embed_dim
            self.atom_embedding = nn.Embedding(self.num_types, self.emb_dim)

            # Input is flattened (num_types * embed_dim)
            input_flat_dim = self.num_types * self.emb_dim
            hidden_dim = cond_conf.composition_hidden_dim

            self.mlp = nn.Sequential(
                Linear(input_flat_dim, hidden_dim, init="relu"),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim, init="relu"),
                nn.ReLU(),
                Linear(hidden_dim, out_dim, init="default"),
            )

        elif self.cond_type == "structure":
            in_dim = cond_conf.fingerprint_dim
            h_dims = cond_conf.structure_hidden_dims
            layers = []
            curr_dim = in_dim
            for h_dim in h_dims:
                layers.append(Linear(curr_dim, h_dim, init="relu"))
                layers.append(nn.ReLU())
                curr_dim = h_dim
            layers.append(Linear(curr_dim, out_dim, init="default"))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Identity()

    def forward(self, x):
        if self.cond_type == "none" or x is None:
            return None

        if self.cond_type == "composition":
            # X: [B, D] (proportions)
            B, D = x.shape
            all_idxs = torch.arange(D, device=x.device)
            embeddings = self.atom_embedding(all_idxs)  # [D, 64]
            # Weighted: [B, D, 1] * [1, D, 64] -> [B, D, 64]
            weighted = x.unsqueeze(-1) * embeddings.unsqueeze(0)
            return self.mlp(weighted.view(B, -1))

        elif self.cond_type == "structure":
            return self.mlp(x)


# IPA tensor and initialization utilities.


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class GaussianSmearing(nn.Module):
    """
    Expands a distance scalar into a vector of Gaussian radial basis functions.
    """

    def __init__(self, start, stop, num_gaussians):
        super().__init__()
        if stop <= start:
            raise ValueError(f"Invalid RBF range: start={start}, stop={stop}")

        offset = torch.linspace(start, stop, num_gaussians)
        # Width chosen so Gaussians overlap smoothly
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

        self.register_buffer("offset", offset)

    def forward(self, dist):
        # Dist: [..., 1] or [...]
        # Returns: [..., num_gaussians]
        if dist.dim() == 0:
            dist = dist.unsqueeze(0)

        diff = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(diff, 2))


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements a single direction of the Triangle Multiplicative Update.
    Standard definition: 'Outgoing' update.
    """

    def __init__(self, c_z, c_hidden):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.layer_norm_in = nn.LayerNorm(c_z)

        # Left projection (A)
        self.linear_a_p = Linear(c_z, c_hidden)
        self.linear_a_g = Linear(c_z, c_hidden, init="gating")

        # Right projection (B)
        self.linear_b_p = Linear(c_z, c_hidden)
        self.linear_b_g = Linear(c_z, c_hidden, init="gating")

        # Output projection
        self.linear_out = Linear(c_hidden, c_z, init="final")
        self.linear_g_out = Linear(c_z, c_z, init="gating")
        self.layer_norm_out = nn.LayerNorm(c_z)

    def forward(self, z, mask=None):
        """
        Args:
            z: [Batch, N, N, C_z]
            mask: [Batch, N, N]
        """
        z_in = z
        z = self.layer_norm_in(z)

        # Projections & Gating
        # [B, N, N, C_hidden]
        a = self.linear_a_p(z) * torch.sigmoid(self.linear_a_g(z))
        b = self.linear_b_p(z) * torch.sigmoid(self.linear_b_g(z))

        # Triangle Multiplication (Outgoing)
        # Equation: z_ij = sum_k (a_ik * b_jk)
        # We mask invalid edges 'k' to zero to prevent noise accumulation
        if mask is not None:
            # Mask A and B based on their respective 'k' dimension availability.
            # A: [B, i, k], mask: [B, i, k]
            a = a * mask.unsqueeze(-1)
            b = b * mask.unsqueeze(-1)

        # Einstein Summation:
        # A: [batch, i, k, channel]
        # B: [batch, j, k, channel] (Note indices j,k matches row j)
        # Out: [batch, i, j, channel]
        x = torch.einsum("bikc,bjkc->bijc", a, b)

        # Output Projection
        z_update = self.linear_out(x)
        g = torch.sigmoid(self.linear_g_out(z))
        z_update = z_update * g

        # Residual
        z_out = z_in + z_update
        z_out = self.layer_norm_out(z_out)

        return z_out


class TriangleInteractionBlock(nn.Module):
    """
    Combines Outgoing and Incoming Triangle Updates into a single full block.
    This ensures proper symmetry and information propagation.
    """

    def __init__(self, c_z, c_hidden):
        super().__init__()
        # Outgoing Update
        self.diff_outgoing = TriangleMultiplicativeUpdate(c_z, c_hidden)
        # Incoming Update (Applied to Transpose)
        self.diff_incoming = TriangleMultiplicativeUpdate(c_z, c_hidden)

    def forward(self, z, mask=None):
        # Outgoing
        z = self.diff_outgoing(z, mask=mask)

        # Incoming
        # Mathematically, Incoming update is the Outgoing update applied to Z transpose.
        # Transpose indices [B, N, N, C] -> [0, 2, 1, 3]
        z = z.permute(0, 2, 1, 3)

        # We must also transpose the mask for the incoming step
        mask_t = mask.permute(0, 2, 1) if mask is not None else None

        z = self.diff_incoming(z, mask=mask_t)

        # Transpose back
        z = z.permute(0, 2, 1, 3)

        return z
