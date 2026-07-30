import logging

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class DiscreteFM:
    def __init__(self, cat_conf):
        self._conf = cat_conf
        self.vocab_size = int(cat_conf.vocab_size)
        self.prior = str(cat_conf.get("prior", "masked"))
        if self.prior not in {"masked", "uniform"}:
            raise ValueError(f"Unknown categorical prior {self.prior!r}.")
        LOGGER.info("Using %s categorical flow", self.prior)
        self.mask_token_idx = self.vocab_size
        self.sampling_method = cat_conf["sampling_method"]
        if self.sampling_method not in {"argmax", "multinomial"}:
            raise ValueError(f"Unknown categorical sampling method {self.sampling_method!r}.")
        self.stochasticity = cat_conf["stochasticity"]
        self.min_temp = cat_conf["min_temperature"]
        self.max_temp = cat_conf["max_temperature"]

    def sample_ref(
        self, n_samples: int, n_fragments: int, device: torch.device, frag_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Samples reference at t=0, which is the pure mask state.
        """
        # Create a tensor filled with the mask token index
        if self.prior == "uniform":
            return torch.randint(0, self.vocab_size, (n_samples, n_fragments), device=device)
        return torch.full((n_samples, n_fragments), self.mask_token_idx, device=device, dtype=torch.long)

    def forward_marginal(self, x_1: torch.Tensor, t: torch.Tensor, frag_mask: torch.Tensor):
        """
        Sample m_t | m_1.
        P(m_t = mask) = 1 - t
        P(m_t = m_1) = t

        Args:
            x_1: [B, N] LongTensor of data fragment IDs.
            t: [B] FloatTensor time in [0, 1].
        """
        B, N = x_1.shape
        # Expand time for broadcasting: [B, N]
        t_exp = t.view(B, 1).expand(B, N)

        corruption = torch.bernoulli(1.0 - t_exp).bool()

        # Where frag_mask is 0 (padding), we don't care
        if frag_mask is not None:
            # Only apply masking logic where frag_mask is True
            corruption = corruption & frag_mask.bool()

        # Construct xt
        x_t = x_1.clone()
        if self.prior == "uniform":
            noise = torch.randint(0, self.vocab_size, (B, N), device=x_1.device)
            x_t[corruption] = noise[corruption]
        else:
            x_t[corruption] = self.mask_token_idx

        return x_t

    def compute_loss(self, model_out: dict, batch: dict) -> torch.Tensor:
        """
        Computes Weighted Cross Entropy Loss.
        Returns: [B] tensor.
        """
        x_t = batch["cat_t"]  # [B, N]
        x_1 = batch["frag_ids"].long()  # [B, N]
        logits = model_out["cat_logits"]  # [B, N, V]
        frag_mask = batch["frag_mask"].bool()

        t = batch.get("t_cat", batch["t"])

        B, N, V = logits.shape

        if self.prior == "uniform":
            token_loss = F.cross_entropy(logits.reshape(-1, V), x_1.reshape(-1), reduction="none").view(B, N)
            return (token_loss * frag_mask.float()).sum(dim=1) / (frag_mask.sum(dim=1) + 1e-9)

        # Identify valid positions: valid fragment and currently masked.
        is_masked = x_t == self.mask_token_idx
        loss_mask = frag_mask & is_masked  # [B, N]

        # Compute per-token CE loss (unreduced)
        logits_flat = logits.reshape(-1, V)
        target_flat = x_1.reshape(-1)
        ce_loss_flat = F.cross_entropy(logits_flat, target_flat, reduction="none")
        ce_loss = ce_loss_flat.view(B, N)

        # Inverse masking-probability weighting for the fixed-token denominator.
        # Clamp t for stability. Shape [B, 1]
        weights = 1.0 / torch.clamp(1.0 - t, min=1e-4).view(B, 1)

        # Apply mask and weights
        # We only learn on tokens that are currently masked
        weighted_loss = ce_loss * weights * loss_mask.float()

        # Sum over atoms -> [B]
        loss_per_sample = weighted_loss.sum(dim=1)

        # Normalize per sample
        # Normalize by number of valid atoms (masked or not) to keep scale consistent
        # Dividing only by masked atoms causes high variance when t -> 0
        num_atoms = frag_mask.sum(dim=1).float() + 1e-9

        return loss_per_sample / num_atoms

    def temperature(self, t: float) -> float:
        """Return the linear temperature schedule at paper time ``t``."""
        return max(self.min_temp + (self.max_temp - self.min_temp) * (1.0 - t), 1e-4)

    def transition_probabilities(self, t: float, dt: float) -> tuple[float, float]:
        """Return Campbell et al. unmask and detailed-balance remask probabilities."""
        remaining_t = max(1.0 - t, 1e-5)
        p_unmask = min(dt * (1.0 + self.stochasticity * t) / remaining_t, 1.0)
        reaches_data = t + dt >= 1.0 - 1e-8
        p_remask = 0.0 if reaches_data else min(dt * self.stochasticity, 1.0)
        return p_unmask, p_remask

    def step(self, s_t: torch.Tensor, logits: torch.Tensor, t: float, dt: float, flow_mask: torch.Tensor):
        """
        Advance one paper-time step with stochastic error correction and annealing.
        """
        device = s_t.device
        B, N = s_t.shape

        curr_temp = self.temperature(t)

        # Compute Probabilities over x0
        scaled_logits = logits / curr_temp
        probs_x0 = F.softmax(scaled_logits, dim=-1)  # [B, N, V]

        if self.prior == "uniform":
            p_update = min(dt / max(1.0 - t, 1e-5), 1.0)
            should_update = torch.rand((B, N), device=device) < p_update
            if flow_mask is not None:
                should_update &= flow_mask.bool()
            if self.sampling_method == "argmax":
                new_samples = torch.argmax(probs_x0, dim=-1)
            else:
                new_samples = torch.multinomial(probs_x0.reshape(-1, self.vocab_size), 1).view(B, N)
            return torch.where(should_update, new_samples, s_t)

        p_unmask, p_remask = self.transition_probabilities(t, dt)

        # Identify Masks
        is_masked = s_t == self.mask_token_idx
        is_data = ~is_masked

        # Sample Transitions
        # Mask -> Data
        should_unmask = torch.rand((B, N), device=device) < p_unmask
        update_unmask = is_masked & should_unmask

        # Data -> Mask (Error Correction)
        should_remask = torch.rand((B, N), device=device) < p_remask
        update_remask = is_data & should_remask

        # Apply flow mask constraints
        if flow_mask is not None:
            mask_bool = flow_mask.bool()
            update_unmask = update_unmask & mask_bool
            update_remask = update_remask & mask_bool

        # Apply Updates
        s_next = s_t.clone()

        # Unmask: Sample new values
        if self.sampling_method == "argmax":
            # Note: Temperature has no effect on argmax, so this disables annealing
            new_x0_samples = torch.argmax(probs_x0, dim=-1)
        elif self.sampling_method == "multinomial":
            # Annealing happens here: high temp flattens probs, low temp sharpens them
            new_x0_samples = torch.multinomial(probs_x0.view(-1, self.vocab_size), 1).view(B, N)

        s_next[update_unmask] = new_x0_samples[update_unmask]

        # Remask: Revert to mask token
        s_next[update_remask] = self.mask_token_idx

        return s_next
