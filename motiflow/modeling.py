"""Construction helpers for the independent MotiFlow model stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from motiflow.models.components.network import VectorFieldNetwork
from motiflow.models.se3_fm import SE3FlowMatcher


@dataclass(frozen=True)
class ModelBundle:
    """Objects that must remain synchronized during training and sampling."""

    model: VectorFieldNetwork
    flow_matcher: SE3FlowMatcher
    fragment_library: dict[int, dict[str, Any]]


def load_fragment_library(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load and validate a fragment library on CPU."""
    library = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(library, dict) or not library:
        raise ValueError("Fragment library must be a non-empty dictionary.")
    ids = sorted(int(key) for key in library)
    if ids != list(range(len(ids))):
        raise ValueError("Fragment library IDs must be contiguous and zero-based.")
    for fragment_id, entry in library.items():
        if "symmetries" not in entry:
            raise ValueError(f"Fragment class {fragment_id} has no symmetry matrices.")
    return library


def build_model_bundle(*, model_config, flow_config, data_config, device: torch.device) -> ModelBundle:
    """Build the flow matcher and IPA model from one resolved configuration."""
    library = load_fragment_library(data_config.library_path)
    vocab_size = len(library)
    configured_vocab_size = data_config.get("vocab_size")
    if configured_vocab_size is not None and int(configured_vocab_size) != vocab_size:
        raise ValueError(f"Configured vocab_size={configured_vocab_size}, library size={vocab_size}.")

    resolved_flow = OmegaConf.create(OmegaConf.to_container(flow_config, resolve=True))
    resolved_model = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    resolved_flow.cat.vocab_size = vocab_size
    resolved_model.embed.vocab_size = vocab_size

    flow_matcher = SE3FlowMatcher(resolved_flow)
    model = VectorFieldNetwork(resolved_model, flow_matcher, library).to(device)
    return ModelBundle(model=model, flow_matcher=flow_matcher, fragment_library=library)
