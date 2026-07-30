#!/usr/bin/env python
"""Generate rigid-motif samples from a MotiFlow checkpoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from motiflow.inference import generate


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(config: DictConfig) -> None:
    print(generate(config))


if __name__ == "__main__":
    main()
