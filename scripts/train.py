#!/usr/bin/env python
"""Train MotiFlow from centralized Hydra configurations."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from motiflow.training.runner import train


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train(config)
    print(OmegaConf.to_yaml(result))


if __name__ == "__main__":
    main()
