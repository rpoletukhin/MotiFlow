#!/usr/bin/env python
"""Evaluate generated MotiFlow samples."""

import hydra
from omegaconf import DictConfig

from motiflow.evaluation.runner import evaluate


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    print(evaluate(config))


if __name__ == "__main__":
    main()
