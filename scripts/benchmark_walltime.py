#!/usr/bin/env python
"""Run the paper wall-clock benchmark."""

import hydra
from omegaconf import DictConfig

from motiflow.benchmarking import benchmark_walltime


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_walltime")
def main(config: DictConfig) -> None:
    print(benchmark_walltime(config))


if __name__ == "__main__":
    main()
