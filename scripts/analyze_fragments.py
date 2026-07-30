#!/usr/bin/env python
"""Compute paper appendix statistics from fragmented artifacts and samples."""

import hydra
from omegaconf import DictConfig

from motiflow.analysis import analyze_fragment_statistics


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_fragments")
def main(config: DictConfig) -> None:
    print(analyze_fragment_statistics(config))


if __name__ == "__main__":
    main()
