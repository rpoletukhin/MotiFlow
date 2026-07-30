"""Verify that local GEOM artifacts are exactly the paper split."""

import argparse
import json
from pathlib import Path

from motiflow.data.splits import load_geom_paper_split_manifest, verify_geom_split_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_geom_paper_split_manifest()
    differences = verify_geom_split_files(args.data_root, manifest)
    if differences:
        formatted = "\n".join(f"- {difference}" for difference in differences)
        raise RuntimeError(f"GEOM artifacts do not match the paper manifest:\n{formatted}")
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))
    print("Verified exact agreement with the frozen GEOM paper artifacts.")


if __name__ == "__main__":
    main()
