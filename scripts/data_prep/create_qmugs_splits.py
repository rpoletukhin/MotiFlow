"""Create the molecule-disjoint QMugs split used by MotiFlow."""

import argparse
import json
from pathlib import Path

from motiflow.data.splits import (
    QMUGS_PAPER_CHUNK_SIZE,
    QMUGS_PAPER_MW_QUANTILES,
    QMUGS_PAPER_SEED,
    QMUGS_PAPER_TARGET_CONFORMERS,
    build_qmugs_split_manifest,
    compare_qmugs_split_manifests,
    create_qmugs_splits,
    load_qmugs_paper_split_manifest,
    write_qmugs_split_chunks,
)
from motiflow.provenance import sha256_file, write_json_atomic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-conformers", type=int, default=QMUGS_PAPER_TARGET_CONFORMERS["train"])
    parser.add_argument("--val-conformers", type=int, default=QMUGS_PAPER_TARGET_CONFORMERS["val"])
    parser.add_argument("--test-conformers", type=int, default=QMUGS_PAPER_TARGET_CONFORMERS["test"])
    parser.add_argument("--mw-quantiles", type=int, default=QMUGS_PAPER_MW_QUANTILES)
    parser.add_argument("--seed", type=int, default=QMUGS_PAPER_SEED)
    parser.add_argument("--chunk-size", type=int, default=QMUGS_PAPER_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-paper-manifest",
        action="store_true",
        help="Fail unless the ordered IDs and cardinalities exactly match the frozen paper split.",
    )
    parser.add_argument(
        "--verify-source-sha256",
        action="store_true",
        help="Also hash the full summary.csv when verifying the paper manifest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets = {
        "train": args.train_conformers,
        "val": args.val_conformers,
        "test": args.test_conformers,
    }
    result = create_qmugs_splits(
        args.summary_csv,
        target_conformer_counts=targets,
        molecular_weight_quantiles=args.mw_quantiles,
        seed=args.seed,
    )
    manifest = build_qmugs_split_manifest(result, chunk_size=args.chunk_size)

    if args.verify_paper_manifest:
        expected = load_qmugs_paper_split_manifest()
        differences = compare_qmugs_split_manifests(manifest, expected)
        if args.verify_source_sha256:
            source_hash = sha256_file(args.summary_csv)
            expected_hash = expected["source"]["sha256"]
            if source_hash != expected_hash:
                differences.append(f"source.sha256: expected {expected_hash!r}, got {source_hash!r}")
        if differences:
            formatted = "\n".join(f"- {difference}" for difference in differences)
            raise RuntimeError(f"QMugs split does not match the frozen paper manifest:\n{formatted}")
        print("Verified exact agreement with the frozen paper split.")

    written = write_qmugs_split_chunks(
        result,
        args.output_dir,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    manifest["output_files"] = {
        split_name: [str(path.relative_to(args.output_dir)) for path in paths] for split_name, paths in written.items()
    }
    manifest_path = args.output_dir / "split_manifest.json"
    write_json_atomic(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
