import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger

from motiflow.data.loaders import get_qm9_splits, load_aligned_data
from motiflow.fragmentation.decomposition import (
    DEFAULT_MAX_AUTOMORPHISMS,
    DEFAULT_MAX_FAILURE_FRACTION,
    build_recursive_vocabulary,
)
from motiflow.fragmentation.errors import FragmentationFailureBudgetExceeded
from motiflow.fragmentation.manager import FragmentClassManager
from motiflow.fragmentation.pipeline import prepare_and_save_dataset, write_failure_quarantine
from motiflow.provenance import build_manifest, write_json_atomic


def _alpha_slug(alpha):
    return str(float(alpha)).rstrip("0").rstrip(".").replace(".", "p")


@hydra.main(version_base=None, config_path="../../configs", config_name="fragment")
def main(cfg: DictConfig):
    RDLogger.DisableLog("rdApp.*")
    config = OmegaConf.to_container(cfg, resolve=True)

    ds_cfg = config["dataset"]
    dataset_name = ds_cfg["name"]
    method_version = config["method_version"]

    # Resolve the configured split names.
    if dataset_name == "qm9":
        split_dict = get_qm9_splits(config)
        splits = list(split_dict.keys())
    else:
        splits = ds_cfg.get("splits", [])

    if not splits:
        raise ValueError(f"No splits defined for dataset '{dataset_name}'.")

    # Find the training split across dataset-specific naming schemes.
    train_split = next((s for s in splits if "train" in s.lower()), splits[0])

    print("\n=============================================")
    print(f" FRAGMENTING DATASET: {dataset_name.upper()}")
    print(f" METHOD VERSION: {method_version}")
    print("=============================================")

    aligned_root = config.get("aligned_root", "./data/preprocessed")
    alpha = config.get("alpha_threshold", 0.1)
    max_failure_fraction = config.get("max_failure_fraction", DEFAULT_MAX_FAILURE_FRACTION)
    max_automorphisms = config.get("max_automorphisms", DEFAULT_MAX_AUTOMORPHISMS)
    profile_name = f"alpha_{_alpha_slug(alpha)}" if config["fragmentation"]["retain_rings"] else "no_rings"
    output_dir = config.get("output_dir") or os.path.join(
        config.get("output_root", "./data/fragmented"), dataset_name, profile_name
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load either single-file splits or QMugs chunk directories.
    splits_dict = {}
    for split in splits:
        data = load_aligned_data(aligned_root, dataset_name, split)
        if data:
            splits_dict[split] = data
            print(f"Loaded {split}: {len(data)} molecules.")

    if train_split not in splits_dict or not splits_dict[train_split]:
        raise ValueError(f"Training split '{train_split}' could not be loaded. Cannot build vocab!")

    # Build the vocabulary from training data only.
    vocabulary_failures = []
    if config["fragmentation"]["retain_rings"]:
        try:
            allowed_vocab = build_recursive_vocabulary(
                splits_dict[train_split],
                alpha,
                max_failure_fraction=max_failure_fraction,
                failure_records=vocabulary_failures,
            )
        except FragmentationFailureBudgetExceeded as error:
            quarantine_path = write_failure_quarantine(
                vocabulary_failures,
                output_dir,
                max_failure_fraction,
            )
            raise FragmentationFailureBudgetExceeded(
                stage=error.stage,
                split=train_split,
                failures=error.failures,
                total=error.total,
                max_failure_fraction=error.max_failure_fraction,
                quarantine_path=quarantine_path,
            ) from error
    else:
        allowed_vocab = set()

    vocab_path = os.path.join(output_dir, "allowed_vocabulary.txt")
    if os.path.exists(vocab_path):
        with open(vocab_path) as file:
            existing_vocabulary = set(file.read().splitlines())
        if existing_vocabulary != allowed_vocab:
            raise FileExistsError(
                f"Existing vocabulary at {vocab_path} differs from the newly mined training vocabulary."
            )
    else:
        with open(vocab_path, "w") as file:
            for smiles in sorted(allowed_vocab):
                file.write(smiles + "\n")
    print(f"\nSaved {len(allowed_vocab)} fragments to {vocab_path}")

    # Initialize the geometric class manager.
    class_manager = FragmentClassManager(
        rmsd_thresh=config.get("rmsd_thresh", 1.0), max_dev_thresh=config.get("max_dev_thresh", 1.0)
    )

    # Run decomposition and ground-truth reconstruction validation.
    result = prepare_and_save_dataset(
        splits_dict=splits_dict,
        train_split_key=train_split,
        class_manager=class_manager,
        output_path=output_dir,
        allowed_vocabulary=allowed_vocab,
        unknown_class_policy=config["unknown_class_policy"],
        dataset_name=dataset_name,
        validate_reconstruction=config["reconstruction_validation"]["enabled"],
        validation_batch_size=config["reconstruction_validation"]["batch_size"],
        max_failure_fraction=max_failure_fraction,
        max_automorphisms=max_automorphisms,
        initial_failures=vocabulary_failures,
        output_shard_size=int(config["output_shard_size"]),
        resume_training=bool(config["resume_training"]),
    )

    project_root = Path(__file__).resolve().parents[2]
    input_files = []
    for split_name, split_data in splits_dict.items():
        shard_paths = getattr(split_data, "shard_paths", None)
        if shard_paths is not None:
            input_files.extend(shard_paths)
        else:
            input_files.append(os.path.join(aligned_root, dataset_name, f"{split_name}_aligned.pt"))
    output_files = [vocab_path, *result["output_files"]]
    manifest = build_manifest(
        stage="fragment",
        dataset=dataset_name,
        method_version=method_version,
        config=config,
        project_root=project_root,
        inputs=input_files,
        outputs=output_files,
        counts={
            **result["counts"],
            "allowed_structural_motifs": len(allowed_vocab),
        },
        failures=result["exclusions"],
    )
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_json_atomic(manifest, manifest_path)
    print(f"Saved provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
