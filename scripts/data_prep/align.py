import glob
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger

from motiflow.data.loaders import get_qm9_splits, load_geom_tasks, load_qm9_tasks, load_qmugs_tasks
from motiflow.data.runner import run_pipeline
from motiflow.provenance import build_manifest, write_json_atomic


@hydra.main(version_base=None, config_path="../../configs", config_name="align")
def main(cfg: DictConfig):
    RDLogger.DisableLog("rdApp.*")
    config = OmegaConf.to_container(cfg, resolve=True)

    ds_cfg = config["dataset"]
    dataset_name = ds_cfg["name"]
    method_version = config["method_version"]

    if dataset_name == "qm9":
        split_dict = get_qm9_splits(config)
        splits = list(split_dict.keys())
    else:
        splits = ds_cfg.get("splits", [])

    for split in splits:
        print(f"\n--- Processing {dataset_name.upper()} : Split '{split}' ---")

        tasks = []
        shard_group = None
        if dataset_name == "qm9":
            tasks = load_qm9_tasks(config, split_dict[split])
        elif dataset_name == "geom":
            tasks = load_geom_tasks(config, split)
        elif dataset_name == "qmugs":
            target_ids = []
            chunk_override = ds_cfg.get("chunk_txt")
            if chunk_override:
                target_ids = [line.strip() for line in open(chunk_override) if line.strip()]
                shard_group = os.path.basename(chunk_override).replace(".txt", "")
            else:
                chunk_dir = os.path.join(ds_cfg["root"], ds_cfg["splits_dir"], f"{split}_ids")
                for c_file in sorted(glob.glob(os.path.join(chunk_dir, "*.txt"))):
                    target_ids.extend([line.strip() for line in open(c_file) if line.strip()])

            tasks = load_qmugs_tasks(config, target_ids)

        if not tasks:
            print(f"Warning: No tasks found for split '{split}'. Skipping.")
            continue

        output_path = os.path.join(config["output_root"], dataset_name, f"{split}_aligned.pt")
        temp_dir = os.path.join(config["output_root"], dataset_name, "temp", split)

        result = run_pipeline(
            tasks=tasks,
            dataset=dataset_name,
            output_path=output_path,
            temp_dir=temp_dir,
            batch_size=ds_cfg.get("batch_size"),
            num_workers=config["processing"]["num_workers"],
            aligner_kwargs=ds_cfg.get("aligner_kwargs", {}),
            task_timeout_seconds=config["processing"].get("task_timeout_seconds", 120.0),
            shard_group=shard_group,
        )

        if dataset_name == "qm9":
            input_files = [
                os.path.join(config["data_root"], ds_cfg["tar_path"]),
                os.path.join(config["data_root"], ds_cfg["splits_path"]),
            ]
        elif dataset_name == "geom":
            input_files = [os.path.join(config["data_root"], ds_cfg["split_files"][split])]
        else:
            input_files = [os.path.join(ds_cfg["root"], ds_cfg["summary_csv"])]
            if ds_cfg.get("chunk_txt"):
                input_files.append(ds_cfg["chunk_txt"])
            else:
                split_dir = os.path.join(ds_cfg["root"], ds_cfg["splits_dir"], f"{split}_ids")
                input_files.extend(sorted(glob.glob(os.path.join(split_dir, "*.txt"))))

        manifest = build_manifest(
            stage="align",
            dataset=dataset_name,
            method_version=method_version,
            config=config,
            project_root=Path(__file__).resolve().parents[2],
            inputs=input_files,
            outputs=result.get("output_files", [output_path]),
            counts={split: result["counts"]},
            failures=result["failures"],
        )
        if dataset_name == "qm9":
            manifest_path = f"{output_path}.manifest.json"
        else:
            manifest_dir = os.path.join(config["output_root"], dataset_name, split)
            if shard_group:
                manifest_dir = os.path.join(manifest_dir, str(shard_group))
            manifest_path = os.path.join(manifest_dir, "manifest.json")
        write_json_atomic(manifest, manifest_path)
        print(f"Saved provenance manifest to {manifest_path}")

        if dataset_name == "qmugs" and ds_cfg.get("chunk_txt"):
            break


if __name__ == "__main__":
    main()
