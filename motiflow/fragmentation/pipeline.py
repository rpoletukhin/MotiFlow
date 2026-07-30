import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

from motiflow.data.conditioning import qm9_conditioning_targets
from motiflow.provenance import write_json_atomic

from .decomposition import (
    DEFAULT_MAX_AUTOMORPHISMS,
    DEFAULT_MAX_FAILURE_FRACTION,
    _filter_perms_preserve_atomic_numbers,
    _sample_failure_record,
    _validate_failure_fraction,
    decompose_to_rigid_fragments,
    decompose_with_frozen_fallback,
)
from .errors import FragmentationFailureBudgetExceeded, FrozenFragmentLibraryError
from .manager import FragmentClassManager, FragmentData
from .math_utils import is_geometrically_degenerate, numpy_to_torch_R_t, robust_kabsch_symmetry, rotation_angle_distance
from .validation import evaluate_ground_truth_fragments


def _ordered_split_names(splits_dict, train_split_key):
    """Return split names with training first, independent of config order."""
    if train_split_key not in splits_dict:
        raise KeyError(f"Training split '{train_split_key}' is missing from loaded splits.")
    return [train_split_key] + [name for name in splits_dict if name != train_split_key]


def compute_se3_targets(
    fragments: list[FragmentData],
    class_manager: FragmentClassManager,
) -> list[FragmentData]:
    """Compute ground-truth rotation and translation for each classified fragment."""
    if class_manager is None:
        raise ValueError("compute_se3_targets requires a FragmentClassManager")

    for i, frag in enumerate(fragments):
        # Skip already-computed
        if frag.rot_gt is not None and frag.trans_gt is not None:
            continue
        if frag.class_id is None:
            raise ValueError(f"Fragment {i} ({frag.smiles}) has no class_id")

        class_idx = frag.class_id
        if class_idx < 0 or class_idx >= len(class_manager.classes):
            raise ValueError(f"Invalid class index {class_idx} for fragment {i}")

        cls = class_manager.classes[class_idx]
        exemplar_pos = cls.exemplar_pos.astype(np.float64)
        exemplar_z = cls.exemplar_z.astype(np.int64)
        perms = cls.perms

        if exemplar_pos.shape[0] != frag.pos.shape[0]:
            raise ValueError(
                f"Atom count mismatch between exemplar (N={exemplar_pos.shape[0]}) "
                f"and fragment {i} (N={frag.pos.shape[0]})"
            )

        P_inst = frag.pos.detach().cpu().numpy().astype(np.float64)
        P_tmpl = exemplar_pos
        z = exemplar_z

        weights_center = class_manager._weights_for_centering(z)
        sel = class_manager._select_orientation_indices(z)
        if sel.size == 0:
            raise ValueError("No orientation indices found for fragment")

        geom_type = is_geometrically_degenerate(P_tmpl)
        allow_reflection_globally = geom_type != "nondegenerate"

        best_rmsd = float("inf")
        best_info = None

        weights_rotation = class_manager._atom_weights_from_z(z)

        for perm in perms:
            P_tmpl_perm = P_tmpl[perm]
            P_sub = P_tmpl_perm[sel]
            Q_sub = P_inst[sel]

            w_sub = weights_rotation[sel] if weights_rotation is not None else None
            if w_sub is None or np.sum(w_sub) <= 0:
                w_sub_pass = None
            else:
                w_sub_pass = w_sub

            R_np, t_np, _, _, reflected = robust_kabsch_symmetry(P_sub, Q_sub, weights=w_sub_pass)

            # Strict 3D Reflection Check
            if reflected and not allow_reflection_globally:
                continue

            P_pred_all = (R_np @ P_tmpl_perm.T).T + t_np
            per_atom_full = np.linalg.norm(P_pred_all - P_inst, axis=1)

            if np.sum(weights_center) <= 0:
                weighted_mse = float(np.mean(per_atom_full**2))
            else:
                weighted_mse = float(np.sum(weights_center * (per_atom_full**2)) / np.sum(weights_center))
            weighted_rmsd = float(np.sqrt(weighted_mse))

            if weighted_rmsd < best_rmsd:
                best_rmsd = weighted_rmsd
                best_info = (R_np, t_np, weighted_rmsd, per_atom_full, reflected, perm)

        if best_info is None:
            raise ValueError(f"No valid alignment found for fragment {i} of class {class_idx}")

        best_R_np, _, best_rmsd, _, _, best_perm = best_info

        # Reconstruct the specific permutation used for the match
        cls = class_manager.classes[class_idx]
        P_tmpl = cls.exemplar_pos.astype(np.float64)  # Canonical
        P_tmpl_perm = P_tmpl[best_perm]  # Permuted

        # Compute the symmetry S that maps Permuted -> Canonical
        # S @ P_perm.T = P_canon.T
        z = cls.exemplar_z.astype(np.int64)
        weights_rotation = class_manager._atom_weights_from_z(z)
        sel = class_manager._select_orientation_indices(z)

        # Align Permuted (Source) to Canonical (Target)
        S_corr, _, _, _, _ = robust_kabsch_symmetry(P_tmpl_perm[sel], P_tmpl[sel], weights=weights_rotation[sel])

        # Correct the Rotation to map Canonical -> Global
        # Current best_R_np maps Permuted -> Global
        # We want: R_canon @ P_canon.T = Global
        # Substitute P_canon.T = S @ P_perm.T
        # R_canon @ S @ P_perm.T = Global
        # Implies R_canon @ S = best_R_np  =>  R_canon = best_R_np @ S.T
        R_target_canon = best_R_np @ S_corr.T

        # Re-calculate translation for consistency
        # T = Centroid_Global - R_canon @ Centroid_Canon
        # (Using weights to match definition)
        weights_center = class_manager._weights_for_centering(z)

        def w_mean(pts, w):
            if np.sum(w) > 1e-12:
                return np.average(pts, axis=0, weights=w)
            return np.mean(pts, axis=0)

        cP = w_mean(P_tmpl, weights_center)
        cQ = w_mean(frag.pos.detach().cpu().numpy(), weights_center)

        t_target_canon = cQ - (R_target_canon @ cP)

        # Save corrected targets
        device = frag.pos.device if isinstance(frag.pos, torch.Tensor) else torch.device("cpu")
        R_torch, t_torch = numpy_to_torch_R_t(R_target_canon, t_target_canon, device=device)
        frag.rot_gt = R_torch
        frag.trans_gt = t_torch

    return fragments


def generate_orbit_from_exemplar_perms(
    class_manager: FragmentClassManager,
    angle_tol: float = 1e-3,
) -> dict[int, list[np.ndarray]]:
    """Detect exemplar symmetries using positive rotation weights."""
    candidates_by_class: dict[int, list[np.ndarray]] = {}
    symmetry_mse_threshold = 0.01

    for cid, cls in enumerate(class_manager.classes):
        exemplar = cls.exemplar_pos.astype(np.float64)
        z = cls.exemplar_z.astype(np.int64)
        weights_rotation = class_manager._atom_weights_from_z(z)

        perms_raw = [list(p) for p in cls.perms]
        perms_filtered = _filter_perms_preserve_atomic_numbers(perms_raw, z)
        seen = set()
        perms_unique = []
        for p in perms_filtered:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                perms_unique.append(p)
        perms_unique.sort(key=lambda x: tuple(x))

        raw_candidates: list[np.ndarray] = []
        sel = class_manager._select_orientation_indices(z)

        for perm in perms_unique:
            perm = np.asarray(perm, dtype=int)
            if perm.shape[0] != exemplar.shape[0]:
                continue

            P_sub = exemplar[perm][sel]
            Q_sub = exemplar[sel]
            w_sub = weights_rotation[sel]

            R_np, t_np, _, _, _ = robust_kabsch_symmetry(P_sub, Q_sub, weights=w_sub)

            P_pred_all = (R_np @ exemplar[perm].T).T + t_np
            per_atom_full = np.linalg.norm(P_pred_all - exemplar, axis=1)

            w_sum = np.sum(weights_rotation)
            weighted_mse_full = float(np.sum(weights_rotation * (per_atom_full**2)) / w_sum)

            if weighted_mse_full < symmetry_mse_threshold:
                raw_candidates.append(R_np)

        deduped: list[np.ndarray] = []
        for R in raw_candidates:
            keep = True
            for R_existing in deduped:
                # Use global utility
                if rotation_angle_distance(R_existing, R) <= angle_tol:
                    keep = False
                    break
            if keep:
                deduped.append(R.copy())

        if len(deduped) == 0:
            deduped.append(np.eye(3, dtype=float))

        cls.candidate_symmetries_exemplar = deduped
        candidates_by_class[cid] = deduped

    return candidates_by_class


def write_failure_quarantine(
    failures: list[dict[str, Any]],
    output_path: str,
    max_failure_fraction: float,
) -> str:
    quarantine_path = os.path.join(output_path, "fragmentation_failures.json")
    write_json_atomic(
        {
            "schema_version": "1.0",
            "max_failure_fraction": max_failure_fraction,
            "num_failures": len(failures),
            "failures": failures,
        },
        quarantine_path,
    )
    return quarantine_path


def _save_processed_shard(records, output_path: str, split_name: str, shard_index: int) -> str:
    """Atomically save one bounded-memory processed split shard."""
    shard_dir = Path(output_path) / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    target = shard_dir / f"shard_{shard_index:05d}.pt"
    with tempfile.NamedTemporaryFile(dir=shard_dir, prefix=f".{target.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(records, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json_atomic(
        {"schema_version": "1.0", "split": split_name, "record_count": len(records)},
        target.with_suffix(".manifest.json"),
    )
    return str(target)


def _processed_record(
    data,
    molecule_index: int,
    fragments: list[FragmentData],
    fallback_history: list[dict[str, str]] | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Convert one decomposition to the stable serialized training schema."""
    record = {
        "orig_mol_index": molecule_index,
        "sample_id": str(getattr(data, "sample_id", molecule_index)),
        "frag_ids": torch.tensor([fragment.class_id for fragment in fragments], dtype=torch.long),
        "trans": torch.stack([fragment.trans_gt for fragment in fragments]),
        "rots": torch.stack([fragment.rot_gt for fragment in fragments]),
        "num_frags": len(fragments),
        "original_indices": [fragment.original_indices for fragment in fragments],
    }
    if fallback_history:
        record["fragmentation_fallbacks"] = fallback_history
    if dataset_name == "qm9":
        record.update(qm9_conditioning_targets(data))
    return record


def _save_torch_atomic(payload: Any, target: Path) -> str:
    """Atomically save a PyTorch artifact beside its final destination."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(target)


def _allowed_vocabulary_fingerprint(allowed_vocabulary: set[str] | None) -> str:
    """Return a stable digest used to validate training checkpoints."""
    entries = sorted(allowed_vocabulary or set())
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_fragment_library(class_manager: FragmentClassManager) -> dict[int, dict[str, Any]]:
    """Convert the class manager to the public fragment-library artifact."""
    fragment_library = {}
    for class_index, fragment_class in enumerate(class_manager.classes):
        symmetries = getattr(fragment_class, "candidate_symmetries_exemplar", [np.eye(3, dtype=float)])
        fragment_library[class_index] = {
            "smiles": fragment_class.smiles,
            "exemplar_z": torch.tensor(fragment_class.exemplar_z, dtype=torch.long),
            "exemplar_pos": torch.tensor(fragment_class.exemplar_pos, dtype=torch.float32),
            "exemplar_edge_index": torch.tensor(fragment_class.exemplar_edge_index, dtype=torch.long),
            "exemplar_edge_attr": torch.tensor(fragment_class.exemplar_edge_attr, dtype=torch.long),
            "symmetries": torch.tensor(np.asarray(symmetries), dtype=torch.float32),
        }
    return fragment_library


def _save_training_artifacts(
    *,
    output_path: str,
    class_manager: FragmentClassManager,
    train_split_key: str,
    train_input_count: int,
    allowed_vocabulary: set[str] | None,
    split_stats: dict[str, dict[str, int]],
    exclusions: list[dict[str, Any]],
    quarantined_failures: list[dict[str, Any]],
    train_shards: list[str],
) -> tuple[dict[int, dict[str, Any]], str, str]:
    """Persist the frozen training library before any held-out split is attempted."""
    previous_frozen = class_manager.frozen
    class_manager.frozen = False
    generate_orbit_from_exemplar_perms(class_manager)
    class_manager.frozen = previous_frozen

    output_dir = Path(output_path)
    fragment_library = _build_fragment_library(class_manager)
    library_path = _save_torch_atomic(fragment_library, output_dir / "fragment_library.pt")
    checkpoint = {
        "schema_version": "1.0",
        "train_split": train_split_key,
        "train_input_count": train_input_count,
        "allowed_vocabulary_sha256": _allowed_vocabulary_fingerprint(allowed_vocabulary),
        "class_manager": class_manager.state_dict(),
        "split_stats": split_stats[train_split_key],
        "exclusions": exclusions,
        "quarantined_failures": quarantined_failures,
        "train_shards": [Path(shard).name for shard in train_shards],
    }
    checkpoint_path = _save_torch_atomic(checkpoint, output_dir / "training_fragment_manager.pt")
    return fragment_library, library_path, checkpoint_path


def _load_training_checkpoint(
    *,
    checkpoint_path: Path,
    class_manager: FragmentClassManager,
    train_split_key: str,
    train_input_count: int,
    allowed_vocabulary: set[str] | None,
) -> dict[str, Any]:
    """Load and validate a completed training-stage fragmentation checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported training checkpoint schema: {checkpoint.get('schema_version')!r}.")
    if checkpoint.get("train_split") != train_split_key:
        raise ValueError(
            f"Training checkpoint split {checkpoint.get('train_split')!r} does not match {train_split_key!r}."
        )
    if int(checkpoint.get("train_input_count", -1)) != train_input_count:
        raise ValueError(
            "Training checkpoint input count does not match the currently loaded training split: "
            f"{checkpoint.get('train_input_count')} != {train_input_count}."
        )
    expected_fingerprint = _allowed_vocabulary_fingerprint(allowed_vocabulary)
    if checkpoint.get("allowed_vocabulary_sha256") != expected_fingerprint:
        raise ValueError("Training checkpoint was built with a different allowed structural vocabulary.")
    class_manager.load_state_dict(checkpoint["class_manager"])
    return checkpoint


def prepare_and_save_dataset(
    splits_dict: dict[str, list[Any]],
    train_split_key: str,
    class_manager: FragmentClassManager,
    output_path: str = ".",
    allowed_vocabulary: set | None = None,
    unknown_class_policy: str = "skip",
    dataset_name: str | None = None,
    validate_reconstruction: bool = True,
    validation_batch_size: int = 1024,
    max_failure_fraction: float = DEFAULT_MAX_FAILURE_FRACTION,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
    initial_failures: list[dict[str, Any]] | None = None,
    output_shard_size: int = 10_000,
    resume_training: bool = False,
):
    """Decompose splits, freezing training-derived classes for held-out data."""
    if unknown_class_policy not in {"error", "skip"}:
        raise ValueError("unknown_class_policy must be either 'error' or 'skip'.")
    max_failure_fraction = _validate_failure_fraction(max_failure_fraction)
    if max_automorphisms < 1:
        raise ValueError("max_automorphisms must be a positive integer.")
    if output_shard_size < 1:
        raise ValueError("output_shard_size must be a positive integer.")

    os.makedirs(output_path, exist_ok=True)
    print("Decomposing molecules...")
    record_buffers = {key: [] for key in splits_dict}
    processed_shards = {key: [] for key in splits_dict}
    shard_indices = {key: 0 for key in splits_dict}
    existing_shards: dict[str, list[Path]] = {}
    checkpoint_path = Path(output_path) / "training_fragment_manager.pt"

    for split_name in splits_dict:
        split_dir = Path(output_path) / split_name
        existing = sorted(split_dir.glob("shard_*.pt")) if split_dir.is_dir() else []
        existing_shards[split_name] = existing
        monolithic = Path(output_path) / f"{split_name}_processed.pt"
        if monolithic.exists():
            raise FileExistsError(
                f"Legacy monolithic output exists for split {split_name!r}. "
                "Use a new output directory to preserve artifact provenance."
            )
        if split_name != train_split_key and existing:
            raise FileExistsError(
                f"Partial held-out output exists for split {split_name!r}. Move that split directory aside "
                "before resuming; only the completed training stage is checkpointed."
            )

    training_resumed = bool(existing_shards[train_split_key])
    if training_resumed and not resume_training:
        raise FileExistsError(
            f"Processed training output already exists for split {train_split_key!r}. "
            "Set resume_training=true to reuse its completed training checkpoint."
        )
    if training_resumed and not checkpoint_path.exists():
        raise FileExistsError(
            "Training shards exist, but training_fragment_manager.pt is absent. This output predates safe "
            "checkpointing or was interrupted during training; move the output directory aside and rerun once."
        )
    if checkpoint_path.exists() and not training_resumed:
        raise FileExistsError(
            "training_fragment_manager.pt exists without training shards. Move the incomplete output directory "
            "aside and rerun to preserve artifact provenance."
        )

    split_stats = {
        name: {
            "input": len(dataset),
            "processed": 0,
            "excluded_small": 0,
            "excluded_linear": 0,
            "unknown_class": 0,
            "failed_vocabulary": 0,
            "failed_decomposition": 0,
            "fallback_molecules": 0,
            "fallback_events": 0,
        }
        for name, dataset in splits_dict.items()
    }
    exclusions = list(initial_failures or [])
    quarantined_failures = list(initial_failures or [])

    if training_resumed:
        checkpoint = _load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            class_manager=class_manager,
            train_split_key=train_split_key,
            train_input_count=len(splits_dict[train_split_key]),
            allowed_vocabulary=allowed_vocabulary,
        )
        checkpoint_shards = checkpoint.get("train_shards", [])
        actual_shards = [path.name for path in existing_shards[train_split_key]]
        if checkpoint_shards != actual_shards:
            raise ValueError("Training checkpoint shard list does not match the existing training artifacts.")
        processed_shards[train_split_key] = [str(path) for path in existing_shards[train_split_key]]
        shard_indices[train_split_key] = len(existing_shards[train_split_key])
        restored_stats = checkpoint["split_stats"]
        split_stats[train_split_key].update(restored_stats)
        exclusions = list(checkpoint["exclusions"])
        quarantined_failures = list(checkpoint["quarantined_failures"])
        print(f"Resumed frozen training library with {len(class_manager.classes)} classes from {checkpoint_path}.")

    vocabulary_failure_indices = {
        int(record["sample_index"])
        for record in quarantined_failures
        if record.get("stage") == "vocabulary" and "sample_index" in record
    }
    quarantine_path = write_failure_quarantine(quarantined_failures, output_path, max_failure_fraction)
    fragment_library: dict[int, dict[str, Any]] | None = None
    library_path = str(Path(output_path) / "fragment_library.pt")

    for split_name in _ordered_split_names(splits_dict, train_split_key):
        if split_name == train_split_key and training_resumed:
            continue

        dataset = splits_dict[split_name]
        print(f"\n  Processing {split_name}...")
        class_manager.frozen = split_name != train_split_key

        for idx, data in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc=f"Decomposing {split_name}"):
            if split_name == train_split_key and idx in vocabulary_failure_indices:
                split_stats[split_name]["failed_vocabulary"] += 1
                continue

            sample_id = str(getattr(data, "sample_id", idx))
            class_count_checkpoint = len(class_manager.classes)
            fallback_history: list[dict[str, str]] = []
            try:
                raw_pos = data.pos.detach().cpu().numpy() if hasattr(data.pos, "detach") else data.pos
                raw_z = data.h.detach().cpu().numpy() if hasattr(data.h, "detach") else np.asarray(data.h)

                heavy_mask = raw_z > 1
                heavy_pos = raw_pos[heavy_mask]

                if len(heavy_pos) < 3:
                    split_stats[split_name]["excluded_small"] += 1
                    exclusions.append(
                        {"split": split_name, "sample_id": sample_id, "reason": "fewer_than_3_heavy_atoms"}
                    )
                    continue

                centered_heavy_pos = heavy_pos - heavy_pos.mean(axis=0)
                singular_values = np.linalg.svd(centered_heavy_pos, compute_uv=False)
                if singular_values[1] < 0.05 * singular_values[0]:
                    split_stats[split_name]["excluded_linear"] += 1
                    exclusions.append({"split": split_name, "sample_id": sample_id, "reason": "globally_linear"})
                    continue

                if split_name == train_split_key:
                    frags = decompose_to_rigid_fragments(
                        data,
                        class_manager,
                        allowed_vocabulary,
                        max_automorphisms=max_automorphisms,
                    )
                else:
                    frags, fallback_history = decompose_with_frozen_fallback(
                        data,
                        class_manager,
                        allowed_vocabulary or set(),
                        max_automorphisms=max_automorphisms,
                    )
                compute_se3_targets(frags, class_manager)
                processed_record = None
                if frags:
                    processed_record = _processed_record(
                        data,
                        idx,
                        frags,
                        fallback_history,
                        dataset_name=dataset_name,
                    )
            except Exception as error:
                class_manager.rollback_to_class_count(class_count_checkpoint)
                is_unknown_class = isinstance(error, FrozenFragmentLibraryError)
                if is_unknown_class:
                    split_stats[split_name]["unknown_class"] += 1
                failure = {
                    "split": split_name,
                    **_sample_failure_record(data, idx, error, "decomposition"),
                }
                split_stats[split_name]["failed_decomposition"] += 1
                exclusions.append(failure)
                quarantined_failures.append(failure)

                if is_unknown_class and unknown_class_policy == "error":
                    quarantine_path = write_failure_quarantine(
                        quarantined_failures,
                        output_path,
                        max_failure_fraction,
                    )
                    raise RuntimeError(
                        f"Irreducible held-out fragment class in split '{split_name}', "
                        f"sample_id='{sample_id}'. See {quarantine_path}."
                    ) from error
                continue

            if fallback_history:
                split_stats[split_name]["fallback_molecules"] += 1
                split_stats[split_name]["fallback_events"] += len(fallback_history)
            if frags:
                assert processed_record is not None
                record_buffers[split_name].append(processed_record)
                split_stats[split_name]["processed"] += 1
                if len(record_buffers[split_name]) >= output_shard_size:
                    shard_path = _save_processed_shard(
                        record_buffers[split_name], output_path, split_name, shard_indices[split_name]
                    )
                    processed_shards[split_name].append(shard_path)
                    shard_indices[split_name] += 1
                    record_buffers[split_name] = []

        if record_buffers[split_name] or not processed_shards[split_name]:
            shard_path = _save_processed_shard(
                record_buffers[split_name], output_path, split_name, shard_indices[split_name]
            )
            processed_shards[split_name].append(shard_path)
            record_buffers[split_name] = []

        quarantine_path = write_failure_quarantine(quarantined_failures, output_path, max_failure_fraction)
        split_failures = split_stats[split_name]["failed_decomposition"]
        if dataset and split_failures / len(dataset) > max_failure_fraction:
            raise FragmentationFailureBudgetExceeded(
                stage="decomposition",
                split=split_name,
                failures=split_failures,
                total=len(dataset),
                max_failure_fraction=max_failure_fraction,
                quarantine_path=quarantine_path,
            )

        if split_name == train_split_key:
            fragment_library, library_path, checkpoint_path_string = _save_training_artifacts(
                output_path=output_path,
                class_manager=class_manager,
                train_split_key=train_split_key,
                train_input_count=len(dataset),
                allowed_vocabulary=allowed_vocabulary,
                split_stats=split_stats,
                exclusions=exclusions,
                quarantined_failures=quarantined_failures,
                train_shards=processed_shards[train_split_key],
            )
            print(
                f"Saved the frozen training library before held-out processing: {library_path} "
                f"and {checkpoint_path_string}."
            )

    total_small = sum(stats["excluded_small"] for stats in split_stats.values())
    total_linear = sum(stats["excluded_linear"] for stats in split_stats.values())
    total_unknown = sum(stats["unknown_class"] for stats in split_stats.values())
    total_vocabulary_failed = sum(stats["failed_vocabulary"] for stats in split_stats.values())
    total_decomposition_failed = sum(stats["failed_decomposition"] for stats in split_stats.values())
    total_fallback_molecules = sum(stats["fallback_molecules"] for stats in split_stats.values())
    total_fallback_events = sum(stats["fallback_events"] for stats in split_stats.values())
    print(
        f"Skipped: {total_small} small, {total_linear} globally linear, "
        f"{total_unknown} irreducible held-out, {total_vocabulary_failed} vocabulary failures, "
        f"{total_decomposition_failed} decomposition failures. "
        f"Refined {total_fallback_molecules} held-out molecules in {total_fallback_events} fallback steps."
    )

    quarantine_path = write_failure_quarantine(quarantined_failures, output_path, max_failure_fraction)

    if fragment_library is None:
        fragment_library = _build_fragment_library(class_manager)
    _save_torch_atomic(fragment_library, Path(library_path))
    checkpoint_path_string = str(checkpoint_path)

    print("\nProcessed splits were saved in bounded-memory shards.")

    validation_path = None
    reconstruction_metrics = {}
    if validate_reconstruction:
        if dataset_name not in {"qm9", "geom", "qmugs"}:
            raise ValueError("dataset_name is required for reconstruction validation.")
        print("\nValidating ground-truth reconstruction...")
        from motiflow.data.loaders import ShardedDataSequence

        for split_name, shard_paths in processed_shards.items():
            if not shard_paths:
                continue
            records = ShardedDataSequence(shard_paths)
            metrics = evaluate_ground_truth_fragments(
                records,
                fragment_library,
                dataset=dataset_name,
                batch_size=validation_batch_size,
            )
            reconstruction_metrics[split_name] = metrics
            print(
                f"  {split_name}: atom stability={100 * metrics['atom_stability']:.2f}%, "
                f"molecule stability={100 * metrics['molecule_stability']:.2f}%, "
                f"validity={100 * metrics['validity']:.2f}%, "
                f"validity x uniqueness={100 * metrics['validity_times_uniqueness']:.2f}%"
            )
        validation_path = os.path.join(output_path, "reconstruction_metrics.json")
        write_json_atomic(reconstruction_metrics, validation_path)

    print("Done.")
    return {
        "counts": {
            "vocabulary_classes": len(fragment_library),
            "splits": split_stats,
            "training_resumed": training_resumed,
        },
        "exclusions": exclusions,
        "reconstruction_metrics": reconstruction_metrics,
        "output_files": [
            library_path,
            checkpoint_path_string,
            *[path for split_paths in processed_shards.values() for path in split_paths],
            *[
                str(Path(path).with_suffix(".manifest.json"))
                for split_paths in processed_shards.values()
                for path in split_paths
            ],
            quarantine_path,
            *([validation_path] if validation_path else []),
        ],
    }
