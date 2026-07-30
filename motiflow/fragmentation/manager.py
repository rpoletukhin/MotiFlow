from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .errors import FrozenFragmentLibraryError
from .math_utils import is_geometrically_degenerate, robust_kabsch_symmetry


class FragmentClass:
    """
    Stores exemplar geometry and automorphism perms (permutations of exemplar indices).
    exemplar_pos: numpy (N,3)
    exemplar_z: numpy (N,)
    perms: list of numpy arrays shape (N,) each representing a permutation of indices 0..N-1
    """

    def __init__(
        self,
        smiles: str,
        exemplar_pos: np.ndarray,
        exemplar_z: np.ndarray,
        exemplar_edge_index: np.ndarray,
        exemplar_edge_attr: np.ndarray,
        perms: List[List[int]],
    ):
        self.smiles = smiles
        self.exemplar_pos = exemplar_pos.copy()
        self.exemplar_z = exemplar_z.copy()
        self.exemplar_edge_index = exemplar_edge_index.copy()
        self.exemplar_edge_attr = exemplar_edge_attr.copy()
        # Perms must be permutations of range(N) with respect to the real-only canonical ordering
        self.perms = [np.array(p, dtype=int) for p in perms]
        # Defensive sanity
        if self.exemplar_pos.shape[0] != self.exemplar_z.shape[0]:
            raise ValueError(
                "FragmentClass exemplar size mismatch: "
                f"pos rows {self.exemplar_pos.shape[0]} vs z length {self.exemplar_z.shape[0]}"
            )
        # Validate perms shapes
        N = self.exemplar_pos.shape[0]
        for p in self.perms:
            if p.shape[0] != N:
                raise ValueError(f"FragmentClass permutation length mismatch: expected {N}, got {p.shape[0]}")


class FragmentClassManager:
    def __init__(
        self,
        rmsd_thresh: float = 1.0,
        max_dev_thresh: float = 1.0,
        heavy_weight: float = 1.0,
        h_weight: float = 0.1,
        stub_weight: float = 0.1,
    ):
        self.classes: List[FragmentClass] = []
        self.by_smiles: Dict[str, List[int]] = {}
        self.rmsd_thresh = float(rmsd_thresh)
        self.max_dev_thresh = float(max_dev_thresh)
        self.heavy_weight = float(heavy_weight)
        self.h_weight = float(h_weight)
        self.stub_weight = float(stub_weight)
        self.frozen = False

    def state_dict(self) -> dict:
        """Serialize the complete training-derived class manager state."""
        classes = []
        for fragment_class in self.classes:
            classes.append(
                {
                    "smiles": fragment_class.smiles,
                    "exemplar_pos": fragment_class.exemplar_pos,
                    "exemplar_z": fragment_class.exemplar_z,
                    "exemplar_edge_index": fragment_class.exemplar_edge_index,
                    "exemplar_edge_attr": fragment_class.exemplar_edge_attr,
                    "perms": [permutation.tolist() for permutation in fragment_class.perms],
                    "candidate_symmetries_exemplar": list(getattr(fragment_class, "candidate_symmetries_exemplar", [])),
                }
            )
        return {
            "schema_version": "1.0",
            "settings": {
                "rmsd_thresh": self.rmsd_thresh,
                "max_dev_thresh": self.max_dev_thresh,
                "heavy_weight": self.heavy_weight,
                "h_weight": self.h_weight,
                "stub_weight": self.stub_weight,
            },
            "classes": classes,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a manager checkpoint, rejecting incompatible class settings."""
        if state.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported fragment-manager schema: {state.get('schema_version')!r}.")
        settings = state["settings"]
        for name in ("rmsd_thresh", "max_dev_thresh", "heavy_weight", "h_weight", "stub_weight"):
            configured = float(getattr(self, name))
            checkpointed = float(settings[name])
            if not np.isclose(configured, checkpointed):
                raise ValueError(
                    f"Fragment-manager setting {name}={configured} does not match checkpoint value {checkpointed}."
                )

        self.classes = []
        self.by_smiles = {}
        for class_state in state["classes"]:
            fragment_class = FragmentClass(
                smiles=class_state["smiles"],
                exemplar_pos=np.asarray(class_state["exemplar_pos"]),
                exemplar_z=np.asarray(class_state["exemplar_z"]),
                exemplar_edge_index=np.asarray(class_state["exemplar_edge_index"]),
                exemplar_edge_attr=np.asarray(class_state["exemplar_edge_attr"]),
                perms=class_state["perms"],
            )
            symmetries = class_state.get("candidate_symmetries_exemplar", [])
            if symmetries:
                fragment_class.candidate_symmetries_exemplar = [np.asarray(rotation) for rotation in symmetries]
            class_index = len(self.classes)
            self.classes.append(fragment_class)
            self.by_smiles.setdefault(fragment_class.smiles, []).append(class_index)

    def rollback_to_class_count(self, class_count: int) -> None:
        """Remove classes created after a failed molecule-level transaction."""
        if not 0 <= class_count <= len(self.classes):
            raise ValueError(f"class_count must be between 0 and {len(self.classes)}, got {class_count}.")
        if class_count == len(self.classes):
            return
        del self.classes[class_count:]
        self.by_smiles = {}
        for class_idx, fragment_class in enumerate(self.classes):
            self.by_smiles.setdefault(fragment_class.smiles, []).append(class_idx)

    def _atom_weights_from_z(self, z: np.ndarray) -> np.ndarray:
        """Weights for Orientation/Rotation (Includes Stubs)"""
        w = np.ones_like(z, dtype=float)
        w[z == 1] = self.h_weight  # Hydrogens
        w[(z != 1) & (z != 0)] = self.heavy_weight  # Heavy Atoms
        w[z == 0] = self.stub_weight  # Stubs (Ghost Heavy Atoms)
        return w

    def _weights_for_centering(self, z: np.ndarray) -> np.ndarray:
        """Weights for Translation/Centroid (Real Atoms Only)"""
        w = np.zeros_like(z, dtype=float)
        w[z == 1] = self.h_weight
        w[(z != 1) & (z != 0)] = self.heavy_weight
        w[z == 0] = 0.0  # Stubs excluded from physical center of mass
        return w

    def _select_orientation_indices(self, z: np.ndarray) -> np.ndarray:
        """
        Use ALL indices; let weights suppress noise. Stubs are included for orientation.
        """
        return np.arange(len(z), dtype=int)

    def _compute_canonical_rotation(
        self, cls: FragmentClass, pos: np.ndarray, z: np.ndarray, perm: np.ndarray, weights_center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute canonical rotation and translation mapping exemplar -> instance.
        Uses the exemplar ordering (cls.exemplar_pos) and the provided perm to reorder
        exemplar points to match `pos` ordering, then computes an SE(3) mapping.
        We use rotation weights that include stubs for orientation and weights_center (real atoms only)
        for translation centering to be consistent with how exemplars are stored.
        """
        N = pos.shape[0]
        Q_reordered = np.zeros_like(pos)
        w_center_reordered = np.zeros_like(weights_center)
        for k in range(N):
            target_idx = perm[k]
            Q_reordered[target_idx] = pos[k]
            w_center_reordered[target_idx] = weights_center[k]

        weights_rotation = self._atom_weights_from_z(cls.exemplar_z)

        # Robust Kabsch / orientation using rotation weights (all points matter)
        R_canon, _, _, _, _ = robust_kabsch_symmetry(cls.exemplar_pos, Q_reordered, weights=weights_rotation)

        # Translation: compute weighted centroids using weights_center (real atoms only)
        if np.sum(w_center_reordered) > 1e-12:
            cQ = np.average(Q_reordered, axis=0, weights=w_center_reordered)
            cP = np.average(cls.exemplar_pos, axis=0, weights=w_center_reordered)
        else:
            cQ = np.mean(Q_reordered, axis=0)
            cP = np.mean(cls.exemplar_pos, axis=0)

        t_canon = cQ - (R_canon @ cP)
        return R_canon, t_canon

    def _compute_inst_to_canonical(
        self, pos: np.ndarray, z: np.ndarray, weights_center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a deterministic canonical exemplar frame for a fragment instance.

        CRITICAL:
         - The exemplar returned is centered so that the weighted centroid (weights_center,
           i.e. real-atoms-only centroid) is at the origin. This ensures translation
           computed later (cQ - R @ cP) is consistent and small.
         - Orientation (rotation) is found via weighted PCA using rotation-weights (which include stubs).
        Returns:
            pos_centered : pos - centroid (so positional information is relative to that centroid)
            R_inst_to_canonical : rotation matrix such that exemplar_pos = pos_centered @ R_inst_to_canonical
        """
        if pos.shape[0] == 0:
            raise ValueError("Empty fragment")

        # Compute center using weights_center (real atoms only) so exemplar has physical centroid at origin.
        wsum = float(np.sum(weights_center))
        if wsum < 1e-12:
            center = np.mean(pos, axis=0)
        else:
            center = np.average(pos, axis=0, weights=weights_center)

        # Subtract this center --> this ensures exemplar weighted centroid (real atoms) will be zero after rotation.
        pos_centered = pos - center

        # Rotation: use rotation-weights (stubs included) to define principal axes.
        weights_rotation = self._atom_weights_from_z(z)
        # If too small, raise error since we are supposed to catch this earlier.
        if pos.shape[0] < 3 or np.sum(weights_rotation) <= 0:
            raise ValueError("Insufficient points/weights for PCA frame definition")

        # Weighted PCA for frame definition (on pos_centered)
        # Compute weighted mean for rotation-weights (note: pos_centered may have small non-zero mean for these weights)
        wrot_sum = np.sum(weights_rotation)
        rot_mean = np.average(pos_centered, axis=0, weights=weights_rotation)
        Pm = pos_centered - rot_mean

        cov = (Pm.T @ (Pm * weights_rotation[:, None])) / (wrot_sum + 1e-20)
        _, evecs = np.linalg.eigh(cov)
        # Build rotation matrix from principal axes: largest eigenvector first
        # We want R such that we rotate instance points into canonical axes
        R_pca = np.column_stack([evecs[:, 2], evecs[:, 1], evecs[:, 0]]).T

        # Enforce right-handedness and sensible sign choice
        pos_rot = pos_centered @ R_pca.T
        for i in range(3):
            skew = np.sum(pos_rot[:, i] ** 3)
            if skew < -1e-9:
                R_pca[i, :] *= -1.0
        if np.linalg.det(R_pca) < 0:
            R_pca[2, :] *= -1.0

        # We want to return R_inst_to_canonical such that exemplar_pos = pos_centered @ R_inst_to_canonical
        R_inst_to_canonical = R_pca.T
        return pos_centered, R_inst_to_canonical

    def _matches_fragment(
        self, cls: FragmentClass, pos: np.ndarray, z: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        Try to match `pos` (instance) to class exemplar `cls`.
        Uses orientation weights (including stubs) for alignment and weights_center for RMSD checks.
        """
        if pos.shape[0] != cls.exemplar_pos.shape[0]:
            return None
        if not np.array_equal(z, cls.exemplar_z):
            return None

        weights_align = self._atom_weights_from_z(z)
        weights_center = self._weights_for_centering(z)

        best_selection_score = float("inf")
        best_result = None

        for perm in cls.perms:
            P_ex_perm = cls.exemplar_pos[perm]

            # Align using rotation weights (stubs included)
            R_sub, _, _, _, reflected = robust_kabsch_symmetry(P_ex_perm, pos, weights=weights_align)

            # Compute centroids for translation using weights_center (real atoms)
            if np.sum(weights_center) > 1e-12:
                cQ = np.average(pos, axis=0, weights=weights_center)
                cP = np.average(P_ex_perm, axis=0, weights=weights_center)
            else:
                cQ = np.mean(pos, axis=0)
                cP = np.mean(P_ex_perm, axis=0)
            t_sub = cQ - (R_sub @ cP)

            P_pred_full = (R_sub @ P_ex_perm.T).T + t_sub
            per_atom_diff_sq = np.sum((P_pred_full - pos) ** 2, axis=1)
            per_atom_full = np.sqrt(per_atom_diff_sq)

            selection_mse = np.average(per_atom_diff_sq, weights=weights_align)

            if np.sum(weights_center) > 1e-12:
                real_mse = np.average(per_atom_diff_sq, weights=weights_center)
            else:
                real_mse = np.mean(per_atom_diff_sq)
            real_rmsd = float(np.sqrt(real_mse))

            geom_type = is_geometrically_degenerate(cls.exemplar_pos)
            allow_reflection = geom_type != "nondegenerate"

            if reflected and not allow_reflection:
                continue

            if selection_mse < best_selection_score:
                best_selection_score = selection_mse
                if np.any(z > 0):
                    max_dev = float(np.max(per_atom_full[z > 0]))
                else:
                    max_dev = float(np.max(per_atom_full))
                best_result = (perm, real_rmsd, per_atom_full, max_dev)

        if best_result is not None:
            perm, real_rmsd, per_atom_full, max_dev = best_result
            if real_rmsd <= self.rmsd_thresh and max_dev <= self.max_dev_thresh:
                R_canon, t_canon = self._compute_canonical_rotation(cls, pos, z, perm, weights_center)
                return R_canon, t_canon, real_rmsd, per_atom_full

        return None

    def find_or_create_class(
        self,
        smiles: str,
        pos: np.ndarray,
        z: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        perms_from_canon: List[List[int]],
    ) -> Tuple[int, FragmentClass, Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]]:
        """
        Creates or finds a class for the fragment with canonical exemplar creation.
        The exemplar will be stored such that the weighted centroid (weights_center) is at origin.
        """
        if not perms_from_canon:
            perms = [list(range(pos.shape[0]))]
        else:
            perms = perms_from_canon

        existing_indices = self.by_smiles.get(smiles, [])

        for cls_idx in existing_indices:
            cls = self.classes[cls_idx]
            match = self._matches_fragment(cls, pos, z)
            if match is not None:
                return cls_idx, cls, match

        if self.frozen:
            failure_kind = "geometry_mismatch" if existing_indices else "unseen_structure"
            raise FrozenFragmentLibraryError(
                smiles=smiles,
                failure_kind=failure_kind,
                candidate_class_count=len(existing_indices),
            )

        # Create new class: compute exemplar in canonical frame
        weights_center = self._weights_for_centering(z)
        pos_centered, R_inst_to_canonical = self._compute_inst_to_canonical(pos, z, weights_center)
        exemplar_pos = (pos_centered @ R_inst_to_canonical).astype(np.float64)

        new_cls = FragmentClass(
            smiles=smiles,
            exemplar_pos=exemplar_pos,
            exemplar_z=z,
            exemplar_edge_index=edge_index,
            exemplar_edge_attr=edge_attr,
            perms=perms,
        )
        new_idx = len(self.classes)
        self.classes.append(new_cls)
        self.by_smiles.setdefault(smiles, []).append(new_idx)

        return new_idx, new_cls, None


class FragmentData:
    def __init__(self, pos: torch.Tensor, z: torch.Tensor, smiles: str):
        """
        pos: torch (N,3) canonical-order positions for real atoms (float32)
        z:   torch (N,) atomic numbers (int64)
        smiles: canonical fragment smiles string (real-only, no '*' dummies)
        """
        self.pos = pos
        self.z = z
        self.smiles = smiles
        self.rot_gt: Optional[torch.Tensor] = None
        self.trans_gt: Optional[torch.Tensor] = None
        self.class_id: Optional[int] = None
        self.original_indices: List[int] = []
