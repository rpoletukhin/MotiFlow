import math
from typing import Optional, Tuple

import numpy as np
import torch

EPS_SMALL = 1e-3


def is_geometrically_degenerate(P: np.ndarray, tol_rel: float = 0.05) -> str:
    """
    Inspect centered coordinates P (N,3) and decide if the point set is:
      - 'linear'    : all points lie approximately on a line,
      - 'planar'    : points lie approximately in a plane,
      - 'nondegenerate' : full 3D.
    Use SVD singular values relative tolerance tol_rel.
    Returns one of the three strings.
    """
    if P.shape[0] < 3:
        # 1 or 2 points: linear degeneration
        return "linear"
    P_c = P - P.mean(axis=0)
    _, S, _ = np.linalg.svd(P_c, full_matrices=False)
    if S[1] < tol_rel * S[0]:
        # Second singular value tiny -> linear
        return "linear"
    if S[2] < tol_rel * S[0]:
        # Third singular value tiny -> planar
        return "planar"
    return "nondegenerate"


def numpy_to_torch_R_t(R_np: np.ndarray, t_np: np.ndarray, device: Optional[torch.device] = None):
    """
    Convert numpy R, t to torch tensors (float32).
    """
    R_t = torch.tensor(R_np.astype(np.float32), dtype=torch.float32, device=device)
    t_t = torch.tensor(t_np.astype(np.float32), dtype=torch.float32, device=device)
    return R_t, t_t


def rotation_angle_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Geodesic distance between two rotation matrices in radians.
    """
    R = R1.T @ R2
    tr = (np.trace(R) - 1.0) / 2.0
    tr = max(-1.0, min(1.0, tr))
    theta = math.acos(tr)
    return float(theta)


def numpy_kabsch_with_reflection_flag(
    P: np.ndarray, Q: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """
    Weighted Kabsch: P -> Q. Returns (R, t, weighted_mse, per_atom_errors, reflected_flag).
    Same semantics as before.
    """
    if P.shape != Q.shape:
        raise ValueError("P and Q must have same shape for Kabsch")
    N = P.shape[0]
    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = weights.astype(float).copy()
        if w.shape[0] != N:
            raise ValueError("weights length mismatch in kabsch")
    wsum = w.sum()
    if wsum == 0:
        raise ValueError("sum of weights is zero in kabsch")
    w = w / wsum

    if N < 2:
        cP = (w[:, None] * P).sum(axis=0)
        cQ = (w[:, None] * Q).sum(axis=0)
        R = np.eye(3, dtype=float)
        t = cQ - R @ cP
        P_pred = (R @ P.T).T + t
        per_atom = np.linalg.norm(P_pred - Q, axis=1)
        weighted_mse = (w * (per_atom**2)).sum()
        return R, t, float(weighted_mse), per_atom, False

    cP = (w[:, None] * P).sum(axis=0)
    cQ = (w[:, None] * Q).sum(axis=0)
    P_c = P - cP
    Q_c = Q - cQ

    H = (P_c * w[:, None]).T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R_raw = V @ U.T

    reflected = False
    if np.linalg.det(R_raw) < 0:
        reflected = True
        V[:, -1] *= -1
        R = V @ U.T
    else:
        R = R_raw

    t = cQ - R @ cP

    P_pred = (R @ P.T).T + t
    diffs = P_pred - Q
    per_atom_sq = np.sum(diffs * diffs, axis=1)
    per_atom = np.sqrt(per_atom_sq)
    weighted_mse = (w * per_atom_sq).sum()

    return R, t, float(weighted_mse), per_atom, reflected


def robust_kabsch_symmetry(
    P: np.ndarray, Q: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    """
    Returns (R, t, weighted_mse, per_atom_errors, reflected_flag).
    Calculates the best SE(3) rotation mapping P -> Q.
    Strictly assumes N >= 3 due to pipeline filtering and Deep Stub injection.
    """
    N = P.shape[0]

    if N < 3:
        raise ValueError(f"robust_kabsch_symmetry called with {N} points. Pipeline guarantees >= 3.")

    R, t, mse, per_atom, reflected = numpy_kabsch_with_reflection_flag(P, Q, weights)

    # Only snap to identity if the MSE is numerically zero AND the matrix is exactly identity
    if mse < EPS_SMALL:
        if np.allclose(R, np.eye(3), atol=1e-4):
            R = np.eye(3)
            t = np.mean(Q, axis=0) - np.mean(P, axis=0)
            per_atom = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
            return R, t, float((weights * (per_atom**2)).sum() / weights.sum()), per_atom, False

    return R, t, mse, per_atom, reflected
