import collections
from typing import Any, Hashable, List, Optional

import numpy as np
import torch
import tqdm
from rdkit import Chem

from .errors import (
    AutomorphismEnumerationLimitError,
    FragmentationFailureBudgetExceeded,
    FragmentationSampleError,
    FrozenFragmentLibraryError,
)
from .manager import FragmentData
from .math_utils import is_geometrically_degenerate

DEFAULT_MAX_AUTOMORPHISMS = 4096
DEFAULT_MAX_FAILURE_FRACTION = 0.01

_AUTOMORPHISM_CACHE: dict[Hashable, tuple[tuple[int, ...], ...]] = {}
_AUTOMORPHISM_OVERFLOW_CAPS: dict[Hashable, int] = {}


def _filter_perms_preserve_atomic_numbers(perms: List[List[int]], exemplar_z: np.ndarray) -> List[List[int]]:
    """Keep only permutations p such that exemplar_z[p] == exemplar_z (element-wise).
    This protects against RDKit returning automorphisms that swap different elements."""
    out = []
    for p in perms:
        p_arr = np.asarray(p, dtype=int)
        if p_arr.shape[0] != exemplar_z.shape[0]:
            continue
        if np.array_equal(exemplar_z[p_arr], exemplar_z):
            out.append(p_arr.tolist())
    return out


def _clean_copy_for_automorphisms(mol: Chem.Mol) -> Chem.Mol:
    mcopy = Chem.RWMol(mol)
    for a in mcopy.GetAtoms():
        try:
            a.SetIsAromatic(False)
        except Exception:
            pass
        try:
            a.SetFormalCharge(0)
        except Exception:
            pass
        try:
            a.SetChiralTag(Chem.CHI_UNSPECIFIED)
        except Exception:
            pass
    for b in mcopy.GetBonds():
        try:
            b.SetIsAromatic(False)
        except Exception:
            pass
        try:
            b.SetStereo(Chem.BondStereo.STEREONONE)
        except Exception:
            pass
    return mcopy.GetMol()


def _canonical_topology(mol: Chem.Mol) -> tuple[Hashable, list[int]]:
    """Return a topology key and the canonical-index-to-input-index mapping."""
    ranks = list(
        Chem.CanonicalRankAtoms(
            mol,
            breakTies=True,
            includeChirality=False,
            includeIsotopes=False,
        )
    )
    canonical_to_input = sorted(range(mol.GetNumAtoms()), key=ranks.__getitem__)
    canonical_mol = Chem.RenumberAtoms(mol, canonical_to_input)
    atom_key = tuple(atom.GetAtomicNum() for atom in canonical_mol.GetAtoms())
    bond_key = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                str(bond.GetBondType()),
            )
            for bond in canonical_mol.GetBonds()
        )
    )
    return (atom_key, bond_key), canonical_to_input


def _topology_key_for_message(topology_key: Hashable) -> str:
    """Create a compact deterministic topology identifier for error records."""
    atom_key, bond_key = topology_key
    atoms = ",".join(str(value) for value in atom_key)
    bonds = ";".join(f"{u}-{v}:{bond_type}" for u, v, bond_type in bond_key)
    return f"atoms={atoms}|bonds={bonds}"


def _get_graph_automorphisms_preserve_element(
    mol: Chem.Mol,
    exemplar_z: Optional[np.ndarray] = None,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
) -> List[List[int]]:
    if max_automorphisms < 1:
        raise ValueError("max_automorphisms must be a positive integer.")

    mc = _clean_copy_for_automorphisms(mol)
    topology_key, canonical_to_input = _canonical_topology(mc)
    overflow_cap = _AUTOMORPHISM_OVERFLOW_CAPS.get(topology_key, 0)
    if max_automorphisms <= overflow_cap:
        raise AutomorphismEnumerationLimitError(
            topology_key=_topology_key_for_message(topology_key),
            num_atoms=mc.GetNumAtoms(),
            max_automorphisms=max_automorphisms,
        )

    canonical_perms = _AUTOMORPHISM_CACHE.get(topology_key)
    if canonical_perms is None:
        canonical_mol = Chem.RenumberAtoms(mc, canonical_to_input)
        raw = canonical_mol.GetSubstructMatches(
            canonical_mol,
            uniquify=False,
            useChirality=False,
            maxMatches=max_automorphisms + 1,
        )
        if len(raw) > max_automorphisms:
            _AUTOMORPHISM_OVERFLOW_CAPS[topology_key] = max(max_automorphisms, overflow_cap)
            raise AutomorphismEnumerationLimitError(
                topology_key=_topology_key_for_message(topology_key),
                num_atoms=mc.GetNumAtoms(),
                max_automorphisms=max_automorphisms,
            )
        canonical_perms = tuple(tuple(match) for match in raw)
        _AUTOMORPHISM_CACHE[topology_key] = canonical_perms
    elif len(canonical_perms) > max_automorphisms:
        raise AutomorphismEnumerationLimitError(
            topology_key=_topology_key_for_message(topology_key),
            num_atoms=mc.GetNumAtoms(),
            max_automorphisms=max_automorphisms,
        )

    input_to_canonical = [0] * len(canonical_to_input)
    for canonical_idx, input_idx in enumerate(canonical_to_input):
        input_to_canonical[input_idx] = canonical_idx
    perms = [
        [canonical_to_input[canonical_perm[input_to_canonical[input_idx]]] for input_idx in range(mc.GetNumAtoms())]
        for canonical_perm in canonical_perms
    ]
    if exemplar_z is not None:
        perms = _filter_perms_preserve_atomic_numbers(perms, exemplar_z)
    return perms


def get_full_rigid_fragment_atoms(mol: Chem.Mol, base_atoms: set) -> set:
    """
    Expands base ring atoms to include attached Hydrogens and
    atoms connected via rigid double/triple bonds, exactly mimicking the cutting logic.
    """
    full_atoms = set(base_atoms)
    added = True
    while added:
        added = False
        for idx in list(full_atoms):
            atom = mol.GetAtomWithIdx(idx)
            for b in atom.GetBonds():
                nbr = b.GetOtherAtom(atom)
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in full_atoms:
                    # Keep if Hydrogen OR if connected by a non-single bond
                    if nbr.GetAtomicNum() == 1 or b.GetBondType() != Chem.BondType.SINGLE:
                        full_atoms.add(nbr_idx)
                        added = True
    return full_atoms


def _generate_candidate_fragment_smiles(mol: Chem.Mol, system_atom_indices: set) -> str:
    """
    Generates a canonical SMILES for the system as an induced subgraph.
    It includes the system atoms and any originally attached Hydrogens.
    It does NOT add stubs or explicit Hydrogen caps at cut sites;
    cut atoms will appear with their implicit valence removed, matching
    the raw fragment library format.
    """
    # Identify atoms to keep: System Ring Atoms + Attached Hydrogens
    atoms_to_keep = set(system_atom_indices)

    for idx in system_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            nbr_idx = nbr.GetIdx()

            # If the neighbor is part of the system, it's already kept.
            if nbr_idx in system_atom_indices:
                continue

            # If the neighbor is an explicit Hydrogen, keep it.
            # (Matches the logic that the fragment includes its Hs)
            if nbr.GetAtomicNum() == 1:
                atoms_to_keep.add(nbr_idx)

            # If the neighbor is a heavy atom outside the system:
            # We do NOTHING. We do not add a stub (*), and we do NOT cap with H.
            # The bond is simply ignored, leaving the internal atom with 'open' valence.

    # Build the temporary RWMol
    label_mol = Chem.RWMol()
    node_map = {}  # orig_idx -> new_idx

    # Add Real Atoms (System + Existing Hs)
    for orig_idx in sorted(list(atoms_to_keep)):
        src_atom = mol.GetAtomWithIdx(orig_idx)
        new_atom = Chem.Atom(src_atom.GetAtomicNum())

        # Critical: SetNoImplicit(True) ensures RDKit doesn't guess Hs
        # For the atoms where we cut bonds. They will appear as-is.
        new_atom.SetNoImplicit(True)

        try:
            new_atom.SetFormalCharge(src_atom.GetFormalCharge())
        except RuntimeError:
            pass
        try:
            new_atom.SetChiralTag(src_atom.GetChiralTag())
        except RuntimeError:
            pass

        # Force non-aromatic to ensure SMILES consistency with input Kekulization
        new_atom.SetIsAromatic(False)

        idx = label_mol.AddAtom(new_atom)
        node_map[orig_idx] = idx

    # Add internal bonds only.
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u in atoms_to_keep and v in atoms_to_keep:
            label_mol.AddBond(node_map[u], node_map[v], bond.GetBondType())

    # Generate SMILES
    # AllHsExplicit=True ensures the H atoms we added are written as nodes (e.g. [H][C])
    return Chem.MolToSmiles(label_mol, isomericSmiles=True, canonical=True, allHsExplicit=True)


def _selected_hierarchy_motifs(hierarchy, rare_motifs):
    """Resolve ring-system -> ring candidates for one molecule."""
    selected = []
    for system in hierarchy:
        if system["level_0"] and system["level_0"] not in rare_motifs:
            selected.append(system["level_0"])
        else:
            selected.extend(ring_smiles for ring_smiles in system["level_1"] if ring_smiles not in rare_motifs)
    return selected


def _count_hierarchy_motifs(mol_hierarchies, rare_motifs):
    """Count selected motif occurrences across the training conformers."""
    counts = collections.Counter()
    for hierarchy in mol_hierarchies:
        counts.update(_selected_hierarchy_motifs(hierarchy, rare_motifs))
    return counts


def _ring_systems_from_atom_rings(atom_rings) -> list[set[int]]:
    """Find connected ring systems in near-linear time using shared atoms."""
    num_rings = len(atom_rings)
    parents = list(range(num_rings))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_ring_by_atom = {}
    for ring_idx, atoms in enumerate(atom_rings):
        for atom_idx in atoms:
            previous_ring = first_ring_by_atom.setdefault(atom_idx, ring_idx)
            union(previous_ring, ring_idx)

    components = {}
    for ring_idx in range(num_rings):
        components.setdefault(find(ring_idx), set()).add(ring_idx)
    return list(components.values())


def _validate_failure_fraction(max_failure_fraction: float) -> float:
    value = float(max_failure_fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("max_failure_fraction must be between 0 and 1, inclusive.")
    return value


def _sample_failure_record(data: Any, index: int, error: Exception, stage: str) -> dict[str, Any]:
    sample_id = str(getattr(data, "sample_id", index))
    if isinstance(error, FragmentationSampleError):
        record = error.as_record()
    else:
        record = {
            "reason": "fragmentation_failed",
            "message": str(error),
        }
    return {
        "stage": stage,
        "sample_id": sample_id,
        "sample_index": index,
        "exception_type": type(error).__name__,
        **record,
    }


def build_recursive_vocabulary(
    dataset,
    threshold_alpha: float,
    *,
    max_failure_fraction: float = DEFAULT_MAX_FAILURE_FRACTION,
    failure_records: Optional[list[dict[str, Any]]] = None,
):
    max_failure_fraction = _validate_failure_fraction(max_failure_fraction)
    total_mols = len(dataset)
    min_count = max(1, int(total_mols * (threshold_alpha / 100.0)))

    mol_hierarchies = []
    failures = failure_records if failure_records is not None else []
    initial_failure_count = len(failures)

    print(f"Mining hierarchies to build vocabulary (threshold={min_count} occurrences)...")
    for sample_index, data in enumerate(tqdm.tqdm(dataset, desc="Extracting Train Hierarchies")):
        try:
            # Apply the same global filters as the main pipeline.
            raw_pos = data.pos.detach().cpu().numpy() if hasattr(data.pos, "detach") else data.pos
            raw_z = data.h.detach().cpu().numpy() if hasattr(data.h, "detach") else np.asarray(data.h)
            heavy_mask = raw_z > 1
            heavy_pos = raw_pos[heavy_mask]

            if len(heavy_pos) < 3:
                continue

            centered_heavy_pos = heavy_pos - heavy_pos.mean(axis=0)
            singular_values = np.linalg.svd(centered_heavy_pos, compute_uv=False)
            if singular_values[1] < 0.05 * singular_values[0]:
                continue

            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                raise FragmentationSampleError(
                    "parse_failed",
                    "RDKit could not parse the molecule SMILES.",
                    smiles=data.smiles,
                )
            mol = Chem.AddHs(mol)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception as error:
                raise FragmentationSampleError(
                    "kekulization_failed",
                    "RDKit could not kekulize the parsed molecule.",
                    smiles=data.smiles,
                ) from error
            conf_pos = data.pos.detach().cpu().numpy()

            atom_rings = mol.GetRingInfo().AtomRings()
            ring_systems = _ring_systems_from_atom_rings(atom_rings)

            def is_planar(indices):
                if len(indices) < 3:
                    return True
                coords = conf_pos[list(indices)]
                coords -= coords.mean(axis=0)
                _, singular_values, _ = np.linalg.svd(coords, full_matrices=False)
                return singular_values[2] < 0.05 * singular_values[0]

            hierarchy = []
            for sys_rings in ring_systems:
                sys_atoms = set()
                for ring_idx in sys_rings:
                    sys_atoms.update(atom_rings[ring_idx])

                full_sys_atoms = get_full_rigid_fragment_atoms(mol, sys_atoms)

                sys_info = {"level_0": None, "level_1": []}
                if is_planar(full_sys_atoms):
                    sys_info["level_0"] = _generate_candidate_fragment_smiles(mol, full_sys_atoms)

                for ring_idx in sys_rings:
                    full_ring_atoms = get_full_rigid_fragment_atoms(mol, set(atom_rings[ring_idx]))
                    if is_planar(full_ring_atoms):
                        sys_info["level_1"].append(_generate_candidate_fragment_smiles(mol, full_ring_atoms))

                hierarchy.append(sys_info)

            mol_hierarchies.append(hierarchy)
        except Exception as error:
            failures.append(_sample_failure_record(data, sample_index, error, "vocabulary"))

    failure_count = len(failures) - initial_failure_count
    if total_mols and failure_count / total_mols > max_failure_fraction:
        raise FragmentationFailureBudgetExceeded(
            stage="vocabulary",
            split="train",
            failures=failure_count,
            total=total_mols,
            max_failure_fraction=max_failure_fraction,
        )

    vocab_counts = collections.Counter()
    all_known_rare_motifs = set()

    while True:
        vocab_counts = _count_hierarchy_motifs(mol_hierarchies, all_known_rare_motifs)

        current_rare = {smiles for smiles, count in vocab_counts.items() if count < min_count}
        if not current_rare:
            break
        all_known_rare_motifs.update(current_rare)

    return set(vocab_counts)


def get_minimal_stub_cuts(
    mol: Chem.Mol,
    allowed_vocabulary: set,
    *,
    return_retained: bool = False,
) -> List[int] | tuple[List[int], list[dict[str, Any]]]:
    """
    Identifies bonds to cut, restricting generation strictly to the allowed_vocabulary.
    """
    bonds_to_cut = set()
    ri = mol.GetRingInfo()
    bond_rings = ri.BondRings()
    atom_rings = ri.AtomRings()
    try:
        conf = mol.GetConformer()
        all_pos = conf.GetPositions()
    except ValueError:
        return ([], []) if return_retained else []

    def _is_set_planar(indices):
        coords = all_pos[list(indices)]
        return is_geometrically_degenerate(coords, tol_rel=0.05) != "nondegenerate"

    # Identify ring systems through shared atoms.
    ring_systems = _ring_systems_from_atom_rings(atom_rings)

    # Top-Down Decomposition
    safe_bonds = set()
    safe_atoms = set()
    retained_motifs: list[dict[str, Any]] = []

    for sys_rings in ring_systems:
        sys_atom_indices = set()
        sys_bond_indices = set()
        for r_idx in sys_rings:
            sys_atom_indices.update(atom_rings[r_idx])
            sys_bond_indices.update(bond_rings[r_idx])

        # Policy A: Check Level 0 (Whole System)
        full_sys_atoms = get_full_rigid_fragment_atoms(mol, sys_atom_indices)
        if _is_set_planar(full_sys_atoms):
            smi = _generate_candidate_fragment_smiles(mol, full_sys_atoms)
            if smi in allowed_vocabulary:
                safe_bonds.update(sys_bond_indices)
                safe_atoms.update(sys_atom_indices)
                retained_motifs.append(
                    {
                        "vocabulary_smiles": smi,
                        "level": "ring_system",
                        "atom_indices": frozenset(full_sys_atoms),
                    }
                )
                continue

        # Policy B: Check Level 1 (Individual Rings)
        kept_atoms_in_sys = set()
        eligible_rings = sorted(list(sys_rings), key=lambda r: -len(atom_rings[r]))

        for r_idx in eligible_rings:
            r_atoms = set(atom_rings[r_idx])
            if len(kept_atoms_in_sys.intersection(r_atoms)) > 0:
                continue

            full_r_atoms = get_full_rigid_fragment_atoms(mol, r_atoms)
            if _is_set_planar(full_r_atoms):
                r_smi = _generate_candidate_fragment_smiles(mol, full_r_atoms)
                if r_smi in allowed_vocabulary:
                    kept_atoms_in_sys.update(r_atoms)
                    safe_atoms.update(r_atoms)
                    safe_bonds.update(bond_rings[r_idx])
                    retained_motifs.append(
                        {
                            "vocabulary_smiles": r_smi,
                            "level": "ring",
                            "atom_indices": frozenset(full_r_atoms),
                        }
                    )

    # Identify Cuts
    for b in mol.GetBonds():
        idx = b.GetIdx()
        u = b.GetBeginAtom()
        v = b.GetEndAtom()

        # Keep hydrogen bonds
        if u.GetAtomicNum() == 1 or v.GetAtomicNum() == 1:
            continue

        # Keep Safe Internal Bonds
        if idx in safe_bonds:
            continue

        # CUT ISOLATION: Cut bonds connecting safe rings to outside
        if u.GetIdx() in safe_atoms or v.GetIdx() in safe_atoms:
            bonds_to_cut.add(idx)
            continue

        # Keep Acyclic Double/Triple bonds
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue

        # Cut Acyclic Single bonds AND bonds inside Unsafe rings
        bonds_to_cut.add(idx)

    sorted_cuts = sorted(bonds_to_cut)
    if return_retained:
        return sorted_cuts, retained_motifs
    return sorted_cuts


def decompose_to_rigid_fragments(
    data: Any,
    class_manager: Optional[Any] = None,
    allowed_vocabulary=None,
    *,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
) -> List[Any]:
    if class_manager is None:
        raise ValueError("decompose_to_rigid_fragments requires a FragmentClassManager")

    smi_original = data.smiles

    # Parse Molecule
    src_mol = Chem.MolFromSmiles(smi_original)
    if src_mol is None:
        raise FragmentationSampleError(
            "parse_failed",
            "RDKit could not parse the molecule SMILES.",
            smiles=smi_original,
        )
    src_mol = Chem.AddHs(src_mol)

    pos_np = data.pos.detach().cpu().numpy().astype(np.float64)

    num_src_atoms = src_mol.GetNumAtoms()
    if pos_np.shape[0] != num_src_atoms:
        raise FragmentationSampleError(
            "atom_count_mismatch",
            "Parsed SMILES and input coordinates have different atom counts.",
            smiles=smi_original,
            smiles_atoms=num_src_atoms,
            coordinate_atoms=pos_np.shape[0],
        )

    # Kekulize to expose aromatic bonds as single/double.
    try:
        Chem.Kekulize(src_mol, clearAromaticFlags=True)
    except Exception as error:
        raise FragmentationSampleError(
            "kekulization_failed",
            "RDKit could not kekulize the parsed molecule.",
            smiles=smi_original,
        ) from error

    # Setup Conformer
    conf = Chem.Conformer(num_src_atoms)
    for i in range(num_src_atoms):
        conf.SetAtomPosition(i, pos_np[i])
        src_mol.GetAtomWithIdx(i).SetIntProp("orig_idx", int(i))
    src_mol.AddConformer(conf)

    parent_ranks = list(Chem.CanonicalRankAtoms(src_mol, breakTies=True))

    # Identify Cuts
    bonds_to_cut, retained_motifs = get_minimal_stub_cuts(
        src_mol,
        allowed_vocabulary,
        return_retained=True,
    )

    # Fragment
    if not bonds_to_cut:
        frag_atom_indices = [set(range(num_src_atoms))]
    else:
        frag_mol = Chem.FragmentOnBonds(src_mol, bonds_to_cut, addDummies=False)
        rdkit_frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        frag_atom_indices = []
        for frag in rdkit_frags:
            idxs = set(a.GetIntProp("orig_idx") for a in frag.GetAtoms() if a.HasProp("orig_idx"))
            frag_atom_indices.append(idxs)

    final_frag_indices = list(frag_atom_indices)
    output_fragments: List[Any] = []

    # Build Rigid Fragments
    for f_idx, frag_indices in enumerate(final_frag_indices):
        if not frag_indices:
            continue

        real_atoms = []
        for orig_idx in sorted(frag_indices):
            pos = conf.GetAtomPosition(orig_idx)
            atom_pos_np = np.array([pos.x, pos.y, pos.z])
            a = src_mol.GetAtomWithIdx(orig_idx)
            real_atoms.append({"z": a.GetAtomicNum(), "pos": atom_pos_np, "orig_idx": orig_idx, "is_stub": False})

        # Check Geometric Stability
        real_pos_np = np.array([x["pos"] for x in real_atoms])
        is_real_stable = False

        # Must have >= 3 atoms to define a plane
        if len(real_atoms) >= 3:
            P_c = real_pos_np - real_pos_np.mean(axis=0)
            S = np.linalg.svd(P_c, compute_uv=False)
            # Must be non-linear (2nd singular value > threshold)
            if S[1] > 1e-3 * S[0]:
                is_real_stable = True

        selected_stubs = []

        # Add Stubs ONLY if Unstable
        if not is_real_stable:
            immediate_stubs = []
            for item in real_atoms:
                a_obj = src_mol.GetAtomWithIdx(item["orig_idx"])
                atom_pos = item["pos"]

                for b in a_obj.GetBonds():
                    nbr = b.GetOtherAtom(a_obj)
                    nbr_idx = nbr.GetIdx()

                    # Neighbor is outside fragment -> Stub candidate
                    if nbr_idx not in frag_indices:
                        nbr_rank = parent_ranks[nbr_idx]
                        nbr_pos = np.array(conf.GetAtomPosition(nbr_idx))

                        vec = nbr_pos - atom_pos
                        norm = np.linalg.norm(vec)
                        if norm < 1e-6:
                            norm = 1.0
                        stub_pos = atom_pos + (vec / norm)

                        immediate_stubs.append(
                            {
                                "rank_key": nbr_rank,
                                "z": 0,
                                "pos": stub_pos,
                                "orig_idx": nbr_idx,
                                "is_stub": True,
                                "attached_to": item["orig_idx"],
                            }
                        )

            immediate_stubs.sort(key=lambda x: x["rank_key"])

            # Check stability again with immediate stubs
            current_points = [x["pos"] for x in real_atoms] + [x["pos"] for x in immediate_stubs]
            if len(current_points) >= 3:
                cp_np = np.array(current_points)
                S = np.linalg.svd(cp_np - cp_np.mean(axis=0), compute_uv=False)
                is_real_stable = S[1] > 1e-3 * S[0]

            distant_stubs = []
            if not is_real_stable:
                # Deep Search for non-linear anchor
                visited = set(frag_indices)
                for s in immediate_stubs:
                    visited.add(s["orig_idx"])

                queue = [s["orig_idx"] for s in immediate_stubs]
                idx_ptr = 0
                found_anchor = None
                base_points = [x["pos"] for x in real_atoms] + [x["pos"] for x in immediate_stubs]

                while idx_ptr < len(queue):
                    curr_idx = queue[idx_ptr]
                    idx_ptr += 1
                    curr_atom = src_mol.GetAtomWithIdx(curr_idx)
                    curr_pos = np.array(conf.GetAtomPosition(curr_idx))

                    test_points = np.array(base_points + [curr_pos])
                    S_test = np.linalg.svd(test_points - test_points.mean(axis=0), compute_uv=False)
                    if len(test_points) >= 3 and S_test[1] > 1e-3 * S_test[0]:
                        found_anchor = curr_idx
                        break

                    for b in curr_atom.GetBonds():
                        nbr = b.GetOtherAtom(curr_atom)
                        if nbr.GetIdx() not in visited:
                            visited.add(nbr.GetIdx())
                            queue.append(nbr.GetIdx())

                if found_anchor is not None:
                    anchor_pos = np.array(conf.GetAtomPosition(found_anchor))
                    distant_stubs.append(
                        {
                            "rank_key": 999999,
                            "z": 0,
                            "pos": anchor_pos,
                            "orig_idx": found_anchor,
                            "is_stub": True,
                            "attached_to": next(iter(frag_indices)),
                        }
                    )
                else:
                    raise RuntimeError(f"Fragment {f_idx} is globally linear in {smi_original}")

            selected_stubs = immediate_stubs + distant_stubs

        if len(real_atoms) + len(selected_stubs) < 3:
            raise RuntimeError(f"Fragment has < 3 points total. Mol: {smi_original}")

        # Build Label Molecule
        label_mol = Chem.RWMol()
        node_map = {}

        for item in real_atoms:
            a_obj = Chem.Atom(item["z"])
            a_obj.SetNoImplicit(True)
            p_a = src_mol.GetAtomWithIdx(item["orig_idx"])
            # Copy properties (Stereo is valuable even for heavy-only)
            try:
                a_obj.SetFormalCharge(p_a.GetFormalCharge())
            except RuntimeError:
                pass
            try:
                a_obj.SetChiralTag(p_a.GetChiralTag())
            except RuntimeError:
                pass
            a_obj.SetIsAromatic(False)
            idx = label_mol.AddAtom(a_obj)
            node_map[item["orig_idx"]] = idx

        for item in selected_stubs:
            a_obj = Chem.Atom(0)
            a_obj.SetNoImplicit(True)
            idx = label_mol.AddAtom(a_obj)
            node_map[id(item)] = idx

        real_ids = {item["orig_idx"] for item in real_atoms}
        for bond in src_mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if u in real_ids and v in real_ids:
                label_mol.AddBond(node_map[u], node_map[v], bond.GetBondType())

        for s in selected_stubs:
            label_mol.AddBond(node_map[s["attached_to"]], node_map[id(s)], Chem.BondType.SINGLE)

        final_label_mol = label_mol.GetMol()
        try:
            ranks = list(Chem.CanonicalRankAtoms(final_label_mol, breakTies=True))
        except RuntimeError:
            ranks = list(range(final_label_mol.GetNumAtoms()))

        for item in real_atoms:
            item["rank"] = ranks[node_map[item["orig_idx"]]]
        for item in selected_stubs:
            item["rank"] = (ranks[node_map[item["attached_to"]]], ranks[node_map[id(item)]])

        real_atoms.sort(key=lambda x: x["rank"])
        selected_stubs.sort(key=lambda x: x["rank"])

        ordered_data = real_atoms + selected_stubs
        ordered_pos = np.array([x["pos"] for x in ordered_data], dtype=np.float32)
        ordered_z = np.array([x["z"] for x in ordered_data], dtype=np.int64)
        ordered_orig = [x["orig_idx"] for x in real_atoms]

        # Check canonical layout
        stub_indices = np.where(ordered_z == 0)[0]
        if stub_indices.size > 0:
            if not np.all(ordered_z[: stub_indices[0]] > 0):
                raise AssertionError("Canonical ordering violated")

        # Extract connectivity and bond types.
        label_idx_to_ordered_idx = {}
        for ord_idx, item in enumerate(ordered_data):
            key = item["orig_idx"] if not item["is_stub"] else id(item)
            label_idx_to_ordered_idx[node_map[key]] = ord_idx

        edge_list = []
        attr_list = []

        for b in final_label_mol.GetBonds():
            u_lbl = b.GetBeginAtomIdx()
            v_lbl = b.GetEndAtomIdx()
            if u_lbl in label_idx_to_ordered_idx and v_lbl in label_idx_to_ordered_idx:
                u_ord = label_idx_to_ordered_idx[u_lbl]
                v_ord = label_idx_to_ordered_idx[v_lbl]

                # Convert RDKit types to Integer Order
                bt = b.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    order = 1
                elif bt == Chem.BondType.DOUBLE:
                    order = 2
                elif bt == Chem.BondType.TRIPLE:
                    order = 3
                else:
                    order = 1  # Fallback

                # Add Undirected
                edge_list.append([u_ord, v_ord])
                edge_list.append([v_ord, u_ord])
                attr_list.append(order)
                attr_list.append(order)

        if edge_list:
            exemplar_edge_index = np.array(edge_list, dtype=np.int64).T
            exemplar_edge_attr = np.array(attr_list, dtype=np.int64)
        else:
            exemplar_edge_index = np.empty((2, 0), dtype=np.int64)
            exemplar_edge_attr = np.empty((0,), dtype=np.int64)

        # Graph Automorphisms
        node_to_ordered = {
            node_map[d["orig_idx"] if not d["is_stub"] else id(d)]: i for i, d in enumerate(ordered_data)
        }
        mol_auto = Chem.Mol(final_label_mol)
        graph_z = np.array([a.GetAtomicNum() for a in final_label_mol.GetAtoms()])
        raw_perms = _get_graph_automorphisms_preserve_element(
            mol_auto,
            exemplar_z=graph_z,
            max_automorphisms=max_automorphisms,
        )
        real_perms = []
        for p in raw_perms:
            try:
                real_perms.append(
                    [node_to_ordered[p[node_map[d["orig_idx"] if not d["is_stub"] else id(d)]]] for d in ordered_data]
                )
            except (IndexError, KeyError):
                continue
        real_perms = _filter_perms_preserve_atomic_numbers(real_perms, ordered_z)
        uniq = sorted(list(set(tuple(x) for x in real_perms)))
        real_perms = [list(x) for x in uniq]
        if not real_perms:
            real_perms = [list(range(len(ordered_pos)))]

        # Create the vocabulary class.
        try:
            lbl = Chem.MolToSmiles(final_label_mol, isomericSmiles=True, canonical=True, allHsExplicit=True)
        except RuntimeError:
            lbl = "Error"

        try:
            class_idx, _, align = class_manager.find_or_create_class(
                lbl,
                ordered_pos,
                ordered_z,
                exemplar_edge_index,
                exemplar_edge_attr,
                perms_from_canon=real_perms,
            )
        except FrozenFragmentLibraryError as error:
            fallback_candidates = sorted(
                [
                    {
                        "vocabulary_smiles": candidate["vocabulary_smiles"],
                        "level": candidate["level"],
                    }
                    for candidate in retained_motifs
                    if candidate["atom_indices"].issubset(frag_indices)
                ],
                key=lambda candidate: (candidate["level"] != "ring_system", candidate["vocabulary_smiles"]),
            )
            raise error.with_fallback_context(fallback_candidates=fallback_candidates) from error

        frag_d = FragmentData(
            pos=torch.tensor(ordered_pos, dtype=torch.float32), z=torch.tensor(ordered_z, dtype=torch.int64), smiles=lbl
        )
        frag_d.class_id = class_idx
        frag_d.original_indices = ordered_orig

        if align:
            R, t, _, _ = align
            frag_d.rot_gt = torch.tensor(R.astype(np.float32))
            frag_d.trans_gt = torch.tensor(t.astype(np.float32))

        output_fragments.append(frag_d)

    return output_fragments


def decompose_with_frozen_fallback(
    data: Any,
    class_manager: Any,
    allowed_vocabulary: set,
    *,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
) -> tuple[List[Any], list[dict[str, str]]]:
    """Recursively refine held-out retained motifs until every class is known."""
    active_vocabulary = set(allowed_vocabulary or set())
    fallback_history: list[dict[str, str]] = []

    while True:
        try:
            fragments = decompose_to_rigid_fragments(
                data,
                class_manager,
                active_vocabulary,
                max_automorphisms=max_automorphisms,
            )
            return fragments, fallback_history
        except FrozenFragmentLibraryError as error:
            candidates = [
                candidate
                for candidate in error.fallback_candidates
                if candidate["vocabulary_smiles"] in active_vocabulary
            ]
            if not candidates:
                raise error.with_fallback_context(fallback_history=fallback_history) from error

            candidate = candidates[0]
            active_vocabulary.remove(candidate["vocabulary_smiles"])
            fallback_history.append(
                {
                    "removed_vocabulary_smiles": candidate["vocabulary_smiles"],
                    "removed_level": candidate["level"],
                    "failed_fragment_smiles": error.smiles,
                    "failure_kind": error.failure_kind,
                }
            )
