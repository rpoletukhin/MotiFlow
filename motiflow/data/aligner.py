import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data

from motiflow.utils.output import suppress_native_output


class TopologyAligner:
    def __init__(self, use_hueckel=False, extract_fragment=True, all_hs_explicit=True, threshold=3.5):
        self.use_hueckel = use_hueckel
        self.extract_fragment = extract_fragment
        self.all_hs_explicit = all_hs_explicit
        self.threshold = float(threshold)

    @staticmethod
    def bleach_mol(mol: Chem.Mol) -> Chem.Mol:
        m = Chem.RWMol(mol)
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
            b.SetStereo(Chem.BondStereo.STEREONONE)
        for a in m.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            a.SetNoImplicit(True)
        return m.GetMol()

    @staticmethod
    def get_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) == 1:
            return mol
        return max(frags, key=lambda m: m.GetNumAtoms())

    def align(
        self,
        sample_id,
        atomic_numbers,
        positions,
        target_smiles,
        aux_data=None,
        source_mol: Chem.Mol | None = None,
    ):
        # Never mutate task metadata owned by the caller.
        aux_data = dict(aux_data or {})

        # Prefer trustworthy dataset topology when it is available. GEOM stores an
        # RDKit molecule and QMugs stores an SDF bond graph, so recomputing their
        # connectivity from distances is both slower and less reliable.
        if source_mol is not None:
            src_geom = Chem.Mol(source_mol)
            if src_geom.GetNumAtoms() != len(atomic_numbers):
                return (
                    "MISMATCH",
                    f"Source topology has {src_geom.GetNumAtoms()} atoms but coordinates have {len(atomic_numbers)}",
                )
            source_z = [atom.GetAtomicNum() for atom in src_geom.GetAtoms()]
            input_z = [int(value) for value in atomic_numbers]
            if source_z != input_z:
                return "MISMATCH", "Source topology atom order does not match atomic numbers"
        else:
            rw_mol = Chem.RWMol()
            conf = Chem.Conformer(len(atomic_numbers))
            for i, z in enumerate(atomic_numbers):
                atom = Chem.Atom(int(z))
                atom.SetNoImplicit(True)
                rw_mol.AddAtom(atom)
                conf.SetAtomPosition(i, positions[i].tolist())
            rw_mol.AddConformer(conf)
            src_geom = rw_mol.GetMol()

            with suppress_native_output(stdout=False):
                try:
                    if self.use_hueckel:
                        try:
                            rdDetermineBonds.DetermineConnectivity(src_geom, useHueckel=True)
                        except (RuntimeError, ValueError):
                            rdDetermineBonds.DetermineConnectivity(src_geom, useHueckel=False)
                    else:
                        rdDetermineBonds.DetermineConnectivity(src_geom, useHueckel=False)
                except Exception as e:
                    return "ERROR", f"Connectivity calc failed: {e}"

        source_atom_indices = torch.arange(src_geom.GetNumAtoms(), dtype=torch.long)
        if self.extract_fragment:
            atom_mappings = []
            source_fragments = Chem.GetMolFrags(
                src_geom,
                asMols=True,
                sanitizeFrags=False,
                fragsMolAtomMapping=atom_mappings,
            )
            if len(source_fragments) > 1:
                largest_index = max(
                    range(len(source_fragments)), key=lambda index: source_fragments[index].GetNumAtoms()
                )
                selected_indices = torch.tensor(atom_mappings[largest_index], dtype=torch.long)
                src_geom = source_fragments[largest_index]
                positions = positions[selected_indices]
                atomic_numbers = atomic_numbers[selected_indices]
                source_atom_indices = source_atom_indices[selected_indices]

        # Target Graph
        try:
            params_raw = Chem.SmilesParserParams()
            params_raw.removeHs = False
            mol_raw = Chem.MolFromSmiles(target_smiles, params=params_raw)
            if mol_raw is None:
                return "ERROR", "SMILES Parse failed"

            if self.extract_fragment:
                mol_frag = self.get_largest_fragment(mol_raw)
                try:
                    mol_clean = Chem.RemoveHs(mol_frag)
                except (RuntimeError, ValueError):
                    mol_clean = mol_frag

                std_canon_smiles = Chem.MolToSmiles(
                    mol_clean, isomericSmiles=True, canonical=True, allHsExplicit=self.all_hs_explicit
                )
                tgt = Chem.MolFromSmiles(std_canon_smiles)
                if tgt is None:
                    return "ERROR", "Standard Reconstruction failed"
                tgt = Chem.AddHs(tgt)
            else:
                # QM9 EXACT matching: No canonicalization before AddHs
                tgt = Chem.AddHs(mol_raw)
                std_canon_smiles = Chem.MolToSmiles(tgt, allHsExplicit=True, isomericSmiles=True)

            if tgt.GetNumAtoms() != src_geom.GetNumAtoms():
                return "MISMATCH", f"Atoms: Pos {src_geom.GetNumAtoms()} vs StdTgt {tgt.GetNumAtoms()}"
        except Exception as e:
            return "ERROR", f"Target Prep failed: {e}"

        # Alignment
        src_skel = self.bleach_mol(src_geom)
        tgt_skel = self.bleach_mol(tgt)

        match_params = Chem.SubstructMatchParameters()
        # A full-graph isomorphism maps every target bond to a source bond. It is
        # unnecessary (and potentially combinatorial) to enumerate all symmetry
        # mappings before checking geometry.
        match_params.uniquify = True
        match_params.useChirality = False
        match_params.maxMatches = 1

        match = src_skel.GetSubstructMatch(tgt_skel, params=match_params)
        if not match:
            return "MISMATCH", "Graph Isomorphism failed"

        # Verification
        valid_perm, final_pos, error_msg = None, None, "All perms failed geometry"
        if self.threshold == float("inf"):
            local_perm = torch.tensor(list(match), dtype=torch.long)
            valid_perm = source_atom_indices[local_perm]
            final_pos = positions[local_perm]
        else:
            local_perm = torch.tensor(match, dtype=torch.long)
            candidate_pos = positions[local_perm]

            is_valid = True
            for bond in tgt.GetBonds():
                dist = (
                    (candidate_pos[bond.GetBeginAtomIdx()] - candidate_pos[bond.GetEndAtomIdx()])
                    .pow(2)
                    .sum()
                    .sqrt()
                    .item()
                )
                if dist > self.threshold:
                    is_valid = False
                    error_msg = f"Bond dist {dist:.2f}A > {self.threshold}A"
                    break

            if is_valid:
                valid_perm = source_atom_indices[local_perm]
                final_pos = candidate_pos

        if valid_perm is None:
            return "MISMATCH", f"Geo Check: {error_msg}"

        # Build PyG Data
        new_z = torch.tensor([a.GetAtomicNum() for a in tgt.GetAtoms()], dtype=torch.long)
        row, col, bond_types = [], [], []
        for bond in tgt.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            t_val = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 4,
            }.get(bt, 0)
            row.extend([u, v])
            col.extend([v, u])
            bond_types.extend([t_val, t_val])

        # Determine which SMILES preserves the exact topological atom order of `tgt`
        if self.extract_fragment:
            topology_smiles = std_canon_smiles
        else:
            topology_smiles = target_smiles

        data = Data(
            sample_id=sample_id,
            h=new_z,
            pos=final_pos,
            smiles=topology_smiles,  # Use the SMILES that dictates `tgt`'s atom order
            canonical_smiles=std_canon_smiles,  # Keep the canonical string explicitly separated
            original_smiles=target_smiles,
            edge_index=torch.tensor([row, col], dtype=torch.long),
            edge_type=torch.tensor(bond_types, dtype=torch.long),
            alignment_permutation=valid_perm,
        )

        if "chembl_id" in aux_data:
            data.chembl_id = aux_data.pop("chembl_id")
        if "conf_id" in aux_data:
            data.conf_id = aux_data.pop("conf_id")

        def to_tensor(val):
            if torch.is_tensor(val):
                return val.float()
            if isinstance(val, (list, tuple)):
                return torch.tensor(val, dtype=torch.float)
            return torch.tensor([val], dtype=torch.float)

        for k, v in aux_data.items():
            if v is not None:
                setattr(data, k, to_tensor(v))

        return "OK", data
