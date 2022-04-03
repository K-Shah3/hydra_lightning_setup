from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops
import torch
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
import rdkit

from src.data.features.base import Feature, T

class MolFeatureExtractionError(Exception):
    pass


allowable_atoms = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "H",  # H?
    "Stop",
    "Unknown",
]

class MoleculeFeature(Feature[list]):
    """Class for handling molecular representation and feature transforms"""

    def __init__(
        self,
        add_Hs: bool,
        kekulize: bool,
        max_atoms: int,
        out_size: int,
        scaffold: bool,
        bool_id_feat: bool = True,
    ):
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.scaffold = scaffold
        self.bool_id_feat = bool_id_feat

        self.ALLOWED_ATOMS = allowable_atoms

        self.ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.ALLOWED_VALENCES = [0, 1, 2, 3, 4, 5, 6]
        self.ALLOWED_HYBRIDIZATIONS = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

    def transform(self, batch: List[T]) -> List[np.ndarray]:
        """Iterates over Mols in batch to apply transformation functions"""
        batch = [self._featurise_moleclue(b) for b in batch]

        # outputs = add_batch_padding(batch)
        # return np.asarray(outputs, dtype=np.int)
        return batch

    def get_input_features(self, mol: rdkit.Chem.Mol):
        """get input features
        Args:
            mol (Mol):
        Returns:
        """

    @staticmethod
    def type_check_num_atoms(mol: rdkit.Chem.Mol, num_max_atoms: int = -1):
        """Check number of atoms in `mol` does not exceed `num_max_atoms`
        If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
        raise `MolFeatureExtractionError` exception.
        Args:
            mol (Mol):
            num_max_atoms (int): If negative value is set, not check number of
                atoms.
        """
        num_atoms = mol.GetNumAtoms()
        if num_max_atoms >= 0 and num_atoms > num_max_atoms:
            # Skip extracting feature. ignore this case.
            raise MolFeatureExtractionError(
                "Number of atoms in mol {} exceeds num_max_atoms {}".format(
                    num_atoms, num_max_atoms
                )
            )

    # --- Atom preprocessing ---

    def _featurise_moleclue(self, mol: rdkit.Chem.Mol) -> Data:
        """Applies featurisation transforms to a rdkit Mol"""
        # Apply RDKit preprocessing
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        if self.scaffold:
            mol = MurckoScaffold.GetScaffoldForMol(mol)

        # Perform checks
        self.type_check_num_atoms(mol, self.max_atoms)

        # Compute features
        node_attr = [self.atom_features(atom) for atom in mol.GetAtoms()]
        edge_attr = [self.bond_features(bond) for bond in mol.GetBonds()]
        # mol_attr = self.mol_descriptors(mol)

        edge_index = self.get_bond_pair(mol)

        return Data(
            x=torch.tensor(node_attr, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(
                edge_attr, dtype=torch.float
            ).repeat_interleave(2, dim=0),
            # mol_descriptors=mol_attr,
        )

    @staticmethod
    def get_bond_pair(mol: rdkit.Chem.Mol):
        bonds = mol.GetBonds()
        res = [[], []]
        for bond in bonds:
            res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        return res

    @staticmethod
    def mol_descriptors(
        mol: rdkit.Chem.Mol, descriptor_list: Optional[List[str]] = None
    ):
        # Retrieve list of possible descriptors
        descriptors = {d[0]: d[1] for d in Descriptors.descList}

        # Subset descriptors to those provided
        if descriptor_list is not None:
            descriptors = {
                k: v for k, v in descriptors.items() if k in descriptor_list
            }
        return {d: descriptors[d](mol) for d in descriptors}

    def bond_features(self, bond, use_chirality: bool = False):
        bt = bond.GetBondType()
        bond_feats = [  # TODO remove Congugated and Ring types
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]  # ,
        # bond.GetIsConjugated(),
        # bond.IsInRing(),
        # ]
        if use_chirality:
            bond_feats += self.one_of_k_encoding_unk(
                str(bond.GetStereo()),
                ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
            )
        return np.array(bond_feats)

    @staticmethod
    def one_of_k_encoding(x: Iterable, allowable_set: Iterable) -> List:
        if x not in allowable_set:
            raise ValueError(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_of_k_encoding_unk(x: Iterable, allowable_set: Iterable) -> List:
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(
        self,
        atom,
        explicit_H: bool = False,
        use_chirality: bool = False,
    ):
        if self.bool_id_feat:
            return np.array(
                self.one_of_k_encoding_unk(
                    atom.GetSymbol(), self.ALLOWED_ATOMS
                )
            )
        else:
            results = (
                self.one_of_k_encoding_unk(
                    atom.GetSymbol(), self.ALLOWED_ATOMS
                )
                + self.one_of_k_encoding(
                    atom.GetDegree(), self.ALLOWED_DEGREES
                )
                + self.one_of_k_encoding_unk(
                    atom.GetImplicitValence(), self.ALLOWED_VALENCES
                )
                + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
                + self.one_of_k_encoding_unk(
                    atom.GetHybridization(),
                    self.ALLOWED_HYBRIDIZATIONS,
                )
                + [atom.GetIsAromatic()]
            )
            # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
            if not explicit_H:
                results = results + self.one_of_k_encoding_unk(
                    atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
                )
            if use_chirality:
                try:
                    results = (
                        results
                        + self.one_of_k_encoding_unk(
                            atom.GetProp("_CIPCode"), ["R", "S"]
                        )
                        + [atom.HasProp("_ChiralityPossible")]
                    )
                except:
                    results = (
                        results
                        + [False, False]
                        + [atom.HasProp("_ChiralityPossible")]
                    )

            return np.array(results)