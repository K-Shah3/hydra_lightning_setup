from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from src.data.features.base import Feature, T
from torch_geometric.data import Data

tensor = torch.FloatTensor
inttensor = torch.LongTensor


class ProteinFeature(Feature[list]):
    """Class for handling protein target representation and feature transforms."""

    def __init__(self, center: bool):
        self.center = center

    def transform(self, batch: List[T], batch_target_name: List[str]) -> np.ndarray:
        """Iterates over targets in batch to apply transformation functions."""
        batch = [self._featurise_protein(b, name) for b, name in zip(batch, batch_target_name)]
        return batch

    def _featurise_protein(self, protein: np.ndarray, name: str) -> Dict[str, np.ndarray]:
        """Performs featurisation of protein. Loads structure & pre-processes for MaSIF."""
        xyz = torch.tensor(protein["target_coords"], dtype=torch.float32)
        types = torch.tensor(protein["target_types"], dtype=torch.float32)
        ligand = torch.tensor(protein["ligand_coords"], dtype=torch.float32)
        return Data(
            target_coords=xyz, target_types=types, ligand_coords=ligand, target_name=name
        )

    @staticmethod
    def load_protein_npy(fpath: Path, center=False, single_pdb=False):
        """Loads a protein surface mesh and its features"""

        # Load the data, and read the connectivity information:
        triangles = (
            None
            if single_pdb
            else inttensor(np.load(fpath / "receptor_triangles.npy")).T
        )
        # Normalize the point cloud, as specified by the user:
        points = (
            None if single_pdb else tensor(np.load(fpath / "receptor_xyz.npy"))
        )
        center_location = (
            None if single_pdb else torch.mean(points, axis=0, keepdims=True)
        )

        atom_coords = tensor(np.load(fpath / "receptor_atomxyz.npy"))
        atom_types = tensor(np.load(fpath / "receptor_atomtypes.npy"))

        if center:
            points = points - center_location
            atom_coords = atom_coords - center_location

        # Interface labels
        iface_labels = (
            None
            if single_pdb
            else tensor(
                np.load(fpath / "receptor_iface_labels.npy").reshape((-1, 1))
            )
        )

        # Features
        chemical_features = (
            None
            if single_pdb
            else tensor(np.load(fpath / "receptor_features.npy"))
        )

        # Normals
        normals = (
            None
            if single_pdb
            else tensor(np.load(fpath / "receptor_normals.npy"))
        )

        return Data(
            xyz=points,
            face=triangles,
            chemical_features=chemical_features,
            y=iface_labels,
            normals=normals,
            center_location=center_location,
            num_nodes=None if single_pdb else points.shape[0],
            atom_coords=atom_coords,
            atom_types=atom_types,
        )
