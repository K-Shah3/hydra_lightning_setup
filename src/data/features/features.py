from enum import Enum
from typing import Any, Dict, NamedTuple
from omegaconf import DictConfig
from src.data.features.base import Feature
from src.data.features.molecule import MoleculeFeature
from src.data.features.protein import ProteinFeature

from pydantic import BaseModel

class FeatureType(Enum):
    SEQUENCE = "sequence"
    ESM = "ESM"
    CONTACT_MAP = "contact_map"
    MOL_GRAPH = "mol_graph"
    MOL_SCAFFOLD = "mol_scaffold"
    PROTEIN_GRAPH = "protein_graph"
    MASIF = "masif" 

def create_feature(
    feature_type: FeatureType,
    config: DictConfig
) -> Feature:
    """Creates feature objects from feature config and type specification"""
    if FeatureType.MOL_GRAPH == feature_type:
        # return MoleculeFeature(config.representation.ligand) Todo configuration parsing
        ligand_config = config.features.ligand_preprocessing
        return MoleculeFeature(
            add_Hs=ligand_config.add_Hs,
            kekulize=ligand_config.kekulize,  # Todo Kekulization not working
            max_atoms=ligand_config.max_atoms,
            out_size=ligand_config.out_size,
            scaffold=ligand_config.scaffold
        )
    elif FeatureType.MASIF == feature_type:
        return ProteinFeature(center=False)  # Todo
    else:
        message = f"Unsupported Type {feature_type}"
        # log.error(message)
        raise ValueError(message)