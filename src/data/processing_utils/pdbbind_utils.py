from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import pandas as pd
from src.data.processing_utils import base
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SDMolSupplier
from tqdm import tqdm
import math

# TODO: figure out logging

@lru_cache
def load_pdb_bind_index_file(
    pdb_bind_dir: str, year: str = "2019", set: str = "refined-set"
) -> pd.DataFrame:
    """Reads a PDBBind dataset index file from disk"""
    if set == "refined-set":
        fname = "INDEX_refined_data.2019"
    else:
        raise NotImplementedError

    df = pd.read_csv(
        Path(pdb_bind_dir) / year / "plain-text-index" / "index" / fname,
        skiprows=6,
        header=None,
        delim_whitespace=True,
    ).drop(columns=5)
    print(df)

    df.columns = [
        "pdb",
        "resolution",
        "release",
        "-logKd/Ki",
        "Kd/Ki",
        "reference",
        "ligand_name",
    ]

    return df


def load(
    config: DictConfig,
) -> Tuple[base.Dataset, base.Dataset, base.Dataset]:
    """Loads DUD-E Dataset into Dataset objects"""
    # Load index file
    df = load_pdb_bind_index_file(
        pdb_bind_dir=config.path,
        year=str(config.year),
        set="refined-set",
    )

    # If using a fraction of the dataset for prototyping, select it here
    # log.info(f"Using {config.dataset.fraction * 100}% of PDBBind dataset")
    df = df.sample(frac=config.fraction)

    # Load dataset from index
    dataset = load_pdb_bind_dataset(config=config, df=df)

    return split(dataset, config.split, test_only=False)


def load_pdb_bind_dataset(
    config: DictConfig, df: pd.DataFrame
) -> base.DictDataset:
    """
    Specifies loading workflow for PDBBind dataset.

    Extract targets from index DF,
    load ligand,
    return total dataset wrapper
    """
    # log.info(
    #     f"Loading PDB Bind {config.dataset.year} dataset from: {config.dataset.path}"
    # )
    # Extract target names
    target_names = df["pdb"]

    # Load protein structures into a dictionary indexed by target
    structures = load_proteins(
        pdb_bind_root_path=Path(config.path)
        / str(config.year)
        / "refined-set",
        target_names=target_names,
    )

    # Some structure do not have crystal ligands so we remove these.
    target_names = [name for name in target_names if name in structures.keys()]

    # Load ligands into a dictionary indexed by target
    ligands = load_mols(
        pdb_bind_root_path=Path(config.path)
        / str(config.year)
        / "refined-set",
        target_names=target_names,
    )

    # Process dictionaries into dataframes
    ligands = (
        pd.DataFrame.from_dict(ligands, orient="index")
        .T.unstack()
        .dropna()
        .reset_index(level=0, drop=False)
    )

    ligands.columns = ["target", "mol"]

    # Get affinity label. Very ugly can be made more pandas friendly
    labels = [
        df.loc[df["pdb"] == target]["-logKd/Ki"].item()
        for target in ligands["target"]
    ]

    samples = {}
    samples["target"] = list(ligands["target"])
    samples["mol"] = list(ligands["mol"])
    samples["label"] = labels
    samples["structures"] = structures

    del ligands
    del structures
    del labels

    return base.DictDataset(samples)


def load_proteins(
    pdb_bind_root_path: Path, target_names: pd.Series
) -> Dict[str, np.array]:
    """
    Loads protein atom coordinates and types for dMaSIF
    :param dude_root_path: Path object pointing to PDBBind dataset root.
    :param target_names: pd.Series containing the target names
    :return: Dictionary of atom types and coordinates
    """
    # log.info(f"Loading protein structures...")
    atom_info = {}
    for target in tqdm(target_names, mininterval=3):
        target_dict = {}
        atom_coords = np.load(
            pdb_bind_root_path / target.lower() / f"{target}_atomxyz.npy"
        )
        atom_types = np.load(
            pdb_bind_root_path / target.lower() / f"{target}_atomtypes.npy"
        )
        # ligand = Chem.MolFromMol2File(
        #    str(pdb_bind_root_path / target.lower() / f"{target}_ligand.mol2")
        # )
        ligand = next(
            pybel.readfile(
                "mol2",
                str(
                    pdb_bind_root_path
                    / target.lower()
                    / f"{target}_ligand.mol2"
                ),
            )
        )
        try:
            # lig_coords = ligand.GetConformer().GetPositions()
            lig_coords = np.array([atom.coords for atom in ligand])
        except:
            # print(ligand.GetConformer().GetPositions())
            # log.info(
            #     f"Crystal Ligand for {target} can't be read. Skipping this structure."
            # )
            continue

        target_dict["target_coords"] = atom_coords
        target_dict["target_types"] = atom_types
        target_dict["ligand_coords"] = lig_coords

        atom_info[target] = target_dict
    # log.info(f"Successfully loaded: {len(atom_info.keys())} structures.")
    return atom_info


def load_mols(
    pdb_bind_root_path: Path, target_names: pd.Series
) -> Dict[str, List[rdkit.Chem.Mol]]:
    """
    Loads molecules in DUD-E from an SDFSupplier using RDKit. Positive and decoy ligands for each target are each stored
    in a SDF file containing all the molecules. We produce a dictionary indexed by the target name and containing
    a list of all the mol objects for that target.
    :param dude_root_path: Path object pointing to DUDE dataset root.
    :param target_names: pd.Series containing the target names
    :param type: str {"actives", "decoys"} indicating whether or not to load active or inactive molecules
    :return: Dictionary of targets and molecules
    """
    # log.info(f"Loading molecules...")

    target_mol_map = {}
    count = 0
    # Some molecules are broken/no files provided. We handle this here.
    for target in tqdm(target_names, mininterval=3):
        try:
            target_mol_map[target] = Chem.MolFromMol2File(
                str(
                    pdb_bind_root_path
                    / target.lower()
                    / f"{target}_ligand.mol2"
                )
            )
            count += 1
        except:
            print(
                str(
                    pdb_bind_root_path
                    / target.lower()
                    / f"{target}_ligand.mol2"
                )
            )
            # log.info(f"{target} has a broken ligand.")
            continue
    # log.info(f"Successfully loaded: {count} {type} molecules.")
    return target_mol_map

def split(
    dataset: base.Dataset, config: DictConfig, test_only: bool = False
) -> Tuple[Union[base.Dataset, None], Union[base.Dataset, None], base.Dataset]:
    """
    Splits dataset into training testing and validation
    """
    if config.ratio_train + config.ratio_valid + config.ratio_test != 1:
        raise ValueError(
            f"Dataset ratios do not sum to 1. Train: {config.ratio_train}, val: {config.ratio_valid}, "
            f"test: {config.ratio_test}"
        )

    if config.shuffle:
        dataset = base.ShuffledDataset(dataset)

    # Get indices for splitting dataset into train, val and test
    train_index_min = 0
    train_index_max = math.ceil(len(dataset) * config.ratio_train)

    valid_index_min = train_index_max
    valid_index_max = math.ceil(
        len(dataset) * (config.ratio_train + config.ratio_valid)
    )

    test_index_min = valid_index_max
    test_index_max = len(dataset)

    # If we only want a test set, we don't need to split the dataset
    if test_only:
        return (
            None,
            None,
            base.PartialDataset(dataset, train_index_min, test_index_max),
        )

    # Construct partial datasets
    if config.ratio_train > 0:
        train_dataset = base.PartialDataset(
            dataset, train_index_min, train_index_max
        )
    else:
        train_dataset = None

    if config.ratio_valid > 0:
        valid_dataset = base.PartialDataset(
            dataset, valid_index_min, valid_index_max
        )
    else:
        valid_dataset = None

    if config.ratio_test > 0:
        test_dataset = base.PartialDataset(dataset, test_index_min, test_index_max)
    else:
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset