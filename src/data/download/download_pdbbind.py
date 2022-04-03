import os
from pathlib import Path
from typing import List

import numpy as np
import wget
from download_utils import protonate, extractPDB, load_structure_np

from tqdm import tqdm

# TODO: figure out logging

URL_ROOT: str = "http://www.pdbbind.org.cn/download/"

PDB_BIND_2019_REFINED: str = URL_ROOT + "pdbbind_v2019_refined.tar.gz"
PDB_BIND_2019_GENERAL: str = URL_ROOT + "pdbbind_v2019_other_PL.tar.gz"
PDB_BIND_2019_INDEX: str = URL_ROOT + "PDBbind_2019_plain_text_index.tar.gz"
PDB_BIND_2016_REFINED: str = URL_ROOT + "pdbbind_v2016_refined.tar.gz"
PDB_BIND_2016_GENERAL: str = (
    URL_ROOT + "pdbbind_v2016_general-set-except-refined.tar.gz"
)
PDB_BIND_2016_INDEX: str = URL_ROOT + "PDBbind_2016_plain_text_index.tar.gz"
CASF_2016: str = URL_ROOT + "CASF-2016.tar.gz"
CASF_2013: str = URL_ROOT + "CASF-2013-updated.tar.gz"
CASF_2007: str = URL_ROOT + "CASF-2007.tar.gz"

SUPPORTED_PDBBIND_YEARS: List[str] = ["2016", "2019"]
SUPPORTED_CASF_YEARS: List[str] = ["2007", "2013", "2016"]


def download_pdbbind(
    out_dir: str = "pdb_bind",
    year: str = "2019",
    refined: bool = True,
    general: bool = False,
) -> str:
    """Downloads PDBBind releases by year
    :param out_dir: Specifies output directory, defaults to "pdb_bind"
    :type out_dir: str, optional
    :param year: Release year to download, defaults to "2019"
    :type year: str, optional
    :param refined: Whether or not to download the refined set, defaults to True
    :type refined: bool, optional
    :param general: Whether or not to download the general set, defaults to False
    :type general: bool, optional
    :return: Path to output directory.
    :rtype: str
    """
    out_path = Path(out_dir) / year
    # Check directory exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Download dataset
    # log.info(
    #     f"Downloading PDBBind {year} to {out_path}. General Set: {general}. Refined set: {refined}"
    # )

    if year == "2019":
        os.system(f"wget -c {PDB_BIND_2019_INDEX} -P {out_path}")
        os.system(
            f"tar -xvf {out_path}/PDBbind_2019_plain_text_index.tar.gz -C {out_path}"
        )
        if general:
            os.system(f"wget -c {PDB_BIND_2019_GENERAL} -P {out_path}")
            os.system(
                f"tar -xvf {out_path}/pdbbind_v2019_other_PL.tar.gz -C {out_path}"
            )
        if refined:
            os.system(f"wget -c {PDB_BIND_2019_REFINED} -P {out_path}")
            os.system(
                f"tar -xvf {out_path}/pdbbind_v2019_refined.tar.gz -C {out_path}"
            )

    else:
        message = f"{year} is not implemented"
        # log.error(message)
        raise NotImplementedError

    return out_path

def protonate_pdbbind(data_dir: Path):
    """
    Protonate structures
    Traverse directory structure and protonate receptor.pdb files
    :param data_dir: Path to dataset.
    :type data_dir: Path
    """
    # log.info(f"Protonating PDBBind Dataset in {data_dir}")
    dirs = [
        f
        for f in os.listdir(Path(data_dir))
        if not f.startswith(".") and f != "index"
    ]

    assert "index" not in dirs

    for target in tqdm(dirs):
        # log.debug(f"Extracting protein from {target}")
        protein_protonated_file = Path(data_dir) / target / f"{target}_protein_protonated.pdb"
        if protein_protonated_file.exists():
            os.remove(Path(data_dir) / target / f"{target}_protein_protonated.pdb")

        protonate(
            in_pdb_file=Path(data_dir) / target / f"{target}_protein.pdb",
            out_pdb_file=Path(data_dir)
            / target
            / f"{target}_protein_protonated.pdb",
        )

def extractPDB_pdbbind(data_dir: Path):
    """Performs extraction from protonated structure to desired subset.
    Eg. removes hetatoms and disordered atoms
    :param data_dir: Path to dude dataset
    :type data_dir: Path
    """
    # log.info("Extracting PDBs from PDBBind")
    dirs = [
        f
        for f in os.listdir(data_dir)
        if not f.startswith(".") and f != "index" and f != "readme"
    ]

    assert "index" not in dirs
    assert "readme" not in dirs

    for target in tqdm(dirs):
        extractPDB(
            infilename=str(
                Path(
                    data_dir / target / f"{target}_protein_protonated.pdb"
                ).resolve()
            ),
            outfilename=str(
                Path(
                    data_dir / target / f"{target}_protein_protonated.pdb"
                ).resolve()
            ),
        )

def save_np(pdb_bind_dir: Path, center=False):
    # Save NPYs
    if refined:
        dirs = [
            f
            for f in os.listdir(pdb_bind_dir / year / "refined-set")
            if not f.startswith(".") and f != "index" and f != "readme"
        ]
        assert "index" not in dirs
        assert "readme" not in dirs

        # log.info(
        #     f"Writing PDBs to NP arrays {pdb_bind_dir}/{year}/refined-set/"
        # )
        for target in tqdm(dirs):
            filename = (
                pdb_bind_dir
                / year
                / "refined-set"
                / target
                / f"{target}_protein_protonated.pdb"
            )
            protein = load_structure_np(filename, center=center)
            np.save(
                pdb_bind_dir
                / year
                / "refined-set"
                / target
                / f"{target}_atomxyz.npy",
                protein["xyz"],
            )
            np.save(
                pdb_bind_dir
                / year
                / "refined-set"
                / target
                / f"{target}_atomtypes.npy",
                protein["types"],
            )

if __name__ == "__main__":
    out_dir = "pdbbind"
    year = "2019"
    refined = True
    general = False
    # out_path = download_pdbbind(out_dir, year, refined, general)
    out_path = Path("pdbbind/2019/refined-set")
    print(f'out path {out_path}')
    # protonate_pdbbind(out_path)
    # extractPDB_pdbbind(out_path)
    # save_np(Path(out_dir))