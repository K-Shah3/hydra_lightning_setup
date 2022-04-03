import logging
from pathlib import Path
from subprocess import PIPE, Popen
import numpy as np
from typing import Dict

from Bio.PDB import *
from Bio.PDB import StructureBuilder
from Bio.SeqUtils import IUPACData

log = logging.getLogger(__name__)

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


class NotDisordered(Select):
    """Selection class to remove disordered atoms and hetatoms"""

    # Exclude disordered atoms.
    def accept_atom(self, atom):
        """
        Check whether an atom is disordered or has AltLocs
        """
        return (
            not atom.is_disordered()
            or atom.get_altloc() == "A"
            or atom.get_altloc() == "1"
        )

    # Exclude hetatoms
    def accept_residue(self, residue):
        """Accepts residue if not a cofactor"""
        return not self.is_cofactor(residue)

    @staticmethod
    def is_het(residue):
        """Checks if a residue is a hetatom"""
        res = residue.id[0]
        return res != " " and res != "W"

    @staticmethod
    def is_cofactor(residue):
        """Checks if a residue is not in 20 standard AAs"""
        return not residue.get_resname() in PROTEIN_LETTERS  # ["FAD", "ANP"]


def find_modified_amino_acids(path) -> set:
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, "r"):
        if line[:6] == "SEQRES":
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(infilename: str, outfilename: str, chain_ids=None):
    """
    Extract the chain_ids from infilename and save in outfilename.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")

    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if chain_ids == None or chain.get_id() in chain_ids:
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())


def protonate(in_pdb_file: Path, out_pdb_file: Path):
    """
    Protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    :param in_pdb_file: file to protonate.
    :param out_pdb_file: output file where to save the protonated pdb file.
    """

    # Remove protons first, in case the structure is already protonated
    log.info(f"Deprotonating {in_pdb_file}. Output: {out_pdb_file}")
    args = ["./reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode("utf-8").rstrip())
    outfile.close()

    # Now add them again.
    log.info(f"Protonating {out_pdb_file}. Output: {out_pdb_file}")
    args = ["./reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode("utf-8"))
    outfile.close()

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname: Path, center: bool) -> Dict[str, np.ndarray]:
    """
    Loads a .ply mesh to return a point cloud and connectivity.
    """
    # Load the data
    # print(fname)
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []

    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    # Extract coordinates
    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}
