import io

import ase

from rdkit.Chem import rdDetermineBonds
from rdkit import Chem

from typing import Sequence, List


def xyz_to_rdkit_molecule(molecules_file: str) -> Chem.Mol:
    """Converts a molecule from xyz format to an RDKit molecule."""
    mol = Chem.MolFromXYZFile(molecules_file)
    return Chem.Mol(mol)


def ase_to_rdkit_molecules(ase_mol: Sequence[ase.Atoms]) -> List[Chem.Mol]:
    """Converts molecules from ase format to RDKit molecules."""
    return [ase_to_rdkit_molecule(mol) for mol in ase_mol]


def ase_to_rdkit_molecule(ase_mol: ase.Atoms) -> Chem.Mol:
    """Converts a molecule from ase format to an RDKit molecule."""
    with io.StringIO() as f:
        ase.io.write(f, ase_mol, format="xyz")
        f.seek(0)
        xyz = f.read()
    mol = Chem.MolFromXYZBlock(xyz)
    return Chem.Mol(mol)


def check_molecule_validity(mol: Chem.Mol) -> bool:
    """Checks whether a molecule is valid using xyz2mol."""

    # We should only have one conformer.
    assert mol.GetNumConformers() == 1

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except ValueError:
        return False

    if mol.GetNumBonds() == 0:
        return False

    return True


def get_all_valid_molecules(molecules: Sequence[Chem.Mol]) -> List[Chem.Mol]:
    """Returns all valid molecules (with bonds inferred)."""
    return [mol for mol in molecules if check_molecule_validity(mol)]


def compute_validity(molecules: Sequence[Chem.Mol]) -> float:
    """Computes the fraction of molecules in a directory that are valid using xyz2mol ."""
    valid_molecules = get_all_valid_molecules(molecules)
    return len(valid_molecules) / len(molecules)
