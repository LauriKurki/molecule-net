import ase
from ase import data

from typing import List, Dict

def compute_pairwise_distances(
    molecules: List[ase.Atoms]
) -> Dict[List[float]]:
    """
    Computes pairwise distances between all bonded atoms in all molecules.
    Returns a list of distances for each bond type.

    Args:
    - molecules `List[ase.Atoms]`: a list of molecules

    Returns:
    - distances `Dict[List[float]]`: a list of distances for each bond type
    """

    distances = {}
    for mol in molecules:
        for bond in mol.get_bonds():
            distances.setdefault(bond[0].symbol + bond[1].symbol, []).append(
                bond[0].distance(bond[1])
            )