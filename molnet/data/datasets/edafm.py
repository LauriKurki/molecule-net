import h5py
import numpy as np
import jax.numpy as jnp
import ase

from typing import List, Tuple, Optional

import tqdm

def get_length(h5_file: str) -> int:
    """Get the length of the EDAFM dataset."""
    length = 0
    splits = ['train', 'val', 'test']
    split_lengths = {}
    with h5py.File(h5_file, 'r') as h5:
        for split in splits:
            l = h5[split]['xyz'].shape[0]
            length += l
            split_lengths[split] = l
    return length, split_lengths

def load_edafm(
    h5_file: str,
) -> Tuple[List[ase.Atoms], List[Tuple[str, int]]]:
    """Get the EDAFM dataset."""
    
    atomic_numbers = jnp.array([1, 6, 7, 8, 9])

    molecules = []
    indices = []
    splits = ['train', 'val', 'test']
    with h5py.File(h5_file, 'r') as h5:
        for split in splits:
            xyz_all = h5[split]['xyz']
            n_molecules = xyz_all.shape[0]

            for i in tqdm.tqdm(range(n_molecules)):
                xyz = xyz_all[i]
                xyz = xyz[xyz[:, 4] > 0] # Remove padding atoms

                # Check if there are any atoms left
                if len(xyz) == 0:
                    continue

                Zs = xyz[:, -1].astype(int)

                # Check if all Zs are in 'atomic_numbers'
                if ~jnp.all(jnp.isin(Zs, atomic_numbers)):
                    continue

                atoms = ase.Atoms(
                    positions=xyz[:, :3],
                    numbers=Zs,
                )

                molecules.append(atoms)
                indices.append((split, i))

                if i > 100:
                    break # TODO: Remove this line

    return molecules, indices

def get_valid_indices(
    h5_file: str,
    atomic_numbers: jnp.ndarray,
) -> List[Tuple[str, int]]:
    """Get the indices of molecules that have all atoms in 'atomic_numbers'."""
    valid_indices = []
    splits = ['train', 'val', 'test']
    with h5py.File(h5_file, 'r') as h5:
        for split in splits:
            xyz_all = h5[split]['xyz']
            n_molecules = xyz_all.shape[0]

            for i in tqdm.tqdm(range(n_molecules)):
                xyz = xyz_all[i]
                xyz = xyz[xyz[:, 4] > 0] # Remove padding atoms

                # Check if there are any atoms left
                if len(xyz) == 0:
                    continue

                Zs = xyz[:, -1].astype(int)

                # Check if all Zs are in 'atomic_numbers'
                if np.all(np.isin(Zs, atomic_numbers)):
                    valid_indices.append((split, i))

    return valid_indices