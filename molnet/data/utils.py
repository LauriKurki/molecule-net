import functools
import jax
import jax.numpy as jnp
import numpy as np

import h5py
import ase

from typing import List, Tuple, Optional

def _pad_xyzs(xyz, max_len):
    xyz_padded = np.zeros((max_len, 5))
    xyz_padded[:xyz.shape[0], :] = xyz
    return xyz_padded

###@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _create_one_atom_position_map(
        xyz: jnp.ndarray,
        sw: jnp.ndarray,
        sw_size: float = 16.0,
        z_cutoff: float = 2.0,
        map_resolution: float = 0.125,
        sigma: float = 0.2,
) -> jnp.ndarray:
    """
    Creates a three dimensional map of the atom positions. Atomic positions
    represented with Gaussian functions.

    Args:
        xyz: `jnp.ndarray` of shape (N, 5) where N is the number of atoms.
        sw: `jnp.ndarray` of shape (2, 3). Scan window.
        sw_size: float. The size of the scan window in Ångströms.
        z_cutoff: float. Where to cutoff atoms.
        map_resolution: float. The resolution of the map in Angstroms.
        sigma: float. The standard deviation of the Gaussian function.

    Returns:
        `jnp.ndarray` of shape (map_size/map_resolution, map_size/map_resolution). The atom position map. 
    """
    xyz = jnp.where(xyz[:, [2]] > xyz[:, [2]].max() - z_cutoff, xyz, -jnp.inf)

    x = jnp.linspace(sw[0,0], sw[1,0], 128)
    y = jnp.linspace(sw[0,1], sw[1,1], 128)
    z = jnp.arange(-z_cutoff, 1e-9, 0.1)
    X, Y, Z = jnp.meshgrid(x, y, z)

    maps = jax.vmap(
        lambda atom: jnp.exp(
            -((X - atom[0]) ** 2 + (Y - atom[1]) ** 2 + (Z - atom[2]) ** 2) / (2*sigma**2)
        )
    )(xyz)

    return maps.sum(axis=0)


def get_image(fname, index, split='train'):
    with h5py.File(fname, 'r') as f:
        x = f[split]['X'][index, 0].transpose(1, 0, 2)
        sw = f[split]['sw'][index, 0]
        xyz = f[split]['xyz'][index]

    def _unpad(xyz):
        return xyz[xyz[:, -1] > 0]
    
    def _top_to_zero(xyz):
        xyz[:, 2] = xyz[:, 2] - xyz[:, 2].max()
        return xyz
    
    xyz = _top_to_zero(_unpad(xyz))

    return x, sw, xyz


def get_image_and_atom_map(
    fname,
    index,
    atomic_numbers,
    split='train',
    z_cutoff=2.0,
    map_resolution=0.125,
    sigma=0.2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, sw, xyz = get_image(fname, index, split)

    # Check if all Zs are in 'atomic_numbers'
    zs = xyz[:, -1].astype(int)
    if ~jnp.all(jnp.isin(zs, atomic_numbers)):
        return None

    scan_window_size = jnp.ceil(sw[1,0] - sw[0,0])

    def filter_by_species(sp):
        # Select rows where the atomic species matches
        return jnp.where(jnp.isin(xyz[:, -1], sp)[:, None], xyz[:, :3], jnp.zeros_like(xyz[:, :3]))

    # Vectorize the filtering function across all species
    species_matrices = jax.vmap(filter_by_species)(atomic_numbers)

    atom_map = jax.vmap(
        lambda xyz: _create_one_atom_position_map(
            xyz,
            sw,
            scan_window_size,
            z_cutoff,
            map_resolution,
            sigma,
        )
    )(species_matrices)
    return x, atom_map, xyz


def atom_map_generator(
    fname: str,
    atomic_numbers: jnp.ndarray,
    batch_size: int,
    z_cutoff: float,
    map_resolution: float,
    sigma: float,
):
    """Generator that creates atom maps from a list of molecules."""

    # Gather a batch of molecules, images, and atom maps
    splits = ['train', 'val', 'test']
    for split in splits:

        with h5py.File(fname, 'r') as f:
            n_molecules = f[split]['X'].shape[0]

        xs = []
        atom_maps = []
        xyzs = []

        for i in range(n_molecules):
            res = get_image_and_atom_map(
                fname,
                i,
                atomic_numbers,
                split,
                z_cutoff,
                map_resolution,
                sigma,
            )

            if res is None:
                continue

            x, atom_map, xyz = res
            xs.append(x)
            atom_maps.append(atom_map)
            xyzs.append(_pad_xyzs(xyz, 54))

            if len(xs) == batch_size:
                yield jnp.stack(xs), jnp.stack(atom_maps), xyzs
                xs = []
                atom_maps = []
                xyzs = []

        if len(xs) > 0:
            yield jnp.stack(xs), jnp.stack(atom_maps), xyzs
            xs = []
            atom_maps = []
            xyzs = []

def get_split(
    i: int,
    split_lengths: dict
) -> str:
    """Get the split of the molecule."""
    if i < split_lengths['train']:
        return 'train'
    elif i < split_lengths['train'] + split_lengths['val']:
        return 'val'
    else:
        return 'test'
