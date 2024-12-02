import os
import jax
import jax.numpy as jnp
import numpy as np

import h5py

#from molnet.data._c.bindings import peak_dist_species

from typing import Dict, Tuple, List

import ctypes
from ctypes import c_int, c_float, POINTER

#this_dir = os.path.dirname(os.path.abspath(__file__))
#clib = ctypes.CDLL(os.path.join(this_dir, "_c", "peaks_lib.so"))

#fp_p = POINTER(c_float)
#int_p = POINTER(c_int)

# fmt: off
#clib.peak_dist.argtypes = [
#    c_int, c_int, c_int, c_int, fp_p,
#    int_p, fp_p,
#    fp_p, fp_p, c_float
#] # fmt: on


def compute_atom_maps(
    batch: Dict[str, np.ndarray],
    z_cutoff: float = 2.0,
    map_resolution: float = 0.125,
    sigma: float = 0.2,
) -> np.ndarray:
    """
    Compute atom maps for a molecule.

    Args:
        xyz: `np.ndarray` of shape (N, 5) where N is the number of atoms.
        sw: `np.ndarray` of shape (2, 3). Scan window.
        z_cutoff: float. Where to cutoff atoms.
        map_resolution: float. The resolution of the map in Angstroms.
        sigma: float. The standard deviation of the Gaussian function.

    Returns:
        `np.ndarray` of shape (n_species, 128, 128, z_cutoff/0.1). The atom maps.
    """

    # For each item in the batch, shift top atom to z=0
    # batch is read-only, so we need to copy it
    batch = {k: v.copy() for k, v in batch.items()}
    for i in range(batch["xyz"].shape[0]):
        batch["xyz"][i, :, 2] = batch["xyz"][i, :, 2] - batch["xyz"][i, :, 2].max()

    return peak_dist_species(
        batch["xyz"],
        (128, 128, int(z_cutoff / 0.1)),
        [batch["sw"][0, 0, 0], batch["sw"][0, 0, 1], -z_cutoff],
        [map_resolution, map_resolution, 0.1],
        sigma,
    )


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
    
    def _shift_scan_window(xyz, sw):
        xyz[:, :2] = xyz[:, :2] - sw[0, :2]
        return xyz

    # if all xyzs are zero, return None (padding)
    if np.all(xyz[:, -1] == 0):
        return None, None, None

    # Remove padding
    xyz = _unpad(xyz)

    # Shift z-coordinates so top is at zero
    xyz = _top_to_zero(xyz)

    # Shift xy-coordinates so scan window starts at zero
    xyz = _shift_scan_window(xyz, sw)

    # shift scan window
    sw = sw - sw[0]

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


def _create_one_atom_position_map_np(
        xyz: np.ndarray,
        z_max: float,
        sw: np.ndarray,
        sw_size: float = 16.0,
        z_cutoff: float = 2.0,
        map_resolution: float = 0.125,
        sigma: float = 0.2,
):
    """
    Creates a three dimensional map of the atom positions for one species.
    Atomic positions represented with Gaussian functions.

    Args:
        xyz: `jnp.ndarray` of shape (N, 5) where N is the number of atoms.
        z_max: float. The maximum z-coordinate of the molecule.
        sw: `jnp.ndarray` of shape (2, 3). Scan window.
        sw_size: float. The size of the scan window in Ångströms.
        z_cutoff: float. Where to cutoff atoms.
        map_resolution: float. The resolution of the map in Angstroms.
        sigma: float. The standard deviation of the Gaussian function.

    Returns:
        `jnp.ndarray` of shape (map_size/map_resolution, map_size/map_resolution). The atom position map. 
    """

    x = np.linspace(sw[0,0], sw[1,0], int(sw_size / map_resolution), dtype=np.float32)
    y = np.linspace(sw[0,1], sw[1,1], int(sw_size / map_resolution), dtype=np.float32)
    z = np.arange(z_max - z_cutoff, z_max+1e-9, 0.1, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z)

    if xyz.shape[0] == 0:
        return np.zeros_like(X)
    
    maps = np.zeros_like(X)

    for atom in xyz:
        maps += np.exp(
            -((X - atom[0]) ** 2 + (Y - atom[1]) ** 2 + (Z - atom[2]) ** 2) / (2*sigma**2)
        )

    return maps


def _create_all_atom_position_maps_cpp(
    xyz: List[np.ndarray],
    sw: np.ndarray,
    z_max: float,
    sw_size: float = 16.0,
    z_cutoff: float = 2.0,
    map_resolution: float = 0.125,
    sigma: float = 0.2,
):
    
    nb = len(xyz)
    nx = int(sw_size / map_resolution)
    ny = int(sw_size / map_resolution)
    nz = int(z_cutoff / 0.1)

    dist = np.empty([nb, nx, ny, nz], dtype=np.float32)
    dist_c = dist.ctypes.data_as(fp_p)

    N_atom = np.array([len(a) for a in xyz], dtype=np.int32).ctypes.data_as(int_p)
    pos = np.concatenate(xyz, axis=0).astype(np.float32).ctypes.data_as(fp_p)

    xyz_start = np.array([sw[0,0], sw[0,1], z_max - z_cutoff], dtype=np.float32).ctypes.data_as(fp_p)
    xyz_step = np.array([map_resolution, map_resolution, 0.1], dtype=np.float32).ctypes.data_as(fp_p)

    # fmt: off
    clib.peak_dist(
        nb, nx, ny, nz, dist_c,
        N_atom, pos,
        xyz_start, xyz_step, sigma
    ) # fmt: on

    return dist


def get_image_and_atom_map_cpp(
    fname,
    index,
    atomic_numbers,
    split='train',
    z_cutoff=2.0,
    map_resolution=0.125,
    sigma=0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, sw, xyz = get_image(fname, index, split)

    if x is None:
        return None, None, None

    z_max = xyz[:, 2].max()
    scan_window_size = np.ceil(sw[1,0] - sw[0,0])

    def filter_by_species(sp):
        # Create a boolean mask for rows matching the species
        mask = xyz[:, -1] == sp
        # Apply the mask to filter rows and select only position columns
        filtered_positions = np.where(mask[:, None], xyz[:, :3], np.zeros_like(xyz[:, :3])-np.inf)
        return filtered_positions

    # Apply filtering and store in a numpy array
    xyz_by_species = np.array([filter_by_species(sp) for sp in atomic_numbers])

    assert xyz_by_species.shape == (
        len(atomic_numbers), xyz.shape[0], 3
    ), (
        xyz_by_species.shape,
        len(atomic_numbers),
        xyz.shape[0],
        3,
    )

    atom_map = _create_all_atom_position_maps_cpp(
        xyz_by_species,
        sw,
        z_max,
        scan_window_size,
        z_cutoff,
        map_resolution,
        sigma,
    )

    return x, atom_map, xyz


def get_image_and_atom_map_np(
    fname,
    atomic_numbers,
    index,
    split='train',
    z_cutoff=2.0,
    map_resolution=0.125,
    sigma=0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, sw, xyz = get_image(fname, index, split)
    
    if x is None:
        return None, None, None

    z_max = xyz[:, 2].max()
    scan_window_size = np.ceil(sw[1,0] - sw[0,0])

    def filter_by_species(sp):
        # Create a boolean mask for rows matching the species
        mask = xyz[:, -1] == sp
        # Apply the mask to filter rows and select only position columns
        filtered_positions = np.where(mask[:, None], xyz[:, :3], np.zeros_like(xyz[:, :3])-np.inf)
        return filtered_positions

    # Apply filtering and store in a numpy array
    xyz_by_species = np.array([filter_by_species(sp) for sp in atomic_numbers])

    assert xyz_by_species.shape == (
        len(atomic_numbers), xyz.shape[0], 3
    ), (
        xyz_by_species.shape,
        len(atomic_numbers),
        xyz.shape[0],
        3,
    )

    atom_map = np.array([
        _create_one_atom_position_map_np(
            xyz,
            z_max,
            sw,
            scan_window_size,
            z_cutoff,
            map_resolution,
            sigma,
        )
        for xyz in xyz_by_species
    ])

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
            x, atom_map, xyz = get_image_and_atom_map_np(
                fname,
                i,
                atomic_numbers,
                split,
                z_cutoff,
                map_resolution,
                sigma,
            )

            if xyz is None:
                continue

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

def get_split_and_index(
    i: int,
    split_lengths: dict
) -> Tuple[int, str]:
    """Get the split of the molecule."""
    if i < split_lengths['train']:
        return i, 'train'
    elif i < split_lengths['train'] + split_lengths['val']:
        return i - split_lengths['train'], 'val'
    else:
        return i - split_lengths['train'] - split_lengths['val'], 'test'
