import os
import numpy as np

import ctypes
from ctypes import c_int, c_float, POINTER


this_dir = os.path.dirname(os.path.abspath(__file__))
clib = ctypes.CDLL(os.path.join(this_dir, "peaks_lib.so"))

fp_p = POINTER(c_float)
int_p = POINTER(c_int)

# fmt: off
clib.peak_dist.argtypes = [
    c_int, c_int, c_int, c_int, fp_p,
    int_p, fp_p,
    fp_p, fp_p, c_float
] # fmt: on

# Define argument types for the new function
clib.peak_dist_species.argtypes = [
    c_int, c_int, c_int, c_int, fp_p,
    int_p, fp_p,
    fp_p, fp_p, c_float
]


def peak_dist(atoms, n_xyz, xyz_start, xyz_step, std):
    nb = len(atoms)
    nx, ny, nz = n_xyz
    dist = np.empty([nb, nx, ny, nz], dtype=np.float32)
    dist_c = dist.ctypes.data_as(fp_p)
    N_atom = np.array([len(a) for a in atoms], dtype=np.int32).ctypes.data_as(int_p)
    pos = np.concatenate(atoms, axis=0).astype(np.float32).ctypes.data_as(fp_p)
    xyz_start = np.array(xyz_start, dtype=np.float32).ctypes.data_as(fp_p)
    xyz_step = np.array(xyz_step, dtype=np.float32).ctypes.data_as(fp_p)

    # fmt: off
    clib.peak_dist(
        nb, nx, ny, nz, dist_c,
        N_atom, pos,
        xyz_start, xyz_step, std
    ) # fmt: on

    return dist


def peak_dist_species(atoms, n_xyz, xyz_start, xyz_step, std, species=[1, 6, 7, 8, 9]):
    nb = len(atoms)
    nx, ny, nz = n_xyz
    num_species = len(species)
    dist_species = np.empty([nb, num_species, nx, ny, nz], dtype=np.float32)
    dist_c = dist_species.ctypes.data_as(fp_p)
    N_atom = np.array([len(a) for a in atoms], dtype=np.int32).ctypes.data_as(int_p)
    pos = np.concatenate(atoms, axis=0).astype(np.float32).ctypes.data_as(fp_p)
    xyz_start = np.array(xyz_start, dtype=np.float32).ctypes.data_as(fp_p)
    xyz_step = np.array(xyz_step, dtype=np.float32).ctypes.data_as(fp_p)

    clib.peak_dist_species(
        nb, nx, ny, nz, dist_c,
        N_atom, pos,
        xyz_start, xyz_step, std
    )
    return dist_species