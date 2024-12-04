import os
import functools

from absl import logging
from absl import app
from absl import flags

import tqdm
import tqdm.contrib.concurrent
from multiprocessing import Manager

import pickle
import scipy
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from ppafm import io

from typing import List, Tuple, Dict

elements = ['H' , 'He',
            'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 
            'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',
             'K', 'Ca', 
            'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr',
             'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'In', 'Sn', 'Sb', 'Te',  'I', 'Xe'
]

FLAGS = flags.FLAGS

def get_total_number_of_keys_in_dict(d):
    total = 0
    for k in d:
        total += len(d[k])
    return total


# https://stackoverflow.com/a/8453514
def random_unit_vector():
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    return vec

def plane_from_3points(points):
    a = points[1] - points[0]
    b = points[2] - points[0]
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    d = -np.dot(n, points[0])
    return np.array([n[0], n[1], n[2], d])

def plane_from_2points(points):
    p0 = points[0] + np.array([5, 5, 5])
    a = points[0] - p0
    b = points[1] - p0
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    d = -np.dot(n, points[0])
    return np.array([n[0], n[1], n[2], d])

def zyz_rotation(alpha, beta, gamma):
    '''
    Extrinsic rotation of angle alpha around z, beta around y, and gamma around z.
    '''

    def R_y(x):
        return [[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]]
    def R_z(x):
        return [[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]]

    return np.dot(R_z(gamma), np.dot(R_y(beta), R_z(alpha)))

def cart_to_sph(vec):
    r = np.linalg.norm(vec)
    phi = np.arctan2(vec[1], vec[0])
    theta = np.arccos(vec[2]/r)
    return r, phi, theta

def get_convex_hull_eqs(xyz, angle_tolerance=5):
    '''
    Get coefficients for equations of planes of convex hull of point cloud. If plane normals
    are within angle_tolerance, they are considered to be the same plane and the other plane
    is ignored.
    '''
    xyz = xyz[:,:3]
    hull = ConvexHull(xyz)
    cosines = 1 - pdist(hull.equations[:,:3], 'cosine')
    angles = np.arccos(cosines) / np.pi * 180
    angles = squareform(angles)
    bad_inds = []
    for i, angle in enumerate(angles):
        if i in bad_inds:
            continue
        inds = np.where(angle < angle_tolerance)[0]
        for ind in inds:
            if ind == i:
                continue
            if ind not in bad_inds:
                bad_inds.append(ind)
    eqs = np.delete(hull.equations, bad_inds, axis=0)
    return eqs, hull

def find_planar_segments(xyz, eqs, dist_tol=0.2, num_atoms=10):
    '''
    Find planar segments on the surface of a molecule. A planar segment is
    a surface plane that contains at least num_atoms atoms within dist_tol of
    the plane.
    '''
    planar_seg_eqs = []
    planar_seg_inds = []
    for i, eq in enumerate(eqs):
        eq = eq / np.linalg.norm(eq[:3])
        dist = np.abs(np.dot(xyz[:,:3], eq[:3]) + eq[-1])
        if len(np.where(dist <= dist_tol)[0]) >= num_atoms:
            planar_seg_eqs.append(eq)
            planar_seg_inds.append(i)
    return planar_seg_eqs, planar_seg_inds

def get_plane_elements(xyz, plane_eqs, dist_tol=0.5):
    '''
    Find and count the number of atoms of different elements in a molecule near planes.
    '''
    plane_elems = []
    for eq in plane_eqs:
        eq = eq / np.linalg.norm(eq[:3])
        dist = np.abs(np.dot(xyz[:,:3], eq[:3]) + eq[-1])
        plane_elems.append(set(xyz[dist <= dist_tol, -1].astype(int)))
    return plane_elems

def _convert_elemements(element_dict):
    element_dict_ = {}
    for i, e in enumerate(element_dict):
        if isinstance(e, str):
            element_dict_[elements.index(e)+1] = element_dict[e]
        else:
            element_dict_[e] = element_dict[e]
    return element_dict_

def choose_rotations_bias(xyz, flat=True, plane_bias={'F': 1, 'Cl': 0.8, 'Br': 1},
    random_bias={'F': 0.5, 'Cl': 0.4, 'Br': 2}, angle_tolerance=5, elem_dist_tol=0.7, 
    flat_dist_tol=0.1, flat_num_atoms=10):

    n_vecs = []

    plane_bias = _convert_elemements(plane_bias)
    random_bias = _convert_elemements(random_bias)
    
    if len(xyz) > 3:
        try:
            eqs, hull = get_convex_hull_eqs(xyz, angle_tolerance=angle_tolerance)
            vertices = hull.vertices
        except scipy.spatial.qhull.QhullError:
            print(f'A problematic molecule encountered.')
            return []
    elif len(xyz) == 3:
        eqs = plane_from_3points(xyz[:,:3])[None]
        vertices = np.array([0, 1, 2])
    elif len(xyz) == 2:
        eqs = plane_from_2points(xyz[:,:3])[None]
        vertices = np.array([0, 1])
    else:
        print(xyz)
        raise RuntimeError('Molecule with less than two atoms.')
    
    if flat:
        planar_seg_eqs, planar_seg_inds = find_planar_segments(xyz, eqs, dist_tol=flat_dist_tol, num_atoms=flat_num_atoms)
        for eq in planar_seg_eqs:
            n_vecs.append(eq[:3])
        eqs = np.delete(eqs, planar_seg_inds, axis=0)

    if plane_bias:
        plane_elems = get_plane_elements(xyz, eqs, dist_tol=elem_dist_tol)
        for eq, elems in zip(eqs, plane_elems):
            for e, p in plane_bias.items():
                if e in elems and (np.random.rand() <= p):
                    n_vecs.append(eq[:3])
                    break

    if random_bias:
        elems = set(xyz[vertices,-1].astype(int))
        for e in random_bias:
            if e not in elems:
                continue
            while random_bias[e] > 0:
                if random_bias[e] < 1 and np.random.rand() > random_bias[e]:
                    break
                while True:
                    n = random_unit_vector()
                    _, phi, theta = cart_to_sph(n)
                    new_xyz = xyz.copy()
                    new_xyz[:,:3] = np.dot(new_xyz[:,:3], zyz_rotation(-phi, -theta, 0).T)
                    eq = np.array([0, 0, 1, -new_xyz[:,2].max()])
                    elems = get_plane_elements(new_xyz, [eq], dist_tol=0.7)
                    if e in elems[0]:
                        break
                if len(n_vecs) > 0:
                    n_vecs_np = np.stack(n_vecs, axis=0)
                    angles = np.arccos(np.dot(n_vecs_np, n)/np.linalg.norm(n_vecs_np, axis=1)) / np.pi * 180
                    if all(angles > angle_tolerance):
                        n_vecs.append(n)
                        random_bias[e] -= 1
                else:
                    n_vecs.append(n)
                    random_bias[e] -= 1

    rotations = []
    for vec in n_vecs:
        _, phi, theta = cart_to_sph(vec)
        rotations.append(zyz_rotation(-phi, -theta, 0))

    return rotations

def return_rotations(
    shared_rotations: Dict[str, List[np.ndarray]],
    filenames: List[str],
    valid_elements: List[str],
    flat: bool = True,
    plane_bias: dict = {'F': 1, 'Cl': 0.8, 'Br': 1},
    random_bias: dict = {'F': 0.5, 'Cl': 0.4, 'Br': 2},
    angle_tolerance: float = 5,
    elem_dist_tol: float = 0.7,
    flat_dist_tol: float = 0.1,
    flat_num_atoms: int = 10
) -> List[np.ndarray]:
    for filename in tqdm.tqdm(filenames):
        xyz, zs, qs, comment = io.loadXYZ(filename)
        cid = comment.split()[-1]
        if np.any(~np.isin(zs, valid_elements)):
            continue
        rots = choose_rotations_bias(
            xyz,
            flat=flat,
            plane_bias=plane_bias,
            random_bias=random_bias,
            angle_tolerance=angle_tolerance,
            elem_dist_tol=elem_dist_tol,
            flat_dist_tol=flat_dist_tol,
            flat_num_atoms=flat_num_atoms
        )
        if len(rots) == 0:
            continue
        shared_rotations[cid] = rots

def rotations_wrapper(shared_rotations, args):
    return return_rotations(shared_rotations, *args)

def main(argv):

    # Define the path to the database and the save directory
    save_path = FLAGS.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    database_path = FLAGS.database_path
    filenames = [
        os.path.join(database_path, f)
        for f in os.listdir(database_path)
        if f.endswith('.xyz')
    ]

    # Define the parameters for the rotation generation
    flat = True
    flat_dist_tol = FLAGS.flat_dist_tol
    elem_dist_tol = FLAGS.elem_dist_tol
    angle_tolerance = FLAGS.angle_tolerance
    plane_bias = {
        'H' : 0.0,
        'C' : 0.0,
        'N' : 0.1,
        'O' : 0.0,
        'F' : 0.5,
    }
    random_bias = {
        'H' : 0.1,
        'C' : 0.0,
        'N' : 0.4,
        'O' : 0.1,
        'F' : 0.5,
    }
    flat_num_atoms = FLAGS.flat_num_atoms
    valid_elements = np.array([1, 6, 7, 8, 9])

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Define the number of workers
    num_workers = FLAGS.num_workers

    # Create argument list for the parallel processing
    args_list = [
        (filenames[i::num_workers], valid_elements, flat, plane_bias, random_bias, angle_tolerance, elem_dist_tol, flat_dist_tol, flat_num_atoms)
        for i in range(num_workers)
    ]

    # Create a shared list for the rotations
    manager = Manager()
    rotations = manager.dict()

    # Run the parallel processing
    tqdm.contrib.concurrent.process_map(
        functools.partial(rotations_wrapper, rotations),
        args_list,
        max_workers=num_workers
    )

    total_rotations = get_total_number_of_keys_in_dict(rotations)
    logging.info(f"Number of rotations created: {total_rotations}")

    # Save the rotations
    with open(save_path, 'wb') as f:
        pickle.dump(rotations, f)


if __name__ == "__main__":
    flags.DEFINE_string("database_path", None, "Path to the database.")
    flags.DEFINE_string("save_path", None, "Path to the save directory.")
    flags.DEFINE_float("angle_tolerance", 5, "Angle tolerance.")
    flags.DEFINE_float("elem_dist_tol", 0.7, "Element distance tolerance.")
    flags.DEFINE_float("flat_dist_tol", 0.2, "Flat distance tolerance.")
    flags.DEFINE_integer("flat_num_atoms", 6, "Flat number of atoms.")

    flags.DEFINE_integer("num_workers", 1, "Number of workers.")

    flags.mark_flags_as_required(["database_path", "save_path"])

    app.run(main)
