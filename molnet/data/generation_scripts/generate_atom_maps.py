import os

import jax
import jax.numpy as jnp

import h5py

from absl import logging
from absl import app
from absl import flags
import tqdm

from molnet.data.utils import atom_map_generator
from molnet.data.datasets import edafm

FLAGS = flags.FLAGS


def main(args) -> None:
    logging.set_verbosity(logging.INFO)

    # Create a generator
    generator = atom_map_generator(
        FLAGS.data_dir,
        atomic_numbers=jnp.array([1, 6, 7, 8, 9]),
        batch_size=FLAGS.batch_size,
        z_cutoff=FLAGS.z_cutoff,
        map_resolution=FLAGS.map_resolution,
        sigma=FLAGS.sigma,
    )

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Get the first batch for shape calculation
    x, atom_map, xyz = next(generator)

    # Calculate dataset shapes
    n_mol = edafm.get_length(FLAGS.data_dir)
    max_mol_len = 54
    X_shape = (
        n_mol,
        x.shape[1],
        x.shape[2],
        x.shape[3],
    )
    Y_shape = (
        n_mol,
        atom_map.shape[1],
        atom_map.shape[2],
        atom_map.shape[3],
        atom_map.shape[4],
    )
    xyz_shape = (n_mol, max_mol_len, 5)

    # Create new group in HDF5 file and add datasets to the group

    with h5py.File(os.path.join(FLAGS.output_dir, 'atom_maps.h5'), 'w') as f:
        X_h5 = f.create_dataset('X', shape=X_shape, chunks=(1,)+X_shape[1:], dtype='f16')
        Y_h5 = f.create_dataset('Y', shape=Y_shape, chunks=(1,)+Y_shape[1:], dtype='f16')
        xyz_h5 = f.create_dataset('xyz', shape=xyz_shape, chunks=(1,)+xyz_shape[1:], dtype='f16')

        # Write the first batch
        X_h5[:FLAGS.batch_size] = x
        Y_h5[:FLAGS.batch_size] = atom_map
        xyz_h5[:FLAGS.batch_size] = xyz

        # Write the rest of the batches
        for i, (x, atom_map, xyz) in enumerate(tqdm.tqdm(generator, total=n_mol//FLAGS.batch_size)):
            X_h5[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] = x
            Y_h5[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] = atom_map
            xyz_h5[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] = xyz


if __name__=='__main__':
    flags.DEFINE_string('data_dir', '/l/data/small_fragments/afm.h5', 'Path to the dataset.')
    flags.DEFINE_string('output_dir', '/l/data/molnet', 'Path to the output file.')
    flags.DEFINE_integer('batch_size', 32, 'Batch size.')
    flags.DEFINE_float('z_cutoff', 2.0, 'Z cutoff.')
    flags.DEFINE_float('map_resolution', 0.125, 'Map resolution.')
    flags.DEFINE_float('sigma', 0.2, 'Sigma.')
    app.run(main)