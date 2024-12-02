import os

import numpy as np
import tensorflow as tf

import h5py

from absl import logging
from absl import app
from absl import flags
import tqdm
import tqdm.contrib.concurrent
from multiprocessing import Pool

from molnet.data import utils
from molnet.data.datasets import edafm

from typing import List, Tuple

FLAGS = flags.FLAGS


def generate_atom_maps(
    data_dir: str,
    indices: List[Tuple[str, int]],
    start: int,
    end: int,
    atomic_numbers: np.ndarray,
    z_cutoff: float,
    map_resolution: float,
    sigma: float,
    output_dir: str,
) -> np.ndarray:
    """Generate atom maps for a slice in the dataset."""

    logging.info(f"Generating fragments {start}:{end}")
    logging.info(f"Saving to {output_dir}")

    signature = {
        "images": tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float32),
        "xyz": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        "atom_map": tf.TensorSpec(shape=(len(atomic_numbers), 128, 128, 26), dtype=tf.float32)
    }

    def generator():
        for i in tqdm.tqdm(range(start, end)):

            split, index = indices[i]
            x, atom_map, xyz, = utils.get_image_and_atom_map_np(
                data_dir,
                atomic_numbers,
                index,
                split,
                z_cutoff,
                map_resolution,
                sigma,
            )
            if x is None:
                continue

            yield {
                "images": x.astype(np.float32),
                "xyz": xyz.astype(np.float32),
                "atom_map": atom_map.astype(np.float32),
            }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save(output_dir)


def _generate_atom_maps_wrapper(args):
    return generate_atom_maps(*args)

def main(args) -> None:
    logging.set_verbosity(logging.INFO)

    atomic_numbers = np.array([1, 6, 7, 8, 9])

    # get valid indices, i.e. indices of molecules that have all atoms
    # in the atomic_numbers list
    valid_indices = edafm.get_valid_indices(FLAGS.data_dir, atomic_numbers)

    # Calculate dataset shapes
    if FLAGS.n_molecules is not None:
        n_mol = FLAGS.n_molecules
    else:
        n_mol = len(valid_indices)
    logging.info(f"Total length of the dataset: {n_mol}")

    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Create a list of arguments to pass to "generate_atom_maps"
    args_list = [
        (
            FLAGS.data_dir,
            valid_indices,
            start,
            start+FLAGS.chunk_size,
            atomic_numbers,
            FLAGS.z_cutoff,
            FLAGS.map_resolution,
            FLAGS.sigma,
            os.path.join(
                FLAGS.output_dir,
                f"maps_{start:06d}_{start + FLAGS.chunk_size:06d}",
            ),
        ) for start in range(0, n_mol, FLAGS.chunk_size)
    ]

    tqdm.contrib.concurrent.process_map(
        _generate_atom_maps_wrapper, args_list, max_workers=FLAGS.num_workers
    )

    #for args in args_list:
    #    generate_atom_maps(*args)


if __name__=='__main__':
    flags.DEFINE_string('data_dir', '/l/data/small_fragments/afm.h5', 'Path to the dataset.')
    flags.DEFINE_string('output_dir', '/l/data/molnet', 'Path to the output file.')
    flags.DEFINE_integer('chunk_size', 1000, 'Chunk size.')
    flags.DEFINE_float('z_cutoff', 2.0, 'Z cutoff.')
    flags.DEFINE_float('map_resolution', 0.125, 'Map resolution.')
    flags.DEFINE_float('sigma', 0.2, 'Sigma.')
    flags.DEFINE_integer('n_molecules', None, 'Number of molecules to generate.')
    flags.DEFINE_integer('num_workers', 8, 'Number of workers.')

    app.run(main)
