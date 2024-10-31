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

FLAGS = flags.FLAGS


def generate_atom_maps(
    data_dir: str,
    start: int,
    end: int,
    atomic_numbers: np.ndarray,
    split_lengths: dict,
    z_cutoff: float,
    map_resolution: float,
    sigma: float,
    output_dir: str,
) -> np.ndarray:
    """Generate atom maps for a slice in the dataset."""

    logging.info(f"Generating fragments {start}:{end}")
    logging.info(f"Saving to {output_dir}")

    signature = {
        "images": tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float16),
        "xyz": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        "atom_map": tf.TensorSpec(shape=(len(atomic_numbers), 128, 128, 21), dtype=tf.float16)
    }

    def generator():
        for i in tqdm.tqdm(range(start, end)):

            split = utils.get_split(i, split_lengths)
            x, atom_map, xyz, = utils.get_image_and_atom_map_np(
                data_dir,
                i,
                atomic_numbers,
                split,
                z_cutoff,
                map_resolution,
                sigma,
            )
            if x is None:
                continue

            yield {
                "images": x.astype(np.float16),
                "xyz": xyz.astype(np.float32),
                "atom_map": atom_map.astype(np.float16),
            }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save(output_dir)


def _generate_atom_maps_wrapper(args):
    return generate_atom_maps(*args)

def main(args) -> None:
    logging.set_verbosity(logging.INFO)

    atomic_numbers = np.array([1, 6, 7, 8, 9])

    # Calculate dataset shapes
    n_mol, split_lengths = edafm.get_length(FLAGS.data_dir)

    # Create new group in HDF5 file and add datasets to the group
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Create a list of arguments to pass to "generate_atom_maps"
    args_list = [
        (
            FLAGS.data_dir,
            start,
            start+FLAGS.chunk_size,
            atomic_numbers,
            split_lengths,
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
        _generate_atom_maps_wrapper, args_list
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
    
    app.run(main)
