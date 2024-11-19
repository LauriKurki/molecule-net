import os

import numpy as np

import webdataset as wds

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

    def generator():
        for i in tqdm.tqdm(range(start, end)):
            sample_key = start+i
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
                "__key__": f"{sample_key:06d}",
                "x.npy": x.astype(np.float16),
                "xyz.npy": xyz.astype(np.float16),
                "map.npy": atom_map.astype(np.float16),
            }

    with wds.TarWriter(output_dir) as sink:
        for item in generator():
            sink.write(item)


def _generate_atom_maps_wrapper(args):
    return generate_atom_maps(*args)


def main(args) -> None:
    logging.set_verbosity(logging.INFO)

    atomic_numbers = np.array([1, 6, 7, 8, 9])

    # get valid indices, i.e. indices of molecules that have all atoms
    # in the atomic_numbers list
    valid_indices = edafm.get_valid_indices(FLAGS.data_dir, atomic_numbers)

    # Calculate dataset shapes
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
                f"maps_{start:06d}_{start + FLAGS.chunk_size:06d}.tar",
            ),
        ) for start in range(0, n_mol, FLAGS.chunk_size)
    ]

    tqdm.contrib.concurrent.process_map(
        _generate_atom_maps_wrapper, args_list, max_workers=8
    )


if __name__=='__main__':
    flags.DEFINE_string('data_dir', '/l/data/small_fragments/afm.h5', 'Path to the dataset.')
    flags.DEFINE_string('output_dir', '/l/data/molnet', 'Path to the output file.')
    flags.DEFINE_integer('chunk_size', 1000, 'Chunk size.')
    flags.DEFINE_float('z_cutoff', 2.0, 'Z cutoff.')
    flags.DEFINE_float('map_resolution', 0.125, 'Map resolution.')
    flags.DEFINE_float('sigma', 0.2, 'Sigma.')
    
    app.run(main)
