import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import tqdm
import tqdm.contrib.concurrent

import tensorflow as tf
import numpy as np

from absl import logging
from absl import app
from absl import flags

from typing import List, Tuple

FLAGS = flags.FLAGS


from ppafm import io
from ppafm.ml.Generator import GeneratorAFMtrainer
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env
init_env(i_platform=0)


class Trainer(GeneratorAFMtrainer):
    def on_sample_start(self):
        self.randomize_distance(delta=0.0)
        self.randomize_tip(max_tilt=0.3)


def create_afmulator():
    afm = AFMulator(
        pixPerAngstrome=8,
        scan_dim=(128, 128, 19),
        scan_window=((0., 0., 5.), (15.9, 15.9, 6.9)),
        iZPP=8,
        QZs=[0.1, 0.0, -0.1, 0.0],
        Qs=[-10, 20, -10, 0],
        df_steps=10,
        npbc=(0, 0, 0)
    )
    return afm


def check_molecule(molecule_id, atomic_species):
    """Check if a molecule is valid."""
    xyz, Zs, qs, comment = io.loadXYZ(os.path.join(FLAGS.molecule_dir, f"{molecule_id}.xyz"))
    # Return true if all Zs are in `atomic_species`
    return np.all(np.isin(Zs, atomic_species))


def flatten_rotations(rots, atomic_species=np.array([1, 6, 7, 8, 9])):
    """Flatten the rotations dictionary and filter out invalid molecules."""
    rotations_flat = []
    for split, molecules in rots.items():
        for molecule_id in molecules:
                molecule_is_valid = check_molecule(molecule_id, atomic_species)
                if not molecule_is_valid:
                    continue
                rotations = molecules[molecule_id]
                for rotation in rotations:
                    rotations_flat.append((molecule_id, rotation))

    return rotations_flat


def generate_afms(
    molecule_dir: str,
    rotations: List[Tuple[str, int]],
    afmulator: AFMulator,
    batch_size: int,
    start: int,
    end: int,
    output_dir: str,
) -> None:
    """Generate AFMs for a chunk in the dataset."""

    logging.info(f"Saving to {output_dir}")

    signature = {
        "x": tf.TensorSpec(shape=(128, 128, 10), dtype=tf.float32),
        "xyz": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        "sw": tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
    }

    # Create the sample generator
    def sample_generator():
        for i in range(start, end):
            molecule_id, rotation = rotations[i]
            xyz, Zs, qs, comment = io.loadXYZ(os.path.join(molecule_dir, f"{molecule_id}.xyz"))

            xyz_center = xyz.mean(axis=0)
            xyzs_rot = np.dot(xyz - xyz_center, rotation) + xyz_center

            yield {
                "xyzs": xyzs_rot,
                "Zs": Zs,
                "qs": qs,
                "rot": np.eye(3),
            }

    # Create the AFM generator
    auxmaps = [] # No auxmaps
    trainer = Trainer(
        afmulator,
        auxmaps,
        sample_generator(),
        sim_type="LJ+PC",
        batch_size=batch_size,
        distAbove=2.1,
        Qs=[[-10, 20, -10, 0]],
        QZs=[[0.1, 0.0, -0.1, 0.0]],
    )

    def pad_xyzs(xyz, max_len):
        if len(xyz) < max_len:
            xyz = np.pad(xyz, ((0, max_len - len(xyz)), (0, 0)))
        return xyz

    # Create generator
    def generator():
        for batch in tqdm.tqdm(trainer, total=(end - start) // batch_size):
            Xs, Ys, xyzs, sws = batch
           
            # Pad the xyzs and stack
            xyzs = np.stack([pad_xyzs(xyz, 54) for xyz in xyzs])

            # Yield items in the batch one by one
            for i in range(len(Xs)):

                sample = {
                    "x": Xs[i, 0],
                    "xyz": xyzs[i],
                    "sw": sws[i, 0],
                }
                yield sample

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save(output_dir)

def _generate_atom_maps_wrapper(args):
    return generate_afms(*args)

def main(argv) -> None:
    del argv

    # Hide the GPU from TensorFlow
    tf.config.set_visible_devices([], 'GPU')

    logging.set_verbosity(logging.INFO)

    # Read the rotations and flatten
    rotations = np.load(FLAGS.rotations_fname, allow_pickle=True)
    rotations = flatten_rotations(rotations)
    logging.info(f"Loaded {len(rotations)} unique rotations.")

    # Calculate dataset shapes
    if FLAGS.n_molecules is not None:
        n_mol = FLAGS.n_molecules
    else:
        n_mol = len(rotations)

    logging.info(f"Total length of the dataset: {n_mol}")

    # Create the AFMulator
    afmulator = create_afmulator()

    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Create the argument list for the generator
    args_list = (
        (
            FLAGS.molecule_dir,
            rotations,
            afmulator,
            FLAGS.batch_size,
            start,
            start+FLAGS.chunk_size,
            os.path.join(
                FLAGS.output_dir,
                f"afms_{start:06d}_{start + FLAGS.chunk_size:06d}",
            ),
        ) for start in range(0, n_mol, FLAGS.chunk_size)
    )

    # Generate the AFMs
    for args in args_list:
        generate_afms(*args)


if __name__ == "__main__":
    flags.DEFINE_string("molecule_dir", "/l/mol_database", "Path to the molecule database.")
    flags.DEFINE_string("rotations_fname", "/l/rotations_210611.pickle", "Path to the rotations file.")
    flags.DEFINE_string("output_dir", "/l/data/molnet/afms/", "Path to the output file.")
    flags.DEFINE_integer("chunk_size", 1000, "Number of molecules to process in each chunk.")
    flags.DEFINE_integer("batch_size", 20, "Batch size for the generator.")
    flags.DEFINE_integer("n_molecules", None, "Number of molecules to process in total.")

    app.run(main)
