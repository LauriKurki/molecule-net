import os
from absl import app, flags, logging

import jax
import jax.numpy as jnp
from flax.training import train_state
import tensorflow as tf

import ase
from ase import io
import numpy as np
from skimage import feature

from molnet import graphics
from molnet.data import input_pipeline_online
from analyses.utils import load_from_workdir

from typing import Optional, Dict, Tuple, List

FLAGS = flags.FLAGS

INDEX_TO_ELEM = {
    0: 'H',
    1: 'C', 2: 'N', 3: 'O', 4: 'F',
    5: 'Si', 6: 'P', 7: 'S', 8: 'Cl',
    9: 'Br'}

@jax.jit
def pred_fn(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List, jnp.ndarray]:
    """
    Predict the atom maps for a batch of inputs.

    Args:
    - state `TrainState`: the evaluation state
    - batch `Dict[str, jnp.ndarray]`: a batch of inputs

    Returns:
    - inputs `jnp.ndarray`: the input images
    - targets `jnp.ndarray`: the target atom maps
    - preds `jnp.ndarray`: the predicted atom maps
    - xyzs `jnp.ndarray`: the (padded) xyz coordinates
    """

    inputs, targets, xyzs = batch['images'], batch['atom_map'], batch['xyz']
    preds = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
        training=False,
    )

    return inputs, targets, preds, xyzs

def image_to_molecule(
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    outdir: str,
    scan_dim: np.ndarray = np.array([16, 16, 1]),
    z_cutoff: float = 1.0,
    peak_threshold: float = 0.5,
    start_save_idx: int = 0,
) -> None:

    n_samples = targets.shape[0]

    for sample in range(n_samples):
        target = targets[sample]
        pred = preds[sample]

        # Flip pred and target z axes
        target = target[..., ::-1, :]
        pred = pred[..., ::-1, :]

        # Find peaks in the target
        target_peaks = feature.peak_local_max(
            target,
            min_distance=5,
            exclude_border=0,
            threshold_rel=peak_threshold
        )

        # Find peaks in the prediction
        pred_peaks = feature.peak_local_max(
            pred,
            min_distance=5,
            exclude_border=0,
            threshold_rel=peak_threshold
        )

        # Convert peaks to xyz coordinates
        target_xyz_from_peaks = target_peaks[:, [1, 0, 2]] * scan_dim / target.shape[:3]
        target_elem_from_peaks = target_peaks[:, 3]

        pred_xyz_from_peaks = pred_peaks[:, [1, 0, 2]] * scan_dim / pred.shape[:3]
        pred_elem_from_peaks = pred_peaks[:, 3]

        # Create the target molecule in ASE
        target_mol = ase.Atoms(
            positions=target_xyz_from_peaks,
            symbols=[INDEX_TO_ELEM[elem] for elem in target_elem_from_peaks],
            cell=scan_dim
        )

        # Create the predicted molecule in ASE
        pred_mol = ase.Atoms(
            positions=pred_xyz_from_peaks,
            symbols=[INDEX_TO_ELEM[elem] for elem in pred_elem_from_peaks],
            cell=scan_dim
        )

        # Top to z_cutoff
        target_mol.positions[:, 2] -= target_mol.get_positions()[:, 2].max() - z_cutoff
        pred_mol.positions[:, 2] -= pred_mol.get_positions()[:, 2].max() - z_cutoff

        # Save the molecules
        target_mol.write(os.path.join(outdir, f"target_{sample+start_save_idx}.xyz"))
        pred_mol.write(os.path.join(outdir, f"pred_{sample+start_save_idx}.xyz"))


def predict_molecules(
    workdir: str,
    outputdir: str,
    num_batches: int,
    batch_size: Optional[int] = None,
    peak_threshold: Optional[float] = 0.5,
):
    """
    Make predictions for a model checkpoint.

    Args:
    - workdir `str`: the directory containing the model checkpoint
    - outputdir `str`: the directory to save the predictions
    - num_batches `int`: the number of batches to predict
    - batch_size `Optional[int]`: the batch size to use for prediction. Mainly for running on 
        local machine with less memory.
    - peak_threshold `Optional[float]`: the threshold (relative to max) for peak detection
    """
    # Create the output directory
    os.makedirs(outputdir, exist_ok=True)

    # Load the model
    state, config = load_from_workdir(workdir, return_attention=False)
    if batch_size is not None:
        config.batch_size = batch_size

    # Load the dataset
    dataset = input_pipeline_online.get_datasets(
        config
    )["val"]

    # Make predictions
    for i in range(num_batches):
        try: 
            batch = next(dataset)
        except StopIteration:
            print(f"Dataset ended")
            break
        _, targets, preds, _ = pred_fn(
            state,
            batch
        )

        # Compute z_cutoff from the z dimension of the target
        z_cutoff = targets.shape[-2] * 0.1 # 0.1 Å per pixel, e.g. 20 pixels = 2 Å

        # Translate the images to molecules
        image_to_molecule(
            targets,
            preds,
            outputdir,
            scan_dim=np.array([16., 16., z_cutoff]),
            z_cutoff=z_cutoff,
            peak_threshold=peak_threshold,
        )


def run_analysis(
    molecule_dir: str,
):
    """
    This function runs the analysis on the predicted and target molecules.

    Currently checks for:
    - pair-wise distances between atoms separated by atom type
    - the number of atoms in the molecules
    - distribution of atom types in the molecules
    """

    # Load the molecules
    preds = [
        io.read(os.path.join(molecule_dir, fname))
        for fname in os.listdir(molecule_dir)
        if "pred" in fname
    ]

    targets = [
        io.read(os.path.join(molecule_dir, fname))
        for fname in os.listdir(molecule_dir)
        if "target" in fname
    ]

    # Pair-wise distances
    # TODO

def main(argv):
    del argv

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info("Starting predictions")
    # Make predictions    
    predict_molecules(
        os.path.abspath(FLAGS.workdir),
        FLAGS.outputdir,
        FLAGS.num_batches,
        FLAGS.batch_size,
        FLAGS.peak_threshold,
    )
    logging.info("Predictions saved. Running analysis.")

    # Run the analysis
    run_analysis(FLAGS.outputdir)


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "The directory containing the model checkpoint.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "molecules"),
        "The directory to save the predictions."
    )
    flags.DEFINE_integer("num_batches", 1, "The number of batches to predict.")
    flags.DEFINE_integer("batch_size", None, "The batch size to use for prediction.")
    flags.DEFINE_float("peak_threshold", 0.5, "The threshold (relative to max) for peak detection.")
    flags.DEFINE_bool("old", False, "Whether to update the loaded config to the new style.")

    flags.mark_flags_as_required(["workdir"])

    app.run(main)
