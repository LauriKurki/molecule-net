import os
from absl import app, flags

import jax
import jax.numpy as jnp
from flax.training import train_state
import numpy as np

from molnet import graphics
from molnet.data import input_pipeline_wds
from analyses.utils import load_from_workdir

from typing import Optional, Dict, Tuple, List

FLAGS = flags.FLAGS

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
    - attention `List[jnp.ndarray]`: the attention maps
    - mse `jnp.ndarray`: the mean squared error
    """

    inputs, targets, xyzs = batch['images'], batch['atom_map'], batch['xyz']
    preds, attention = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
        training=False,
    )

    # mse
    mse = jnp.mean(jnp.square(targets - preds), axis=(1, 2, 3, 4))
    
    return inputs, targets, preds, attention, xyzs, mse


def make_predictions(
    workdir: str,
    outputdir: str,
    num_batches: int,
    batch_size: Optional[int] = None,
    peak_threshold: Optional[float] = 0.5,
    old: bool = False,
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
    - old `bool`: whether to update the loaded config to the new style
    """
    # Create the output directory
    os.makedirs(outputdir, exist_ok=True)

    # Load the model
    state, config = load_from_workdir(workdir, return_attention=True, old=old)
    if batch_size is not None:
        config.batch_size = batch_size

    # Load the dataset
    rng = jax.random.PRNGKey(0)
    datarng, rng = jax.random.split(rng)
    dataset = input_pipeline_wds.get_datasets(
        config
    )["val"]

    # Make predictions
    for i in range(num_batches):
        try: 
            batch = next(dataset)
        except StopIteration:
            print(f"Dataset ended")
            break
        inputs, targets, preds, attention_maps, xyzs, error = pred_fn(
            state,
            batch
        )
        
        # Save the predictions
        jnp.savez(
            os.path.join(outputdir, f"predictions_{i}.npz"),
            inputs=inputs,
            targets=targets,
            preds=preds,
            *{f"attention_{j}": attention for j, attention in enumerate(attention_maps)}
        )

        # Plot the attention maps
        graphics.save_attention_maps(
            inputs, attention_maps, outputdir, start_save_idx=i*config.batch_size
        )
        # Plot the predictions
        graphics.save_predictions(
            inputs, targets, preds, error, outputdir, start_save_idx=i*config.batch_size
        )

        # Compute z_cutoff from the z dimension of the target
        z_cutoff = targets.shape[-2] * 0.1 # 0.1 Å per pixel, e.g. 20 pixels = 2 Å
        # Plot the predictions as molecules
        graphics.save_predictions_as_molecules(
            inputs,
            targets,
            preds,
            xyzs,
            outputdir,
            scan_dim=np.array([16., 16., z_cutoff]),
            peak_threshold=peak_threshold,
            z_cutoff=z_cutoff,
            start_save_idx=i*config.batch_size
        )

        # Write predictions in simplified format (sum over heights and species)
        inputs_summed = inputs.sum(axis=(3, 4))[..., None]
        preds_summed = preds.sum(axis=(3, 4))[..., None]
        targets_summed = targets.sum(axis=(3, 4))[..., None]

        # scale everything to [0, 1] after shifting to positive values
        inputs_summed = inputs_summed - inputs_summed.min()
        preds_summed = preds_summed - preds_summed.min()
        targets_summed = targets_summed - targets_summed.min()

        inputs_summed = inputs_summed / inputs_summed.max()
        preds_summed = preds_summed / preds_summed.max()
        targets_summed = targets_summed / targets_summed.max()

        assert inputs_summed.ndim == 4, inputs_summed.shape
        assert preds_summed.ndim == 4, preds_summed.shape
        assert targets_summed.ndim == 4, targets_summed.shape

        # Plot the simple predictions
        graphics.save_simple_predictions(
            inputs_summed,
            preds_summed,
            targets_summed,
            outputdir,
            start_save_idx=i*config.batch_size
        )


def main(argv):
    del argv

    # Make predictions    
    make_predictions(
        os.path.abspath(FLAGS.workdir),
        FLAGS.outputdir,
        FLAGS.num_batches,
        FLAGS.batch_size,
        FLAGS.peak_threshold,
        FLAGS.old,
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "The directory containing the model checkpoint.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "The directory to save the predictions."
    )
    flags.DEFINE_integer("num_batches", 1, "The number of batches to predict.")
    flags.DEFINE_integer("batch_size", None, "The batch size to use for prediction.")
    flags.DEFINE_float("peak_threshold", 0.5, "The threshold (relative to max) for peak detection.")
    flags.DEFINE_bool("old", False, "Whether to update the loaded config to the new style.")

    flags.mark_flags_as_required(["workdir"])

    app.run(main)
