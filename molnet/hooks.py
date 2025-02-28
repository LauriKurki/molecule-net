import os
import pickle
from typing import Any, Dict, Callable

import jax
import jax.numpy as jnp
import numpy as np

from absl import logging

import flax
from dataclasses import dataclass
from clu import metric_writers, checkpoint, metrics
import flax.jax_utils

from molnet import train_state
from molnet import train
from molnet import graphics

@dataclass
class CheckpointHook:
    checkpoint_path: str
    max_to_keep: int

    def __init__(self, checkpoint_path: str, max_to_keep: int):
        self.checkpoint_path = checkpoint_path
        self.max_to_keep = max_to_keep
        self.ckpt = checkpoint.Checkpoint(
            self.checkpoint_path,
            max_to_keep=self.max_to_keep
        )

    def restore_or_init(
        self,
        state: train_state.TrainState
    ) -> train_state.TrainState:
        restored = self.ckpt.restore_or_initialize(
            {
                "state": state
            }
        )
        return restored["state"]
    
    def __call__(
        self,
        state: train_state.TrainState
    ):

        # Unreplicate from all devices
        #state = flax.jax_utils.unreplicate(state)

        # Save the state
        with open(
            os.path.join(self.checkpoint_path, f"params_{state.get_step()}.pkl"), "wb"
        ) as f:
            pickle.dump(state.params, f)

        # Save the best params
        with open(
            os.path.join(self.checkpoint_path, "params_best.pkl"), "wb"
        ) as f:
            pickle.dump(state.best_params, f)

        # Save the state
        self.ckpt.save(
            {
                "state": state
            }
        )


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}/{key}": val for key, val in result.items()}


@dataclass
class LogTrainingMetricsHook:
    writer: metric_writers.SummaryWriter
    prefix: str = "train"

    def __call__(
        self,
        train_metrics: metrics.Collection,
        step: int
    ):
        """Logs the training metrics and returns an empty metrics collection."""
        
        train_metrics = train_metrics.compute()

        self.writer.write_scalars(
            step,
            add_prefix_to_keys(train_metrics, self.prefix),
        )

        self.writer.flush()

        return train.Metrics.empty()


@dataclass
class EvaluationHook:
    evaluate_model_fn: Callable
    writer: metric_writers.SummaryWriter
    update_state_with_eval_metrics: bool = True

    def __call__(
        self,
        state: train_state.TrainState,
    ) -> train_state.TrainState:
        # Evaluate the model.
        eval_metrics = self.evaluate_model_fn(
            state,
        )

        # Compute and write metrics.
        for split in eval_metrics:
            eval_metrics[split] = eval_metrics[split].compute()
            self.writer.write_scalars(
                state.get_step(), add_prefix_to_keys(eval_metrics[split], split)
            )

        self.writer.flush()

        if not self.update_state_with_eval_metrics:
            return state

        # Note best state seen so far.
        # Best state is defined as the state with the lowest validation loss.
        try:
            min_val_loss = state.metrics_for_best_params["val_eval"]["total_loss"]
        except (AttributeError, KeyError):
            logging.info("No best state found yet.")
            min_val_loss = float("inf")

        if jnp.all(eval_metrics["val_eval"]["loss"] < min_val_loss):
            state = state.replace(
                best_params=state.params,
                metrics_for_best_params=eval_metrics,
                step_for_best_params=state.step,
            )
            logging.info("New best state found at step %d.", state.get_step())

        return state
    

@dataclass
class PredictionHook:
    workdir: str
    predict_fn: Callable
    peak_threshold: float
    preds_are_logits: bool
    writer: metric_writers.SummaryWriter

    def __call__(
        self,
        state: train_state.TrainState,
        num_batches: int,
        final: bool
    ):

        # Create the output directory
        output_dir = os.path.join(
            self.workdir,
            "predictions",
            f"final_step_{state.get_step()}" if final else f"step_{state.get_step()}"
        )  
        os.makedirs(output_dir, exist_ok=True)

        # Predict on the test set
        (
            inputs, targets, preds, xyzs, losses
        ) = self.predict_fn(
            state,
            num_batches
        )

        assert (
            inputs.shape[:-1] == preds.shape[:-1] == targets.shape[:-1]
        ), (
            inputs.shape,
            preds.shape,
            targets.shape,
        ) # [num_samples, nX, nY, nZ, num_species]

        # Get z_cutoff from the z dimension of the target
        z_cutoff = targets.shape[-2] * 0.1 # 0.1 Å per pixel, e.g. 20 pixels = 2 Å
        scan_dim = np.array([16., 16., z_cutoff])

        # Write predictions in simplified format (sum over heights and species)
        preds_summed = preds.sum(axis=(3, 4))[..., None]
        targets_summed = targets.sum(axis=(3, 4))[..., None]

        # scale everything to [0, 1] after shifting to positive values
        preds_summed = preds_summed - preds_summed.min()
        targets_summed = targets_summed - targets_summed.min()

        preds_summed = preds_summed / preds_summed.max()
        targets_summed = targets_summed / targets_summed.max()

        assert preds_summed.ndim == 4, preds_summed.shape # [num_samples, nX, nY, 1]
        assert targets_summed.ndim == 4, targets_summed.shape # [num_samples, nX, nY, 1]

        # Get last slice of the inputs
        inputs_summed_over_species = inputs.sum(axis=-1)
        inputs_close = inputs_summed_over_species[..., -1][..., None]

        assert inputs_close.ndim == 4, inputs_close.shape # [num_samples, nX, nY, 1]

        # Write detailed predictions
        graphics.save_predictions(
            inputs, targets, preds, losses, output_dir
        )

        graphics.save_simple_predictions(
            inputs_close, targets_summed, preds_summed, output_dir
        )

        graphics.save_predictions_as_molecules(
            inputs, targets, preds, xyzs, output_dir,
            peak_threshold=self.peak_threshold,
            scan_dim=scan_dim,
            z_cutoff=z_cutoff,
            preds_are_logits=self.preds_are_logits
        )

        # Disable writing of images for now
        #self.writer.write_images(
        #    state.get_step(),
        #    {
        #        "inputs": inputs_summed,
        #        "targets": targets_summed,
        #        "preds": preds_summed,
        #    },
        #)


@dataclass
class SegmantionPredictionHook:
    workdir: str
    predict_fn: Callable
    writer: metric_writers.SummaryWriter

    def __call__(
        self,
        state: train_state.TrainState,
        num_batches: int,
        final: bool
    ):

        # Create the output directory
        output_dir = os.path.join(
            self.workdir,
            "predictions",
            f"final_step_{state.get_step()}" if final else f"step_{state.get_step()}"
        )  
        os.makedirs(output_dir, exist_ok=True)

        # Predict on the test set
        (
            inputs, targets, preds, xyzs
        ) = self.predict_fn(
            state,
            num_batches
        )

        assert (
            inputs.shape[:-1] == preds.shape[:-1] == targets.shape
        ), (
            inputs.shape,
            preds.shape,
            targets.shape,
        )

        # Write detailed predictions
        graphics.save_segmentation_predictions(
            inputs, targets, preds, output_dir
        )