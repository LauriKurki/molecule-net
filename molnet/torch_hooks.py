import os
import re
from glob import glob

from absl import logging

from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from molnet import train_torch
from molnet import torch_train_state
from molnet import graphics

from typing import Any, Dict, Callable, Tuple

@dataclass
class TrainMetricsLoggingHook:
    writer: SummaryWriter
    prefix: str = "train"

    def __call__(
        self,
        metrics: torchmetrics.Metric,
        step: int
    ):
        """Logs training metrics to tensorboard."""
        train_metrics = metrics.compute()

        self.writer.add_scalar(
            "Loss/train",
            train_metrics,
            step
        )
        logging.info(f"Step: {step}, train loss: {train_metrics:.2e}")
        metrics.reset()

        self.writer.flush()

        return metrics
    

@dataclass
class CheckpointHook:
    checkpoint_path: str
    max_to_keep: int

    def __init__(self, checkpoint_path: str, max_to_keep: int):
        self.checkpoint_path = checkpoint_path
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_path, exist_ok=True)

    def restore_or_init(self, model, optimizer) -> Tuple[int, float]:
        """
        Restore the model and optionally the optimizer state if a checkpoint exists.
        Return step number and validation loss.
        """
        checkpoints = self._get_existing_checkpoints()
        if len(checkpoints) > 0:
            latest_checkpoint = checkpoints[-1]

            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            validation_loss = checkpoint['validation_loss']

            logging.info(f"Restored from checkpoint: {latest_checkpoint}")
            return checkpoint.get('step', 0), validation_loss  # return the last saved step number
        else:
            logging.info("No checkpoint found. Initializing from scratch.")
            return 0, 0.0

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        validation_loss: float,
        best_state: Dict[str, Any]
    ):
        """
        Save the model and optimizer state to a checkpoint.
        """
        checkpoint_name = os.path.join(self.checkpoint_path, f"checkpoint_{step}.pth")
        
        # Save current state
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss,
        }, checkpoint_name)

        # Save the best state
        best_checkpoint_name = os.path.join(self.checkpoint_path, "checkpoint_best.pth")
        torch.save({
            "step": best_state.step_for_best_model,
            "model_state_dict": best_state.best_model,
            "optimizer_state_dict": best_state.best_optimizer,
            "validation_loss": best_state.metrics_for_best_model,
        }, best_checkpoint_name)

        logging.info(f"Saved checkpoint: {checkpoint_name}")
        self._cleanup_old_checkpoints()

    def _get_existing_checkpoints(self):
        """
        Get the list of existing checkpoint files sorted by epoch number.
        """
        checkpoints = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        checkpoints.sort(key=lambda x: x.split("_")[-1].split(".")[0])
        return checkpoints

    def _cleanup_old_checkpoints(self):
        """
        Delete older checkpoints to maintain only the latest `max_to_keep` checkpoints.
        """
        checkpoints = self._get_existing_checkpoints()
        if len(checkpoints) > self.max_to_keep:
            for ckpt in checkpoints[:-self.max_to_keep]:
                os.remove(ckpt)


@dataclass
class EvaluationHook:
    evaluate_model_fn: Callable
    writer: SummaryWriter

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        best_state: torch_train_state.State,
    ) -> Tuple[torch_train_state.State, float]:
        """
        Evaluate the model and log the metrics.
        """
        metrics = self.evaluate_model_fn(model)
        eval_metrics = metrics.compute()

        self.writer.add_scalar(
            "Loss/val",
            eval_metrics,
            step
        )

        self.writer.flush()

        try:
            min_val_loss = best_state.metrics_for_best_model
        except KeyError:
            logging.info("No best model found.")
            min_val_loss = float("inf")

        # If the current model is better than the best model, update the best model
        if metrics < min_val_loss:
            best_state.best_model = model.state_dict()
            best_state.best_optimizer = optimizer.state_dict()
            best_state.step_for_best_model = step
            best_state.metrics_for_best_model = metrics

            logging.info(f"New best model found at step {step}.")
            
        return best_state, eval_metrics


@dataclass
class PredictionHook:
    workdir: str
    predict_fn: Callable
    peak_threshold: float

    def __call__(
        self,
        model: torch.nn.Module,
        num_batches: int,
        step: int,
        final: bool,
    ):
        """
        Make predictions using the model and log the predictions.
        """
        # Create the output directory
        output_dir = os.path.join(
            self.workdir,
            "predictions",
            f"final_step_{step}" if final else f"step_{step}"
        )  
        os.makedirs(output_dir, exist_ok=True)

        # Predict on the test set
        (
            inputs, targets, preds, xyzs, losses
        ) = self.predict_fn(
            model,
            num_batches
        )

        # Transfer everything back to the CPU and numpy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        xyzs = xyzs.cpu().numpy()

        # transpose the inputs, targets and preds so that channel dimension is last
        inputs = inputs.transpose(0, 2, 3, 4, 1)
        targets = targets.transpose(0, 2, 3, 4, 1)
        preds = preds.transpose(0, 2, 3, 4, 1)

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

        assert inputs_summed.ndim == 4, inputs_summed.shape # [num_samples, nX, nY, 1]
        assert preds_summed.ndim == 4, preds_summed.shape # [num_samples, nX, nY, 1]
        assert targets_summed.ndim == 4, targets_summed.shape # [num_samples, nX, nY, 1]

        # Write detailed predictions
        graphics.save_predictions(
            inputs, targets, preds, losses, output_dir
        )

        graphics.save_simple_predictions(
            inputs_summed, targets_summed, preds_summed, output_dir
        )

        graphics.save_predictions_as_molecules(
            inputs, targets, preds, xyzs, output_dir, peak_threshold=self.peak_threshold
        )
