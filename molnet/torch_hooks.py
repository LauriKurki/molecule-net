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

from typing import Any, Dict, Callable

@dataclass
class TrainMetricsLoggingHook:
    writer: SummaryWriter
    prefix: str = "train"

    def __call__(
        self,
        train_metrics: torchmetrics.Metric,
        step: int
    ):
        """Logs training metrics to tensorboard."""
        train_metrics = train_metrics.compute()

        self.writer.add_scalars(
            self.prefix,
            {
                "mse": train_metrics,
            },
            step
        )

        self.writer.flush()

        return torchmetrics.MeanSquaredError().to(train_torch.device)
    

@dataclass
class CheckpointHook:
    checkpoint_path: str
    max_to_keep: int

    def __init__(self, checkpoint_path: str, max_to_keep: int):
        self.checkpoint_path = checkpoint_path
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_path, exist_ok=True)

    def restore_or_init(self, model, optimizer):
        """
        Restore the model and optionally the optimizer state if a checkpoint exists.
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
        validation_loss: float,
        step: int,
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
            "model_state_dict": best_state.best_model.state_dict(),
            "optimizer_state_dict": best_state.best_optimizer.state_dict(),
            "validation_loss": best_state.metrics_for_best_model,
        }, best_checkpoint_name)

        logging.info(f"Saved checkpoint: {checkpoint_name}")
        self._cleanup_old_checkpoints()

    def _get_existing_checkpoints(self):
        """
        Get the list of existing checkpoint files sorted by epoch number.
        """
        checkpoints = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        print(checkpoints)
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
    ):
        """
        Evaluate the model and log the metrics.
        """
        metrics = self.evaluate_model_fn(model)
        metrics = metrics.compute()

        self.writer.add_scalars(
            "eval",
            {
                "mse": metrics,
            },
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
            
        return best_state, metrics
