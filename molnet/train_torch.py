import os
import torchmetrics.regression
import yaml
import time
from dataclasses import dataclass

from absl import logging

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchmetrics
import ml_collections

from molnet.torch_models import create_model
from molnet.data import input_pipeline_wds
from molnet import torch_utils
from molnet import torch_hooks
from molnet import torch_train_state

from typing import Tuple

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = torch.device("cpu")


def predict_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int
):

    inputs = []
    predictions = []
    targets = []
    losses = []
    xyzs = []

    with torch.no_grad():
        model.eval()
        for i in range(num_batches):
            batch = next(dataloader)
            batch = [b.to(device) for b in batch]
            x, y, xyz = batch
            y_pred = model(x)
            loss_per_sample = torch.nn.functional.mse_loss(y_pred, y, reduction="none")
            loss = loss_per_sample.mean(dim=(1, 2, 3, 4))

            inputs.append(x)
            predictions.append(y_pred)
            targets.append(y)
            losses.append(loss)
            xyzs.append(xyz)

    return [torch.cat(t) for t in [inputs, targets, predictions, xyzs, losses]]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_eval_steps: int,
):

    with torch.no_grad():
        model.eval()
        metrics = torchmetrics.MeanSquaredError().to(device)
        for i in range(num_eval_steps):
            batch = next(dataloader)
            batch = [b.to(device) for b in batch]
            x, y, xyz = batch
            y_pred = model(x)
            metrics.update(y_pred, y)

    return metrics


def train_step(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    metrics: torchmetrics.Metric
) -> torch.Tensor:
    batch = [b.to(device) for b in batch]
    x, y, xyz = batch
    y_pred = model(x)

    metrics.update(y_pred, y)
    
    loss = torch.nn.functional.mse_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Set up writer for logging.
    writer = SummaryWriter(workdir)

    # Save config to workdir.
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Get datasets.
    logging.info("Loading datasets.")
    datasets = input_pipeline_wds.get_datasets(config)

    # Create model.
    logging.info("Creating model.")
    model = create_model(config.model)
    model = model.to(device)

    # Print model summary.
    summary(model, (1, 128, 128, 10), batch_size=config.batch_size)

    # Create optimizer.
    optimizer = torch_utils.create_optimizer(config)(model.parameters(), lr=config.learning_rate)
    
    # Create scheduler.
    # TODO

    # Create hooks.
    # Logging
    log_hook = torch_hooks.TrainMetricsLoggingHook(writer)
    train_metrics = torchmetrics.MeanSquaredError().to(device)

    # Checkpointing
    checkpoint_path = os.path.join(workdir, "checkpoints")
    checkpoint_hook = torch_hooks.CheckpointHook(checkpoint_path, max_to_keep=1)
    step, best_validation_loss = checkpoint_hook.restore_or_init(model, optimizer)

    # Create dict for storing best model
    best_state = torch_train_state.State(
        best_model=model.state_dict(),
        best_optimizer=optimizer.state_dict(),
        step_for_best_model=step,
        metrics_for_best_model=best_validation_loss
    )

    # Evaluation hook
    eval_hook = torch_hooks.EvaluationHook(
        lambda model: evaluate_model(
            model,
            datasets['val'],
            config.num_eval_steps
        ),
        writer
    )

    # Prediction hook
    prediction_hook = torch_hooks.PredictionHook(
        workdir=workdir,
        predict_fn=lambda model, num_batches: predict_model(
            model,
            datasets['val'],
            num_batches
        ),
        peak_threshold=config.peak_threshold,
    )


    # Training loop
    logging.info(f"Starting training at step {step} / {config.num_train_steps}.")

    for step in range(step, config.num_train_steps):

        if step % config.eval_every_steps == 0:
            logging.info(f"Step {step}: Evaluating model.")
            best_state, val_loss = eval_hook(model, optimizer, step, best_state)
            checkpoint_hook(model, optimizer, step, val_loss, best_state)

        if step % config.predict_every_steps == 0:
            logging.info(f"Step {step}: Predicting model.")
            prediction_hook(model, config.predict_num_batches, step, final=False)

        # Get batch
        try:
            t0 = time.perf_counter()
            batch = next(datasets['train'])
            logging.log_first_n(
                logging.INFO,
                f"Loaded batch in {time.perf_counter() - t0:.2f} seconds.",
                20
            )
        except StopIteration:
            logging.info(
                "End of dataset. This should not happen since dataset is infinite."
            )
            break

        # Train model
        model.train()
        loss = train_step(model, optimizer, batch, train_metrics)

        logging.log_first_n(
            logging.INFO,
            f"Train step complete: {(time.perf_counter() - t0)*1e3:.2f} ms",
            20
        )

        # Log metrics
        if step % config.log_every_steps == 0:
            train_metrics = log_hook(train_metrics, step)
