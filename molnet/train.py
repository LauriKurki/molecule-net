import os
import time
import functools

import flax.jax_utils
import flax.jax_utils
import flax.jax_utils
import flax.struct
import jax
import jax.numpy as jnp
import flax
import chex
import ml_collections
import yaml

from absl import logging

from clu import (
    checkpoint,
    metrics,
    metric_writers,
    parameter_overview,
    periodic_actions
)

from molnet import utils, train_state, hooks, loss
from molnet.data import input_pipeline
from molnet.models import create_model

from typing import Any, Dict, Iterator, Tuple


@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss") # type: ignore


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}/{key}": val for key, val in result.items()}


def device_batch(
    batch_iterator: Iterator[Dict[str, Any]]
):
    """Batches a set of inputs to the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, b in enumerate(batch_iterator):
        if idx % num_devices == num_devices - 1:
            batch.append(b)
            batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *batch)
            yield batch

            batch = []
        else:
            batch.append(b)


def batch_to_numpy(batch_tuple):
    batch_tuple = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch_tuple)
    batch = {
        "images": batch_tuple[0],
        "atom_map": batch_tuple[1],
        "xyz": batch_tuple[2]
    }
    return batch


@functools.partial(jax.pmap, axis_name="device")
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, Any],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Train step."""

    def loss_fn(params):
        preds, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch["images"],
            training=True,
            mutable='batch_stats',
        )
        batch_loss = loss.mse(
            preds,
            batch["atom_map"]
        )

        return batch_loss, (preds, updates)

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (batch_loss, (_, updates)), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="device")
    batch_stats = jax.lax.pmean(updates["batch_stats"], axis_name="device")

    batch_metrics = Metrics.gather_from_model_output(
        axis_name="device",
        loss=batch_loss,
    )

    # Update parameters
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=batch_stats,
    )

    return new_state, batch_metrics


@functools.partial(jax.pmap, axis_name="device")
def eval_step(
    state: train_state.TrainState,
    batch: Dict[str, Any],
):
    """Evaluation step."""
    preds = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch["images"],
        training=False
    )
    batch_loss = loss.mse(
        preds,
        batch["atom_map"]
    )
    return Metrics.gather_from_model_output(
        axis_name="device",
        loss=batch_loss,        
    )


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Iterator[Dict[str, Any]]],
    num_eval_steps: int,
):
    """Evaluate over all datasets."""

    eval_metrics = {}
    for split, dataloader in datasets.items():
        split_metrics = Metrics.empty()
        split_metrics = flax.jax_utils.replicate(split_metrics)

        # Loop over graphs.
        for step in range(num_eval_steps):
            batch = next(device_batch(dataloader))
            #batch = batch_to_numpy(batch)
           
            # Compute metrics for this batch.
            batch_metrics = eval_step(state, batch)
            split_metrics = split_metrics.merge(batch_metrics)

        split_metrics = flax.jax_utils.unreplicate(split_metrics)
        eval_metrics[split + "_eval"] = split_metrics

    return eval_metrics


@functools.partial(jax.pmap, axis_name="device")
def predict_step(state, batch):
    inputs, targets, xyzs = batch['images'], batch['atom_map'], batch['xyz']
    preds = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
        training=False,
    )
    loss_by_image = jnp.mean(
        (preds - targets) ** 2,
        axis=(1, 2, 3, 4),
    )
    return inputs, targets, preds, xyzs, loss_by_image

def predict_with_state(state, dataloader, num_batches):
    losses = []
    preds = []
    inputs = []
    targets = []
    xyzs = []
    
    for i in range(num_batches):
        batch = next(device_batch(dataloader))
        #batch = batch_to_numpy(batch)

        (
            batch_inputs, batch_targets, batch_preds, batch_xyzs, batch_loss
        ) = predict_step(
            state,
            batch
        )

        device_size = batch_inputs.shape[0]
        local_batch_size = batch_inputs.shape[1]
        # Reshape to [num_devices * local_batch_size, ...]
        batch_inputs = jnp.reshape(batch_inputs, (device_size * local_batch_size, *batch_inputs.shape[2:]))
        batch_targets = jnp.reshape(batch_targets, (device_size * local_batch_size, *batch_targets.shape[2:]))
        batch_preds = jnp.reshape(batch_preds, (device_size * local_batch_size, *batch_preds.shape[2:]))
        batch_xyzs = jnp.reshape(batch_xyzs, (device_size * local_batch_size, *batch_xyzs.shape[2:]))
        batch_loss = jnp.reshape(batch_loss, (device_size * local_batch_size,))

        inputs.append(batch_inputs)
        targets.append(batch_targets)
        preds.append(batch_preds)
        xyzs.append(batch_xyzs)
        losses.append(batch_loss)

    inputs = jnp.concatenate(inputs)
    targets = jnp.concatenate(targets)
    preds = jnp.concatenate(preds)
    xyzs = jnp.concatenate(xyzs)
    losses = jnp.concatenate(losses)

    return inputs, targets, preds, xyzs, losses


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
) -> None:

    # Create writer for logs
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Save config to workdir
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Get datasets
    logging.info("Loading datasets.")
    rng = jax.random.PRNGKey(config.rng_seed)
    #rng, data_rng = jax.random.split(rng)
    #datasets = input_pipeline.get_datasets(data_rng, config)
    datasets = input_pipeline.get_datasets(config)
    train_ds = datasets["train"]

    # Create model
    logging.info("Creating model.")
    batch = next(train_ds)
    rng, init_rng = jax.random.split(rng)
    model = create_model(config.model)

    variables = model.init(init_rng, batch["images"], training=True)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    parameter_overview.log_parameter_overview(params)

    # Create optimizer
    tx = utils.create_optimizer(config)

    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        best_params=params,
        step_for_best_params=0,
        metrics_for_best_params={},
    )
    
    # Create hooks
    logging.info("Creating hooks.")
    
    # Logging
    log_hook = hooks.LogTrainingMetricsHook(writer)
    train_metrics = flax.jax_utils.replicate(Metrics.empty())

    # Checkpointing
    checkpoint_path = os.path.join(workdir, "checkpoints")
    checkpoint_hook = hooks.CheckpointHook(checkpoint_path, max_to_keep=1)
    state = checkpoint_hook.restore_or_init(state)
    initial_step = state.get_step()

    state = flax.jax_utils.replicate(state)

    # Evaluation
    evaluate_model_hook = hooks.EvaluationHook(
        evaluate_model_fn=lambda state: evaluate_model(
            state,
            datasets,
            config.num_eval_steps,
        ),
        writer=writer,
    )

    # Prediction
    prediction_hook = hooks.PredictionHook(
        workdir=workdir,
        predict_fn=lambda state, num_batches: predict_with_state(
            state,
            datasets["val"],
            num_batches,
        ),
        peak_threshold=config.peak_threshold,
        writer=writer,
    )

    profiler = periodic_actions.Profile(
        logdir=os.path.join(workdir, "profile"),
        every_secs=10800        
    )
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps,
        writer=writer,
        every_secs=300,
    )

    # Training loop
    logging.info(f"Starting training loop at step {initial_step} out of {config.num_train_steps}.")
    for step in range(initial_step, config.num_train_steps+1):

        if step % config.eval_every_steps == 0:
            logging.info("Evaluating model.")
            state = evaluate_model_hook(state)
            checkpoint_hook(state)

        if step % config.predict_every_steps == 0:
            logging.info(f"Predicting with current state at step {step}.")
            prediction_hook(
                state,
                num_batches=config.predict_num_batches,
                final=False
            )

        try:
            t0 = time.perf_counter()
            batch = next(device_batch(train_ds))
            #batch = batch_to_numpy(batch)

            logging.log_first_n(
                logging.INFO,
                f"Time to load batch: {(time.perf_counter() - t0)*1e3:.2f} ms",
                10,
            )
        except StopIteration:
            logging.info("No more data, exiting training loop.")
            break

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, batch_metrics = train_step(state, batch)
            train_metrics = train_metrics.merge(batch_metrics)

        logging.log_first_n(
            logging.INFO,
            f"Train step complete: {(time.perf_counter() - t0)*1e3:.2f} ms",
            10,
        )

        if step % config.log_every_steps == 0:
            train_metrics = log_hook(train_metrics, step)

        #profiler(step)
        report_progress(step)

    # Do final predictions
    logging.info("Predicting with final state after training.")
    prediction_hook(
        state,
        num_batches=config.predict_num_batches_at_end_of_training,
        final=True
    )
        
    return state

