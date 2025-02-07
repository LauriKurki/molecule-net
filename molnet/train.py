import os
import time
import functools

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

from configs import root_dirs
from molnet import utils, train_state, hooks, loss
from molnet.data import input_pipeline_online, input_pipeline_water
from molnet.models import create_model

from typing import Any, Dict, Iterator, Tuple, Callable


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


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, Any],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Train step."""

    def loss_wrapper(params):
        preds, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch["images"],
            training=True,
            mutable='batch_stats',
        )
        batch_loss = loss_fn(
            preds,
            batch["atom_map"]
        )

        return batch_loss, (preds, updates)

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
    (batch_loss, (_, updates)), grads = grad_fn(state.params)

    batch_metrics = Metrics.single_from_model_output(
        loss=batch_loss,
    )

    # Update parameters
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=updates["batch_stats"],)

    return new_state, batch_metrics


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(
    state: train_state.TrainState,
    batch: Dict[str, Any],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
):
    """Evaluation step."""
    preds = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch["images"],
        training=False
    )
    batch_loss = loss_fn(
        preds,
        batch["atom_map"]
    )
    return Metrics.single_from_model_output(
        loss=batch_loss,        
    )


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Iterator[Dict[str, Any]]],
    num_eval_steps: int,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
):
    """Evaluate over all datasets."""

    eval_metrics = {}
    for split, dataloader in datasets.items():
        split_metrics = Metrics.empty()
        # Loop over graphs.
        for step in range(num_eval_steps):
            #batch = next(device_batch(data_iterator))
            batch = next(dataloader)
           
            # Compute metrics for this batch.
            batch_metrics = eval_step(state, batch, loss_fn)
            split_metrics = split_metrics.merge(batch_metrics)

        eval_metrics[split + "_eval"] = split_metrics

    return eval_metrics


@jax.jit
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
        batch = next(dataloader)

        (
            batch_inputs, batch_targets, batch_preds, batch_xyzs, batch_loss
        ) = predict_step(
            state,
            batch
        )

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

    # Set root dir
    config.root_dir = root_dirs.get_root_dir(config.dataset)

    # Save config to workdir
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Get datasets
    logging.info("Loading datasets.")
    rng = jax.random.PRNGKey(config.rng_seed)
    #rng, data_rng = jax.random.split(rng)
    #datasets = input_pipeline.get_datasets(data_rng, config)
    if "water" in config.dataset:
        datasets = input_pipeline_water.get_datasets(config)
    else:
        datasets = input_pipeline_online.get_datasets(config)
    train_ds = datasets["train"]

    # Create model
    logging.info("Creating model.")
    rng, init_rng = jax.random.split(rng)
    model = create_model(config.model)

    dummy_input = next(train_ds)["images"]
    variables = model.init(init_rng, dummy_input, training=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    parameter_overview.log_parameter_overview(params)

    # Create optimizer
    tx = utils.create_optimizer(config)
    # Create loss function
    loss_fn = loss.get_loss_function(config.loss_fn)

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
    train_metrics = Metrics.empty()

    # Checkpointing
    checkpoint_path = os.path.join(workdir, "checkpoints")
    checkpoint_hook = hooks.CheckpointHook(checkpoint_path, max_to_keep=1)
    state = checkpoint_hook.restore_or_init(state)
    initial_step = state.get_step()

    # Evaluation
    evaluate_model_hook = hooks.EvaluationHook(
        evaluate_model_fn=lambda state: evaluate_model(
            state,
            datasets,
            config.num_eval_steps,
            loss_fn
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
        preds_are_logits=config.model.output_activation == "log-softmax",
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
            batch = next(train_ds)

            logging.log_first_n(
                logging.INFO,
                f"Time to load batch: {(time.perf_counter() - t0)*1e3:.2f} ms",
                10,
            )
        except StopIteration:
            logging.info("No more data, exiting training loop.")
            break

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, batch_metrics = train_step(state, batch, loss_fn)
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

