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

#@functools.partial(jax.pmap, axis_name="device")
@jax.jit
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
        z_slices = preds.shape[-2]
        batch_loss = loss.mse(
            preds,
            batch["atom_map"][..., -z_slices:, :]
        )

        return batch_loss, (preds, updates)

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (batch_loss, (_, updates)), grads = grad_fn(state.params)

    batch_metrics = Metrics.single_from_model_output(
        loss=batch_loss,
    )

    # Update parameters
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=updates["batch_stats"],)

    return new_state, batch_metrics


#@functools.partial(jax.pmap, axis_name="device")
@jax.jit
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
    preds_z = preds.shape[-2]
    batch_loss = loss.mse(
        preds,
        batch["atom_map"][..., -preds_z:, :]
    )
    return Metrics.single_from_model_output(
        loss=batch_loss,        
    )


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Iterator[Dict[str, Any]]],
    num_eval_steps: int,
):
    """Evaluate over all datasets."""

    eval_metrics = {}
    for split, data_iterator in datasets.items():
        split_metrics = Metrics.empty()
        #split_metrics = flax.jax_utils.replicate(split_metrics)

        # Loop over graphs.
        for step in range(num_eval_steps):
            #batch = next(device_batch(data_iterator))
            batch = next(data_iterator)
           
            # Compute metrics for this batch.
            batch_metrics = eval_step(state, batch)
            split_metrics = split_metrics.merge(batch_metrics)

        #split_metrics = flax.jax_utils.unreplicate(split_metrics)
        eval_metrics[split + "_eval"] = split_metrics

    return eval_metrics            


def train(
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
    rng, data_rng = jax.random.split(rng)
    datasets = input_pipeline.get_datasets(data_rng, config)
    train_ds = datasets["train"]

    # Create model
    logging.info("Creating model.")
    x_init = next(train_ds)['images']
    rng, init_rng = jax.random.split(rng)
    model = create_model(config)
    
    variables = model.init(init_rng, x_init, training=True)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

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
        train_metrics=Metrics.empty(),
    )
    print(state.train_metrics)
    
    # Create hooks
    logging.info("Creating hooks.")
    log_hook = hooks.LogTrainingMetricsHook(writer)

    # Training loop
    logging.info("Starting training loop.")

    for step in range(config.num_train_steps):

        #if step % config.log_every_steps == 0:
        #    log_hook(state)

        batch = next(train_ds)
        state, batch_metrics = train_step(state, batch)
        print(batch_metrics)
        
        state = state.replace(
            train_metrics=state.train_metrics.merge(batch_metrics)
        )
        #log_hook.is_empty = False

    return state