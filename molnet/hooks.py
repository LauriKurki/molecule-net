import os
import pickle
from typing import Any, Dict, Callable

import jax
import jax.numpy as jnp

from absl import logging

import flax
from dataclasses import dataclass
from clu import metric_writers, checkpoint
import flax.jax_utils

from molnet import train_state, train

@dataclass
class CheckpointHook:
    checkpoint_path: str
    max_keep: int

    def __init__(self, checkpoint_path: str, max_keep: int):
        self.checkpoint_path = checkpoint_path
        self.max_keep = max_keep
        self.ckpt = checkpoint.Checkpoint(
            self.checkpoint_path,
            max_to_keep=self.max_keep
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
class LogTrainMetricsHook:
    writer: metric_writers.SummaryWriter
    is_empty: bool = True

    def __call__(
        self,
        state: train_state.TrainState
    ):
        # Unreplicate from all devices
        train_metrics = state.train_metrics
        #train_metrics = flax.jax_utils.unreplicate(state.train_metrics)

        if not self.is_empty:
            # Log the metrics
            train_metrics = train_metrics.compute()
            self.writer.write_scalars(
                step=state.get_step(),
                scalars=add_prefix_to_keys(train_metrics, "train")
            )
            state = state.replace(
                train_metrics=train.Metrics.empty()
                #train_metrics=flax.jax_utils.replicate(train.Metrics.empty())
            )
            self.is_empty = True

        self.writer.flush()

        return state

@dataclass
class EvaluateModelHook:
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
            min_val_loss = state.metrics_for_best_params["val_eval"]["loss"]
        except (AttributeError, KeyError):
            logging.info("No best state found yet.")
            min_val_loss = float("inf")

        if jnp.all(eval_metrics["val_eval"]["loss"] < min_val_loss):
            state = state.replace(
                best_params=state.params,
                metrics_for_best_params=eval_metrics,
                #metrics_for_best_params=flax.jax_utils.replicate(eval_metrics),
                step_for_best_params=state.step,
            )
            logging.info("New best state found at step %d.", state.get_step())

        return state
