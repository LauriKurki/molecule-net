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
class LogTrainingMetricsHook:
    writer: metric_writers.SummaryWriter
    is_empty: bool = True
    prefix: str = "train"

    def __call__(
        self,
        state: train_state.TrainState
    ):
        
        if not self.is_empty:
            train_metrics = state.train_metrics.compute()

            self.writer.write_scalars(
                state.get_step(),
                add_prefix_to_keys(train_metrics, self.prefix),
            )

            state = state.replace(train_metrics=train.Metrics.empty())
            self.is_empty = True

        self.writer.flush()

        return state
