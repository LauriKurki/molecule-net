import flax
import flax.struct
from flax.training import train_state

from clu import metrics
from typing import Any, Dict, Optional


class TrainState(train_state.TrainState):
    batch_stats: Any
    best_params: Any
    step_for_best_params: int
    metrics_for_best_params: metrics.Collection

    def get_step(self) -> int:
        try:
            return int(self.step)
        except TypeError:
            return int(self.step[0])

class EvaluationState(train_state.TrainState):
    batch_stats: Any
