import flax
import flax.struct
from flax.training import train_state

from clu import metrics
from typing import Any, Dict, Optional


class TrainState(train_state.TrainState):
    batch_stats: Any
    best_params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=False)
    step_for_best_params: int
    metrics_for_best_params: Optional[
        Dict[str, metrics.Collection]
    ] = flax.struct.field(pytree_node=False)
    train_metrics: metrics.Collection = flax.struct.field(pytree_node=False)

    def get_step(self) -> int:
        try:
            return int(self.step)
        except TypeError:
            return int(self.step[0])
