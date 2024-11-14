from dataclasses import dataclass
from typing import Any

from torchmetrics import MetricCollection

@dataclass
class State:
    best_model: Any
    best_optimizer: Any
    step_for_best_model: int
    metrics_for_best_model: float
