from absl import logging

from torch import optim
import ml_collections


def create_optimizer(config: ml_collections.ConfigDict):
    if config.optimizer == "adam":
        return optim.Adam
    elif config.optimizer == "adamw":
        return optim.AdamW
    elif config.optimizer == "sgd":
        return optim.SGD
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")   


def create_scheduler(
    opt: optim.Optimizer,
    config: ml_collections.ConfigDict,
) -> optim.lr_scheduler.LRScheduler:
    """Create the learning rate scheduler. Not sure if this works."""
    
    if config.learning_rate_schedule == "constant":
        return optim.lr_scheduler.LambdaLR(
            opt, lambda x: 1.0
        )
    elif config.learning_rate_schedule == "sgdr":
        return optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=config.num_train_steps
        )
    else:
        return None


def print_summary(model):
    summary = {}
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total}")
    for name, module in model.named_children():
        parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary[name] = parameters

        print(f"{name}: {parameters}")


    return summary