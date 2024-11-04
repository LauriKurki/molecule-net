import ml_collections
import optax

def create_optimizer(
    config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:
    """Create the optimizer."""

    if config.get("learning_rate_schedule") is not None:
        if config.learning_rate_schedule == "constant":
            learning_rate_or_schedule = optax.constant_schedule(config.learning_rate)
        elif config.learning_rate_schedule == "sgdr":
            num_cycles = (
                1
                + config.num_train_steps
                // config.learning_rate_schedule_kwargs.decay_steps
            )
            learning_rate_or_schedule = optax.sgdr_schedule(
                cosine_kwargs=(
                    config.learning_rate_schedule_kwargs for _ in range(num_cycles)
                )
            )
    else:
        learning_rate_or_schedule = config.learning_rate

    if config.optimizer == "adam":
        tx = optax.adam(learning_rate=learning_rate_or_schedule)
    elif config.optimizer == "adamw":
        tx = optax.adamw(learning_rate=learning_rate_or_schedule)
    elif config.optimizer == "sgd":
        tx = optax.sgd(learning_rate=learning_rate_or_schedule)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")
    
    return tx
