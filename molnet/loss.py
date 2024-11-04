import jax
import jax.numpy as jnp

def mse(y_true, y_pred):
    assert y_true.shape == y_pred.shape, (
        f"y_true.shape={y_true.shape} and y_pred.shape={y_pred.shape} must be the same."
    )
    return jnp.mean((y_true - y_pred) ** 2)
