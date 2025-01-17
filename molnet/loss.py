import jax
import jax.numpy as jnp

from operator import getitem

from optax import losses

from typing import Callable


def safe_log(x):
    return jnp.log(jnp.where(x == 0, 1e-9, x))


def mse(y_pred, y_true):
    """
    Mean Squared Error.

    Args:
    - y_pred `jnp.ndarray`: predicted values
    - y_true `jnp.ndarray`: true values

    Returns:
    jnp.ndarray, mean squared error.     
    """
    return jnp.mean((y_true - y_pred) ** 2)


def mae(y_pred, y_true):
    """
    Mean Absolute Error.

    Args:
    - y_pred `jnp.ndarray`: predicted values
    - y_true `jnp.ndarray`: true values

    Returns:
    jnp.ndarray, mean absolute error.     
    """
    return jnp.mean(jnp.abs(y_true - y_pred))


def kl_divergence(logits, targets):
    """
    Kullback-Leibler Divergence.

    Args:
    - y_pred `jnp.ndarray`: predicted values
    - y_true `jnp.ndarray`: true values

    Returns:
    jnp.ndarray, Kullback-Leibler divergence.     
    """

    # Substract the maximum value to avoid numerical instability.
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits_max = jax.lax.stop_gradient(logits_max)
    logits = logits - logits_max

    # Compute CE loss
    loss = -(targets * logits).sum(axis=-1)
    loss = loss + jnp.log(
        jnp.sum(
            jnp.exp(logits),
        axis=-1)
    )

    # Compute self-entropy
    lower_bound = -(targets * safe_log(targets)).sum(axis=-1)
    lower_bound = jax.lax.stop_gradient(lower_bound)

    # Substract self-entropy
    loss = loss - lower_bound

    return jnp.mean(loss)

def cross_entropy_loss(logits, labels):
    """
    Computes the softmax cross-entropy loss for multi-class semantic segmentation in 3D.
    
    Args:
        logits: [D, H, W, num_classes] - Raw model outputs (logits).
        labels: [D, H, W] - Integer class labels for each voxel.
        
    Returns:
        Mean cross-entropy loss.
    """
    num_classes = logits.shape[-1]
    logits = logits.reshape(-1, num_classes)  # Flatten to [N, num_classes]
    labels = labels.flatten()                 # Flatten to [N]

    # Apply log-softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log probabilities of the correct class using advanced indexing
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)

    return loss.mean()


def get_loss_function(loss_fn: str) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Get the loss function based on the name.

    Args:
    - loss_fn `str`: loss function name

    Returns:

    `Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]`: loss function.
    """
    if loss_fn.lower() == "mse":
        return mse
    elif loss_fn.lower() == "mae":
        return mae
    elif loss_fn.lower() == "kl_divergence":
        return kl_divergence
    elif loss_fn.lower() == "cross_entropy":
        return cross_entropy_loss
    elif loss_fn.lower() == "focal_loss":
        raise NotImplementedError("Focal loss not implemented.")
    else:
        raise ValueError(f"Loss function {loss_fn} not supported.")