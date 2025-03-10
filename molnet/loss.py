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

def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    """
    Computes the focal loss for multi-class semantic segmentation in 3D.
    
    Args:
        logits: [batch, D, H, W, num_classes] - Raw model outputs (logits).
        labels: [batch, D, H, W] - Integer class labels for each voxel.
        gamma: float - Focal loss gamma parameter.
        alpha: float - Focal loss alpha parameter.
        
    Returns:
        Mean focal loss.
    """
    num_classes = logits.shape[-1]
    logits = logits.reshape(-1, num_classes)  # Flatten to [N, num_classes]
    labels = labels.flatten()                 # Flatten to [N, ]

    # Apply softmax for numerical stability
    probs = jax.nn.softmax(logits, axis=-1)

    # Gather probabilities of the correct class using advanced indexing
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    probs = jnp.sum(one_hot_labels * probs, axis=-1)

    # Compute focal loss
    loss = -alpha * (1 - probs)**gamma * jnp.log(probs)

    return loss.mean()

def dice_loss(logits, labels, epsilon=1e-9):
    """
    Computes the Dice loss for multi-class semantic segmentation in 3D.
    
    Args:
        logits: [batch, D, H, W, num_classes] - Raw model outputs (logits).
        labels: [batch, D, H, W] - Integer class labels for each voxel.
        epsilon: float - Small constant to avoid division by zero.
        
    Returns:
        Mean Dice loss.
    """
    num_classes = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)  # Convert logits to probabilities
    labels_one_hot = jax.nn.one_hot(labels, num_classes)  # Convert labels to one-hot
    
    # Compute true positive, false positive, and false negative
    tp = jnp.sum(probs * labels_one_hot, axis=(0, 1, 2, 3))
    fp = jnp.sum(probs * (1 - labels_one_hot), axis=(0, 1, 2, 3))
    fn = jnp.sum((1 - probs) * labels_one_hot, axis=(0, 1, 2, 3))

    # Compute Dice coefficient
    nom = 2 * tp + epsilon
    denom = 2 * tp + fp + fn + epsilon
    dc = nom / jnp.clip(denom, epsilon, None)
    dc = jnp.mean(dc)

    return  1 - dc


def get_loss_function(
    loss_fn: str,
    loss_kwargs: dict = None
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Get the loss function based on the name.

    Args:
        - loss_fn `str`: loss function name
        - loss_kwargs `dict`: loss function keyword arguments

    Returns:

    `Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]`: loss function.
    """
    loss_fn_name = loss_fn.lower()
    dc_coef = loss_kwargs.get("dc_coef", 1.0)
    ce_coef = loss_kwargs.get("ce_coef", 1.0)

    if loss_fn_name == "mse":
        return mse
    elif loss_fn_name == "mae":
        return mae
    elif loss_fn_name == "kl_divergence":
        return kl_divergence
    elif loss_fn_name == "cross_entropy":
        return cross_entropy_loss
    elif loss_fn_name == "focal_loss":
        return focal_loss
    elif loss_fn_name == "dice_loss":
        return dice_loss
    elif loss_fn_name == "dc_and_ce":
        def dc_and_ce(logits, labels):
            dc_loss = dice_loss(logits, labels)
            ce_loss = cross_entropy_loss(logits, labels)
            return dc_coef * dc_loss + ce_coef * ce_loss, (dc_loss, ce_loss)
        return dc_and_ce
    else:
        raise ValueError(f"Loss function {loss_fn} not supported.")