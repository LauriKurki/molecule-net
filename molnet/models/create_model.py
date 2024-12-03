import jax
import jax.numpy as jnp
import flax.linen as nn

import ml_collections

from molnet.models import UNet, AttentionUNet

from typing import Callable


def get_activation(activation: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the activation function based on the name."""
    if activation is None:
        return None
    elif activation.lower() == "relu":
        return nn.relu
    elif activation.lower() == "sigmoid":
        return nn.sigmoid
    elif activation.lower() == "tanh":
        return nn.tanh
    elif activation.lower() == "leaky_relu":
        return nn.leaky_relu
    elif activation.lower() == "gelu":
        return nn.gelu
    elif activation.lower() == "softmax":
        return nn.softmax
    elif activation.lower() == "log-softmax":
        return nn.log_softmax
    else:
        raise ValueError(f"Activation {activation} not supported.")

def get_dtype(dtype: str) -> jnp.dtype:
    """Get the dtype based on the name."""
    if dtype.lower() == "float32":
        return jnp.float32
    elif dtype.lower() == "float64":
        return jnp.float64
    elif dtype.lower() == "bfloat16":
        return jnp.bfloat16
    else:
        raise ValueError(f"Dtype {dtype} not supported.")


def create_model(
    config: ml_collections.ConfigDict
) -> nn.Module:
    """Create a model based on the configuration."""
    if config.model_name.lower() == "unet":
        model = UNet(
            output_channels=config.output_channels,
            filters=config.filters,
            kernel_size=config.kernel_size
        )
    elif config.model_name.lower() == "attention-unet":
        model = AttentionUNet(
            dtype=get_dtype(config.dtype),
            output_channels=config.output_channels,
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            attention_channels=config.attention_channels,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,            
            conv_activation=get_activation(config.conv_activation),
            attention_activation=get_activation(config.attention_activation),
            output_activation=get_activation(config.output_activation),
            return_attention_maps=config.return_attention_maps
        )

    
    return model