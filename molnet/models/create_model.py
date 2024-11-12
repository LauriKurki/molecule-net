import jax
import jax.numpy as jnp
import flax.linen as lnn
import torch.nn as tnn

import ml_collections

from molnet.models import UNet, AttentionUNet

from typing import Callable, Union


def get_activation(
    activation: str,
    code: str
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the activation function based on the name and code."""
    if activation.lower() == "relu":
        return lnn.relu if code == "jax" else tnn.ReLU
    elif activation.lower() == "sigmoid":
        return lnn.sigmoid if code == "jax" else tnn.Sigmoid
    elif activation.lower() == "tanh":
        return lnn.tanh if code == "jax" else tnn.Tanh
    elif activation.lower() == "leaky_relu":
        return lnn.leaky_relu if code == "jax" else tnn.LeakyReLU
    elif activation.lower() == "gelu":
        return lnn.gelu if code == "jax" else tnn.GELU
    else:
        raise ValueError(f"Activation {activation} not supported.")



def create_model(
    config: ml_collections.ConfigDict
) -> Union[lnn.Module, tnn.Module]:
    """Create a model based on the configuration."""
    if config.model_name.lower() == "unet":
        model = UNet(
            output_channels=config.output_channels,
            filters=config.filters,
            kernel_size=config.kernel_size
        )
    elif config.model_name.lower() == "attention-unet":
        model = AttentionUNet(
            output_channels=config.output_channels,
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            attention_channels=config.attention_channels,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,            
            conv_activation=get_activation(config.conv_activation),
            attention_activation=get_activation(config.attention_activation),
            return_attention_maps=config.return_attention_maps
        )

    
    return model