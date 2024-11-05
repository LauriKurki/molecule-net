import jax
import jax.numpy as jnp
import flax.linen as nn

import ml_collections

from molnet.models import UNet, AttentionUNet

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
            output_channels=config.output_channels,
            channels=config.channels,
            attention_channels=config.attention_channels,
            kernel_size=config.kernel_size,
            return_attention_maps=config.return_attention_maps
        )

    
    return model